"""
Main Script to Run Federated Learning System

This script runs the entire FL system with support for:
- Classification using labeled packet sequences
- Time series prediction using GPS and IMU data

Supports client-specific data files from prepare_fl_data_for_run_fl.py.
Sample usage at the project root directory: 
[1] python .\fl-classification\run_fl_system_classification.py --data-dir .\data\train\packets\ --config config_classification --learning-mode classification
[2] python .\fl-classification\run_fl_system_classification.py --data-dir .\data\train\gps-imu\ --config config_regression --learning-mode regression
"""
import subprocess
import time
import sys
import os
import signal
import requests
import glob
import logging
import argparse
import threading
from typing import Optional
#import config_classification as config
#from config_classification import *

from typing import Optional, TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    # Hint to type checkers so that symbols like SERVER_HOST, SERVER_PORT,
    # CHECKPOINT_DIR, etc. are known at analysis time. At runtime the actual
    # config module is selected dynamically in main().
    from config_packets_only import *  # type: ignore[import,unused-wildcard-import]

# Global handle to the active config module; will be set in main() based on
# the --config argument (defaults to "config_packets_only").
config = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Track running processes
processes = []


def cleanup_processes():
    """Clean up all running processes"""
    print("\n\nCleaning up processes...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    print("All processes terminated.")


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print('\n\nInterrupted by user')
    cleanup_processes()
    sys.exit(0)


def wait_for_server(max_retries=30):
    """Wait for server to be ready"""
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/status"
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Server is ready at {url}")
                if 'learning_mode' in status:
                    print(f"  Learning mode: {status['learning_mode']}")
                if 'output_size' in status:
                    print(f"  Output size: {status['output_size']}")
                return True
        except requests.exceptions.Timeout:
            # Server might be starting up, continue waiting
            pass
        except requests.exceptions.ConnectionError:
            # Server not ready yet, continue waiting
            pass
        except Exception as e:
            # Other errors, log but continue
            if i % 10 == 0:
                print(f"  Server check error: {type(e).__name__}")
        time.sleep(1)
        if i % 5 == 0:
            print(f"Waiting for server... ({i}/{max_retries})")
    return False


def get_client_process_status(client_procs):
    """Get status of client processes."""
    status = {}
    for i, proc in enumerate(client_procs, 1):
        client_id = getattr(proc, 'client_id', f"client_{i}")
        if proc.poll() is None:
            status[client_id] = "running"
        else:
            status[client_id] = f"finished (code: {proc.returncode})"
    return status


def stream_process_output(proc, client_id):
    """Stream process output to terminal in real-time."""
    def _stream():
        try:
            print(f"[{client_id}] Output stream started")
            sys.stdout.flush()
            for line in iter(proc.stdout.readline, ''):
                if line:
                    # Print with client ID prefix
                    print(f"[{client_id}] {line.rstrip()}")
                    sys.stdout.flush()  # Ensure immediate output
            print(f"[{client_id}] Output stream ended (process finished)")
            sys.stdout.flush()
        except Exception as e:
            print(f"[{client_id}] Error reading output: {e}")
            sys.stdout.flush()
    
    thread = threading.Thread(target=_stream, daemon=True)
    thread.start()
    return thread


def monitor_training():
    """Monitor training progress by checking server status and client output"""
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/status"
    last_round = -1
    rounds_without_change = 0
    max_rounds_without_change = 60  # Exit if no progress for 60 checks (2 minutes)
    check_count = 0
    last_client_losses = {}  # Track last seen loss for each client
    
    # Adaptive timeout and backoff for busy server
    base_timeout = 30  # Base timeout: 30 seconds
    current_timeout = base_timeout
    consecutive_timeouts = 0
    check_interval = 2  # Base check interval: 2 seconds
    max_timeout = 120  # Maximum timeout: 2 minutes

    print("\n" + "=" * 70)
    print("MONITORING TRAINING PROGRESS")
    print("=" * 70)
    print("(Updates shown when rounds complete)\n")

    try:
        while rounds_without_change < max_rounds_without_change:
            time.sleep(check_interval)
            check_count += 1
            
            # Check if client processes are still alive
            client_procs = processes[1:]  # Skip server
            alive_count = sum(1 for p in client_procs if p.poll() is None)
            if alive_count == 0 and check_count > 5:
                # All clients have finished
                print(f"\n✓ All clients have finished processing")
                break
            
            # Get client process status
            client_status = get_client_process_status(client_procs)
            
            try:
                # Use adaptive timeout - increase if server is busy
                response = requests.get(url, timeout=current_timeout)
                if response.status_code == 200:
                    status = response.json()
                    current_round = status['current_round']

                    # Get per-client loss history if available
                    client_losses = {}
                    if 'loss_history' in status and status['loss_history']:
                        for client_id, losses in status['loss_history'].items():
                            if losses:
                                client_losses[client_id] = losses[-1]  # Latest loss
                    
                    # Also check current round losses
                    if 'current_client_losses' in status and status['current_client_losses']:
                        for client_id, loss in status['current_client_losses'].items():
                            client_losses[client_id] = loss

                    # Only print when a new round completes
                    if current_round > last_round:
                        mode_str = ""
                        if 'learning_mode' in status:
                            mode_str = f", Mode: {status['learning_mode']}"
                        if 'client_modes' in status and status['client_modes']:
                            mode_str += f", Client modes: {status['client_modes']}"
                        
                        print(f"\n{'='*70}")
                        print(f"✓ Round {current_round} complete - "
                              f"Clients: {status['registered_clients']}/{status['total_expected']}, "
                              f"Strategy: {status['aggregation_strategy']}{mode_str}")
                        
                        # Show server operation if it was active
                        if 'current_operation' in status and status['current_operation'] != 'idle':
                            op = status['current_operation']
                            duration = status.get('operation_duration')
                            if duration:
                                print(f"  Server operation: {op} (took {duration:.2f}s)")
                            else:
                                print(f"  Server operation: {op}")
                        
                        # Show per-client progress
                        if client_losses:
                            print(f"\n  Per-Client Loss (Round {current_round}):")
                            for client_id in sorted(client_losses.keys()):
                                loss = client_losses[client_id]
                                prev_loss = last_client_losses.get(client_id)
                                trend = ""
                                if prev_loss is not None:
                                    if loss < prev_loss:
                                        trend = " ↓ (improving)"
                                    elif loss > prev_loss:
                                        trend = " ↑ (worse)"
                                    else:
                                        trend = " → (stable)"
                                proc_status = client_status.get(client_id, "unknown")
                                print(f"    {client_id}: {loss:.6f}{trend} [{proc_status}]")
                                last_client_losses[client_id] = loss
                        else:
                            # Show client status even if no losses yet
                            print(f"\n  Client Status:")
                            for client_id in sorted(client_status.keys()):
                                print(f"    {client_id}: [{client_status[client_id]}]")
                        
                        # Show average loss if available
                        if 'round_losses' in status and status['round_losses']:
                            avg_loss = status['round_losses'][-1]
                            print(f"\n  Average Loss: {avg_loss:.6f}")
                        
                        print(f"{'='*70}\n")
                        last_round = current_round
                        rounds_without_change = 0
                        # Reset timeout on successful check
                        consecutive_timeouts = 0
                        current_timeout = base_timeout
                        check_interval = 2
                    else:
                        rounds_without_change += 1
                        # Show periodic status if no progress
                        if rounds_without_change % 10 == 0:
                            print(f"  Waiting... (Round {current_round}, {alive_count} clients active)")
                            
                            # Show server operation if available
                            if 'current_operation' in status and status['current_operation'] != 'idle':
                                op = status['current_operation']
                                duration = status.get('operation_duration')
                                details = status.get('operation_details', {})
                                if duration:
                                    print(f"  Server: {op} (running for {duration:.1f}s)")
                                else:
                                    print(f"  Server: {op}")
                                if details:
                                    detail_str = ', '.join([f"{k}={v}" for k, v in details.items() if k != 'error'])
                                    if detail_str:
                                        print(f"    Details: {detail_str}")
                            
                            if client_losses:
                                print(f"  Latest losses: {', '.join([f'{cid}: {loss:.4f}' for cid, loss in sorted(client_losses.items())])}")
                            # Show client status
                            if client_status:
                                running = [cid for cid, stat in client_status.items() if 'running' in stat]
                                if running:
                                    print(f"  Active clients: {', '.join(running)}")

            except requests.exceptions.Timeout:
                consecutive_timeouts += 1
                rounds_without_change += 1
                
                # Exponential backoff: increase timeout and check interval
                if consecutive_timeouts > 0:
                    # Increase timeout up to max
                    current_timeout = min(base_timeout * (2 ** min(consecutive_timeouts, 3)), max_timeout)
                    # Increase check interval to reduce load
                    check_interval = min(2 * (1.5 ** min(consecutive_timeouts, 2)), 10)
                
                # Only print message occasionally to avoid spam
                if rounds_without_change % 5 == 0:
                    print(f"  Server timeout (server busy, using {current_timeout}s timeout, checking every {check_interval:.1f}s)...")
                    print(f"    (Server may be aggregating weights, saving checkpoints, or processing client updates)")
            except requests.exceptions.ConnectionError:
                consecutive_timeouts += 1
                rounds_without_change += 1
                # Reset timeout on connection error (different issue)
                if rounds_without_change % 5 == 0:
                    print(f"  Cannot connect to server (check if server is running)...")
            except Exception as e:
                consecutive_timeouts += 1
                rounds_without_change += 1
                if rounds_without_change % 10 == 0:
                    print(f"  Server check failed: {type(e).__name__}: {str(e)[:100]}")

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
        pass


def cleanup_old_checkpoints():
    """Delete old checkpoint files"""
    if os.path.exists(CHECKPOINT_DIR):
        pt_files = glob.glob(f"{CHECKPOINT_DIR}/*.pt")
        if pt_files:
            logger.info(f"Cleaning up {len(pt_files)} old checkpoint files...")
            for pt_file in pt_files:
                os.remove(pt_file)
            logger.info("✓ Old checkpoints deleted")


def determine_learning_mode(
    gps_file: Optional[str],
    imu_file: Optional[str],
    packet_file: Optional[str],
    use_labels: bool
) -> str:
    """Determine learning mode based on available data files."""
    if packet_file and use_labels:
        return 'classification'
    elif packet_file and not use_labels:
        # Packet file without labels - could be regression
        return 'regression'
    elif gps_file or imu_file:
        return 'regression'
    else:
        return 'regression'  # Default


def determine_output_size(learning_mode: str) -> Optional[int]:
    """Determine output size based on learning mode."""
    if learning_mode == 'classification':
        return 1  # Binary classification
    else:
        # Regression: use config or calculate from features
        if hasattr(config, 'MODEL_CONFIG') and 'output_size' in config.MODEL_CONFIG:
            return config.MODEL_CONFIG['output_size']
        elif hasattr(config, 'INPUT_SIZE'):
            return config.INPUT_SIZE
        else:
            # Calculate from features
            gps_features = getattr(config, 'GPS_FEATURES', [])
            imu_features = getattr(config, 'IMU_FEATURES', [])
            packet_features = getattr(config, 'PACKET_FEATURES', [])
            total = len(gps_features) + len(imu_features) + len(packet_features)
            return total if total > 0 else None


def get_client_files(
    client_id: str,
    data_dir: Optional[str],
    gps_file: Optional[str],
    imu_file: Optional[str],
    packet_file: Optional[str]
) -> tuple:
    """Get client-specific file paths."""
    # Extract client number from client_id (e.g., 'client_1' -> 0)
    client_num = None
    try:
        parts = client_id.split('_')
        if len(parts) > 1:
            client_num = int(parts[-1]) - 1  # Convert to 0-indexed
    except (ValueError, IndexError):
        pass

    client_gps_file = gps_file
    client_imu_file = imu_file
    client_packet_file = packet_file

    # If data_dir is provided, try to find client-specific files
    if data_dir and client_num is not None:
        gps_path = os.path.join(data_dir, f"gps_client_{client_num:03d}.csv")
        imu_path = os.path.join(data_dir, f"imu_client_{client_num:03d}.csv")
        packet_path = os.path.join(data_dir, f"packet_client_{client_num:03d}.csv")

        if os.path.exists(gps_path):
            client_gps_file = gps_path
            logger.info(f"  Using client-specific GPS file: {gps_path}")
        if os.path.exists(imu_path):
            client_imu_file = imu_path
            logger.info(f"  Using client-specific IMU file: {imu_path}")
        if os.path.exists(packet_path):
            client_packet_file = packet_path
            logger.info(f"  Using client-specific packet file: {packet_path}")

    return client_gps_file, client_imu_file, client_packet_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run Federated Learning System with Packet Support'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory with client-specific data files from prepare_fl_data_for_run_fl.py'
    )
    parser.add_argument(
        '--gps-file',
        type=str,
        default=None,
        help='GPS CSV file (optional, can use data-dir for client-specific files)'
    )
    parser.add_argument(
        '--imu-file',
        type=str,
        default=None,
        help='IMU CSV file (optional, can use data-dir for client-specific files)'
    )
    parser.add_argument(
        '--packet-file',
        type=str,
        default=None,
        help='Labeled packet CSV file (optional, can use data-dir for client-specific files)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config_classification',
        help='Config module to use (e.g., config, config_classification)'
    )
    
    parser.add_argument(
        '--use-labels',
        action='store_true',
        help='Use packet labels for supervised learning (auto-enabled if packet file found)'
    )
    parser.add_argument(
        '--learning-mode',
        type=str,
        choices=['classification', 'regression'],
        default=None,
        help='Learning mode (None = auto-detect)'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=None,
        help='Model output size (None = auto-detect)'
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load selected config module and expose its UPPERCASE attributes
    # globally so existing code (SERVER_HOST, SERVER_PORT, etc.) works.
    # ------------------------------------------------------------------
    global config
    config_module_name = args.config
    try:
        config = importlib.import_module(config_module_name)
    except ImportError as e:
        print(f"[ERROR] Could not import config module '{config_module_name}': {e}")
        sys.exit(1)

    for _name, _value in vars(config).items():
        if _name.isupper():
            globals()[_name] = _value

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Determine learning mode
    if args.learning_mode:
        learning_mode = args.learning_mode
    else:
        # Use config defaults if available, otherwise determine from files
        gps_file = args.gps_file or getattr(config, 'GPS_FILE', None)
        imu_file = args.imu_file or getattr(config, 'IMU_FILE', None)
        packet_file = args.packet_file or getattr(config, 'PACKET_FILE', None)
        use_labels = args.use_labels or getattr(config, 'USE_LABELS', False)
        learning_mode = determine_learning_mode(gps_file, imu_file, packet_file, use_labels)

    # Determine output size
    if args.output_size:
        output_size = args.output_size
    else:
        output_size = determine_output_size(learning_mode)
    
    # Determine input size
    if hasattr(config, 'INPUT_SIZE'):
        input_size = config.INPUT_SIZE
    else:
        # Calculate from features
        gps_features = getattr(config, 'GPS_FEATURES', [])
        imu_features = getattr(config, 'IMU_FEATURES', [])
        packet_features = getattr(config, 'PACKET_FEATURES', [])
        input_size = len(gps_features) + len(imu_features) + len(packet_features)
        if input_size == 0:
            input_size = None  # Let server auto-detect

    # Print configuration from the selected config module (if available)
    if hasattr(config, "print_config"):
        config.print_config()

    # Print additional configuration
    print("\n" + "=" * 70)
    print("ADDITIONAL CONFIGURATION")
    print("=" * 70)
    print(f"Learning Mode: {learning_mode}")
    if output_size:
        print(f"Output Size: {output_size}")
    if input_size:
        print(f"Input Size: {input_size}")
    if args.data_dir:
        print(f"Data Directory: {args.data_dir}")
    if args.gps_file:
        print(f"GPS File: {args.gps_file}")
    if args.imu_file:
        print(f"IMU File: {args.imu_file}")
    if args.packet_file:
        print(f"Packet File: {args.packet_file}")
    print("=" * 70)

    # Clean up old checkpoints
    cleanup_old_checkpoints()

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("STARTING FEDERATED LEARNING SYSTEM")
    print("=" * 70)

    # Get the directory where this script is located (used for both server and client scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Start server
    print(f"\n🚀 Starting FL Server on {SERVER_HOST}:{SERVER_PORT}...")
    # Use -u flag to ensure unbuffered output for real-time streaming
    server_script = os.path.join(script_dir, 'fl_server_classification.py')
    server_cmd = [
        sys.executable, '-u', server_script,
        '--host', SERVER_HOST,
        '--port', str(SERVER_PORT),
        '--num-clients', str(NUM_CLIENTS),
        '--min-clients', str(MIN_CLIENTS),
        '--aggregation', AGGREGATION_STRATEGY
    ]
    
    if learning_mode:
        server_cmd.extend(['--learning-mode', learning_mode])
    if output_size:
        server_cmd.extend(['--output-size', str(output_size)])
    if input_size:
        server_cmd.extend(['--input-size', str(input_size)])

    # Ensure server picks up the same config module
    server_env = os.environ.copy()
    server_env["FL_CONFIG_MODULE"] = config_module_name
    
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=server_env
    )
    processes.append(server_proc)
    
    # Store process info for output streaming
    server_proc.client_id = "server"
    server_proc.server_cmd = ' '.join(server_cmd)
    
    # Start streaming server output in real-time
    print(f"  Starting output stream for server...")
    server_stream_thread = stream_process_output(server_proc, "server")

    # Wait for server to be ready
    print("\nWaiting for server to start...")
    if not wait_for_server():
        print("❌ Server failed to start!")
        cleanup_processes()
        return

    # Start clients
    print(f"\n🚀 Starting {NUM_CLIENTS} FL Clients...")
    for i in range(1, NUM_CLIENTS + 1):
        client_id = f"client_{i}"
        print(f"   Starting {client_id}...")

        # Get client-specific files
        client_gps, client_imu, client_packet = get_client_files(
            client_id,
            args.data_dir,
            args.gps_file or getattr(config, 'GPS_FILE', None),
            args.imu_file or getattr(config, 'IMU_FILE', None),
            args.packet_file or getattr(config, 'PACKET_FILE', None)
        )

        # Build client command
        # Use -u flag to ensure unbuffered output for real-time streaming
        client_script = os.path.join(script_dir, 'fl_client_classification.py')
        client_cmd = [
            sys.executable, '-u', client_script,
            '--client-id', client_id,
            '--server-url', f'http://{SERVER_HOST}:{SERVER_PORT}'
        ]

        # Add data files if provided
        if client_gps:
            client_cmd.extend(['--gps-file', client_gps])
        if client_imu:
            client_cmd.extend(['--imu-file', client_imu])
        if client_packet:
            client_cmd.extend(['--packet-file', client_packet])
            # Auto-enable labels if packet file is found
            if not args.use_labels:
                client_cmd.append('--use-labels')
        elif args.use_labels:
            client_cmd.append('--use-labels')

        # Add data directory if provided
        if args.data_dir:
            client_cmd.extend(['--data-dir', args.data_dir])

        # Ensure client picks up the same config module
        client_env = os.environ.copy()
        client_env["FL_CONFIG_MODULE"] = config_module_name

        client_proc = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=client_env
        )
        processes.append(client_proc)
        
        # Store process info for later output reading
        client_proc.client_id = client_id
        client_proc.client_cmd = ' '.join(client_cmd)
        
        # Start streaming output in real-time
        print(f"  Starting output stream for {client_id}...")
        stream_thread = stream_process_output(client_proc, client_id)
        client_proc.stream_thread = stream_thread
        
        time.sleep(0.5)

    print(f"\n✓ All {NUM_CLIENTS} clients started!")
    
    # Give clients time to initialize and register
    print("\nWaiting for clients to initialize and register...")
    print("(Client output will be streamed in real-time below)")
    time.sleep(5)
    
    # Check initial client status
    print("\nChecking client process status...")
    client_procs = processes[1:]
    for proc in client_procs:
        client_id = getattr(proc, 'client_id', 'unknown')
        if proc.poll() is None:
            print(f"  ✓ {client_id}: Process running (output streaming enabled)")
        else:
            print(f"  ⚠ {client_id}: Process exited with code {proc.returncode}")
            # Try to read any error output
            try:
                stdout, _ = proc.communicate(timeout=1)
                if stdout:
                    print(f"    Last output: {stdout[-300:]}")
            except:
                pass

    # Monitor training
    monitor_training()

    # Wait for all clients to finish (with timeout)
    print("\n\nWaiting for training to complete...")
    max_wait_time = 300  # Maximum 5 minutes wait per client
    
    # Check server status one more time
    try:
        response = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"Final server status: Round {status['current_round']}, "
                  f"{status['registered_clients']} clients registered")
            if 'round_losses' in status and status['round_losses']:
                print(f"  Average loss: {status['round_losses'][-1]:.6f}")
            if 'loss_history' in status:
                print(f"  Clients with loss history: {list(status['loss_history'].keys())}")
    except requests.exceptions.Timeout:
        print("  Server status check timed out (server may be busy)")
    except Exception as e:
        print(f"  Could not get final server status: {type(e).__name__}")
    
    # Show client output for debugging
    print("\n📋 Client Output Summary:")
    for i, proc in enumerate(processes[1:], 1):
        client_id = getattr(proc, 'client_id', f"client_{i}")
        print(f"\n  {client_id}:")
        proc_status = proc.poll()
        if proc_status is None:
            print(f"    Status: Still running")
        else:
            print(f"    Status: Exited with code {proc_status}")
            # Try to read output (only if process is finished)
            try:
                # Use communicate with timeout to get remaining output
                # Note: communicate() can be called on finished processes
                stdout, stderr = proc.communicate(timeout=2)
                if stdout:
                    # Show last few lines
                    lines = stdout.strip().split('\n')
                    print(f"    Last {min(10, len(lines))} lines of output:")
                    for line in lines[-10:]:
                        if line.strip():
                            print(f"      {line[:100]}")
            except subprocess.TimeoutExpired:
                print(f"    Could not read output (timeout)")
            except Exception as e:
                print(f"    Could not read output: {type(e).__name__}")
    
    for i, proc in enumerate(processes[1:], 1):  # Skip server
        client_id = f"client_{i}"
        
        if proc.poll() is None:  # Process still running
            print(f"Waiting for {client_id} to complete (max {max_wait_time}s)...")
            try:
                # Wait with timeout
                proc.wait(timeout=max_wait_time)
                return_code = proc.returncode
                if return_code == 0:
                    print(f"✓ {client_id} completed successfully")
                else:
                    print(f"⚠ {client_id} exited with code {return_code}")
            except subprocess.TimeoutExpired:
                print(f"⚠ {client_id} timed out after {max_wait_time}s - terminating")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print(f"✓ {client_id} terminated gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠ {client_id} did not terminate - killing")
                    proc.kill()
                    proc.wait()
                    print(f"✓ {client_id} killed")
        else:
            # Process already finished
            return_code = proc.returncode
            if return_code == 0:
                print(f"✓ {client_id} already completed")
            else:
                print(f"⚠ {client_id} exited earlier with code {return_code}")

    # Give server time to save final checkpoint
    time.sleep(2)

    # Request server to save loss plot
    try:
        print("\n📊 Generating loss plot...")
        response = requests.post(f"http://{SERVER_HOST}:{SERVER_PORT}/save_plot", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Loss plot saved: {result['plot_path']}")
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)

    # Print checkpoints
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    if checkpoints:
        print(f"\n📁 Saved Checkpoints ({len(checkpoints)}):")
        for ckpt in sorted(checkpoints):
            print(f"   {CHECKPOINT_DIR}/{ckpt}")

    # Print plot location
    plot_path = f"{PLOT_DIR}/training_loss.png"
    if os.path.exists(plot_path):
        print(f"\n📈 Training Loss Plot:")
        print(f"   {plot_path}")

    print(f"\n✓ Training artifacts saved successfully!")
    print(f"  Learning mode: {learning_mode}")
    if output_size:
        print(f"  Output size: {output_size}")

    # Cleanup
    cleanup_processes()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup_processes()
        sys.exit(1)
