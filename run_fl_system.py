"""
Main Script to Run Federated Learning System

This script runs the entire FL system with all configured parameters.
Simply run: python run_fl_system.py
"""
import subprocess
import time
import sys
import os
import signal
import requests
import glob
import logging
from config import *

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
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print(f"✓ Server is ready at {url}")
                return True
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"Waiting for server... ({i}/{max_retries})")
    return False


def monitor_training():
    """Monitor training progress by checking server status"""
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/status"
    last_round = -1
    rounds_without_change = 0
    max_rounds_without_change = 30  # Exit if no progress for 30 checks

    print("\n" + "=" * 70)
    print("MONITORING TRAINING PROGRESS")
    print("=" * 70)
    print("(Updates shown when rounds complete)\n")

    try:
        while rounds_without_change < max_rounds_without_change:
            time.sleep(2)
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    current_round = status['current_round']

                    # Only print when a new round completes
                    if current_round > last_round:
                        print(f"✓ Round {current_round} complete - "
                              f"Clients: {status['registered_clients']}/{status['total_expected']}, "
                              f"Strategy: {status['aggregation_strategy']}")
                        last_round = current_round
                        rounds_without_change = 0
                    else:
                        rounds_without_change += 1

            except:
                rounds_without_change += 1

    except KeyboardInterrupt:
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


def main():
    """Main entry point"""

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Print configuration
    print_config()

    # Clean up old checkpoints
    cleanup_old_checkpoints()

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("STARTING FEDERATED LEARNING SYSTEM")
    print("=" * 70)

    # Start server
    print(f"\n🚀 Starting FL Server on {SERVER_HOST}:{SERVER_PORT}...")
    server_cmd = [
        sys.executable, 'fl_server.py',
        '--host', SERVER_HOST,
        '--port', str(SERVER_PORT),
        '--num-clients', str(NUM_CLIENTS),
        '--min-clients', str(MIN_CLIENTS),
        '--aggregation', AGGREGATION_STRATEGY
    ]

    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(server_proc)

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

        client_cmd = [
            sys.executable, 'fl_client.py',
            '--client-id', client_id,
            '--gps-file', GPS_FILE,
            '--imu-file', IMU_FILE,
            '--server-url', f'http://{SERVER_HOST}:{SERVER_PORT}'
        ]

        client_proc = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(client_proc)
        time.sleep(0.5)

    print(f"\n✓ All {NUM_CLIENTS} clients started!")

    # Monitor training
    monitor_training()

    # Wait for all clients to finish
    print("\n\nWaiting for training to complete...")
    for i, proc in enumerate(processes[1:], 1):  # Skip server
        proc.wait()
        print(f"✓ Client {i} completed")

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

    # Cleanup
    cleanup_processes()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup_processes()
        sys.exit(1)
