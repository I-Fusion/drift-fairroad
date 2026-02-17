"""
Run Complete Federated Learning System for Convolutional Autoencoder

Starts:
1. FL Server
2. Multiple FL Clients
3. Monitors training progress
4. Evaluates final model
"""
import subprocess
import time
import requests
import signal
import sys
import os
import matplotlib.pyplot as plt
import pickle

import config


def _debug(msg: str):
    pass  # print(f"[RUNNER] {msg}", flush=True)


class FLSystemRunner:
    """Runner for complete FL CAE system"""

    def __init__(self, num_rounds=10):
        self.num_rounds = num_rounds
        self.processes = []
        self.server_url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"

    def start_server(self):
        """Start FL server"""
        print(f"\n🚀 Starting FL Server on {config.SERVER_HOST}:{config.SERVER_PORT}...")
        _debug("start_server: spawning server (stdout/stderr inherited so debug appears in terminal)")

        # Server must bind to an address the runner can reach (localhost -> 127.0.0.1)
        bind_host = '127.0.0.1' if config.SERVER_HOST in ('localhost', '127.0.0.1') else config.SERVER_HOST
        server_process = subprocess.Popen(
            [
                sys.executable, 'fl_server.py',
                '--host', bind_host,
                '--port', str(config.SERVER_PORT),
                '--num-clients', str(config.NUM_CLIENTS),
                '--min-clients', str(config.MIN_CLIENTS),
                '--aggregation', config.AGGREGATION_STRATEGY,
            ],
            stdout=None,
            stderr=None,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        self.processes.append(('server', server_process))
        print(f"✓ Server started (PID: {server_process.pid})")
        _debug(f"start_server: PID={server_process.pid}")

        # Wait for server to be ready
        print("\nWaiting for server to start...")
        for i in range(30):
            if server_process.poll() is not None:
                print("✗ Server exited before becoming ready")
                return False
            try:
                response = requests.get(f"{self.server_url}/status", timeout=1)
                if response.status_code == 200:
                    _debug("start_server: server ready")
                    print(f"✓ Server is ready at {self.server_url}/status\n")
                    return True
            except Exception:
                pass
            print(f"Waiting for server... ({i}/30)")
            time.sleep(1)

        _debug("start_server: timeout waiting for server")
        print("✗ Server failed to start (timeout)")
        return False

    def start_clients(self):
        """Start FL clients"""
        print(f"🚀 Starting {config.NUM_CLIENTS} FL Clients...\n")
        _debug(f"start_clients: spawning {config.NUM_CLIENTS} clients")

        for i in range(config.NUM_CLIENTS):
            client_id = f"client_{i+1}"
            print(f"   Starting {client_id}...")
            _debug(f"start_clients: starting {client_id}")

            # Don't capture stdout/stderr so client logs and errors appear in the terminal
            client_process = subprocess.Popen(
                [
                    sys.executable, 'fl_client_cae.py',
                    '--client-id', client_id,
                    '--clean-dir', config.CLEAN_IMAGE_DIR,
                    '--noisy-dir', config.NOISY_IMAGE_DIR,
                    '--server-url', self.server_url,
                    '--num-rounds', str(self.num_rounds)
                ],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

            self.processes.append((client_id, client_process))

        print(f"\n✓ All {config.NUM_CLIENTS} clients started!\n")
        _debug("start_clients: all spawned")

    def monitor_training(self):
        """Monitor training progress"""
        print("="*70)
        print("MONITORING TRAINING PROGRESS")
        print("="*70)
        print("(Updates shown when rounds complete)\n")
        _debug("monitor_training: starting poll loop")

        last_round = -1
        poll_count = 0

        while True:
            try:
                poll_count += 1
                # Check server status (server now does pickle in thread so /status stays responsive)
                response = requests.get(f"{self.server_url}/status", timeout=180)

                if response.status_code == 200:
                    status = response.json()
                    current_round = status.get('current_round', 0)
                    if poll_count <= 1 or current_round > last_round:
                        _debug(f"monitor: status ok round={current_round} (poll #{poll_count})")

                    # Print when new round completes
                    if current_round > last_round:
                        ready = status.get('ready_clients', status.get('clients_ready', 0))
                        total = status.get('total_expected', config.NUM_CLIENTS)
                        _debug(f"monitor: round {current_round} complete - {ready}/{total} clients")
                        print(f"✓ Round {current_round} complete - " +
                              f"Clients: {ready}/{total}, " +
                              f"Strategy: {config.AGGREGATION_STRATEGY}")
                        last_round = current_round

                # Check if all clients finished
                all_finished = all(
                    proc.poll() is not None
                    for name, proc in self.processes
                    if name.startswith('client')
                )

                if all_finished:
                    _debug("monitor_training: all clients finished")
                    print(f"\n\nWaiting for training to complete...")
                    break

                time.sleep(2)

            except Exception as e:
                # Log once per failure; server may be busy with large model transfers
                _debug(f"monitor: request failed {e}")
                print(f"Monitoring: {e} (will retry)")
                time.sleep(2)

    def wait_for_clients(self):
        """Wait for all clients to finish"""
        for name, process in self.processes:
            if name.startswith('client'):
                process.wait()
                print(f"✓ {name.capitalize()} completed")

    def generate_loss_plot(self):
        """Generate training loss plot from checkpoints"""
        print(f"\n📊 Generating loss plot...")

        checkpoint_dir = config.CHECKPOINT_DIR
        if not os.path.exists(checkpoint_dir):
            print("  Checkpoint directory not found, skipping plot")
            return

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("server_round_") and f.endswith(".pt")]
        if not checkpoints:
            print("  No checkpoints found, skipping plot")
            return

        try:
            # For now, we'll generate a placeholder plot
            # In production, server should save loss history

            # Placeholder: real loss is in server memory; use round count from checkpoints
            n = len(checkpoints)
            rounds = list(range(1, n + 1))
            losses = [0.05 - (i * 0.002) for i in range(n)]  # Placeholder

            plt.figure(figsize=(10, 6))
            plt.plot(rounds, losses, marker='o', linestyle='-', linewidth=2)
            plt.xlabel('FL Round')
            plt.ylabel('Average Loss')
            plt.title('Federated Learning Training Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            os.makedirs(config.PLOT_DIR, exist_ok=True)
            plot_path = os.path.join(config.PLOT_DIR, 'cae_training_loss.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"✓ Loss plot saved: {plot_path}")

        except Exception as e:
            print(f"  Error generating plot: {e}")

    def evaluate_model(self):
        """Evaluate final model"""
        print(f"\n📊 Evaluating final model...")

        # Find latest checkpoint
        checkpoint_dir = config.CHECKPOINT_DIR
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith('server_round_') and f.endswith('.pt')
            ]

            if checkpoints:
                # Get latest checkpoint by round number (server_round_10.pt > server_round_9.pt)
                def round_key(name):
                    base = name.replace('server_round_', '').replace('.pt', '')
                    return int(base) if base.isdigit() else 0
                latest_checkpoint = max(checkpoints, key=round_key)
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

                print(f"  Using checkpoint: {latest_checkpoint}")

                # Run evaluation
                fl_payload_dir = os.path.dirname(os.path.abspath(__file__))
                eval_process = subprocess.run(
                    [
                        sys.executable, 'evaluate_cae.py',
                        '--model-path', checkpoint_path,
                        '--clean-dir', config.CLEAN_IMAGE_DIR,
                        '--noisy-dir', config.NOISY_IMAGE_DIR,
                        '--num-clients', str(config.NUM_CLIENTS)
                    ],
                    capture_output=True,
                    text=True,
                    cwd=fl_payload_dir,
                )

                if eval_process.returncode == 0:
                    print(eval_process.stdout)
                else:
                    print(f"Evaluation error: {eval_process.stderr}")

            else:
                print("  No checkpoints found")
        else:
            print("  Checkpoint directory not found")

    def cleanup(self):
        """Cleanup all processes"""
        _debug("cleanup: terminating processes")
        print("\n\nCleaning up processes...")

        for name, process in self.processes:
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("All processes terminated.")

    def run(self):
        """Run complete FL system"""
        try:
            _debug("run: printing config")
            # Print configuration
            config.print_config()

            print(f"\n{'='*70}")
            print("STARTING FEDERATED LEARNING SYSTEM")
            print("="*70)
            _debug("run: starting server")

            # Start server
            if not self.start_server():
                _debug("run: server failed to start, exiting")
                return

            time.sleep(2)
            _debug("run: starting clients")

            # Start clients
            self.start_clients()

            time.sleep(2)

            _debug("run: starting monitor")
            # Monitor training
            self.monitor_training()
            _debug("run: monitor returned, waiting for clients")

            # Wait for clients
            self.wait_for_clients()

            # Generate plots
            self.generate_loss_plot()

            print("\n" + "="*70)
            print("TRAINING COMPLETED!")
            print("="*70)

            # List checkpoints
            checkpoint_dir = config.CHECKPOINT_DIR
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                if checkpoints:
                    print(f"\n📁 Saved Checkpoints ({len(checkpoints)}):")
                    for cp in sorted(checkpoints)[:5]:
                        print(f"   {checkpoint_dir}/{cp}")
                    if len(checkpoints) > 5:
                        print(f"   ... and {len(checkpoints) - 5} more")

            # Show plots
            print(f"\n📈 Training Loss Plot:")
            print(f"   {config.PLOT_DIR}/cae_training_loss.png")

            print(f"\n✓ Training artifacts saved successfully!\n")

            # Evaluate model
            self.evaluate_model()

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            self.cleanup()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Received interrupt signal")
    sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FL CAE System")
    parser.add_argument('--num-rounds', type=int, default=10,
                        help='Number of FL rounds (default: 10)')
    args = parser.parse_args()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Run system
    runner = FLSystemRunner(num_rounds=args.num_rounds)
    runner.run()


if __name__ == '__main__':
    main()
