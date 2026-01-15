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
from config import *

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

    print("\n" + "=" * 70)
    print("MONITORING TRAINING PROGRESS")
    print("=" * 70)

    try:
        while True:
            time.sleep(5)
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    current_round = status['current_round']

                    if current_round > last_round:
                        print(f"\n📊 Round {current_round}:")
                        print(f"   Registered: {status['registered_clients']}/{status['total_expected']}")
                        print(f"   Ready: {status['ready_clients']}/{status['total_expected']}")
                        print(f"   Strategy: {status['aggregation_strategy']}")
                        last_round = current_round

            except:
                pass

    except KeyboardInterrupt:
        pass


def main():
    """Main entry point"""

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Print configuration
    print_config()

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)

    # Print checkpoints
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    if checkpoints:
        print(f"\n📁 Saved Checkpoints ({len(checkpoints)}):")
        for ckpt in sorted(checkpoints):
            print(f"   {CHECKPOINT_DIR}/{ckpt}")

    print(f"\n✓ Final model saved in: {CHECKPOINT_DIR}/")

    # Cleanup
    cleanup_processes()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup_processes()
        sys.exit(1)
