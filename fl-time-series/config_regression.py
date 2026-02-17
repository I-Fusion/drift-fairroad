"""
Configuration for Federated Learning System - GPS/IMU Regression (No Packets)

This config is meant for regression / time-series prediction using ONLY
GPS and IMU data. Packet data is not used.

Typical usage with the packet-aware runner:

    python run_fl_system_with_packets.py --config config_regression --learning-mode regression

or, with prepared client-specific splits:

    python run_fl_system_with_packets.py \\
        --config config_regression \\
        --data-dir data/prepared_clients \\
        --learning-mode regression
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model selection
MODEL_PATH = "models.lstm_model"
MODEL_CLASS = "LSTMModel"

# Model architecture parameters
MODEL_CONFIG = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
}

# ============================================================================
# DATA CONFIGURATION - GPS/IMU ONLY (NO PACKETS)
# ============================================================================

# Data file paths (can be overridden by CLI args or prepared client files)
GPS_FILE = "data/waypoint_injection/mission_2_wp_23_attack_add_wp_5_alt_0005_gps.csv"
IMU_FILE = "data/waypoint_injection/mission_2_wp_23_attack_add_wp_5_alt_0005_imu.csv"

# Features to use from data
GPS_FEATURES = ["Lat", "Lng", "Alt", "Spd", "GCrs", "VZ"]
IMU_FEATURES = ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]

# Timestamp column
TIMESTAMP_COL = "TimeUS"

# Sampling strategy: 'downsample' (IMU -> GPS) or 'upsample' (GPS -> IMU)
SAMPLING_STRATEGY = "downsample"

# Explicitly indicate that packet data is not used in this config
PACKET_FILE = None
PACKET_FEATURES = []
PACKET_TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Label"
USE_LABELS = False

# ============================================================================
# WINDOW CONFIGURATION
# ============================================================================

# Sliding window parameters
WINDOW_SIZE = 50         # Number of timesteps per window
OVERLAP = 25             # Number of overlapping timesteps
WINDOWS_PER_ROUND = 10   # Windows to process before syncing with server

# ============================================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================================

# Server configuration
SERVER_HOST = "localhost"
SERVER_PORT = 5544
NUM_CLIENTS = 3
MIN_CLIENTS = 2           # Minimum clients needed to start aggregation

# Aggregation strategy: 'fedavg', 'fedavgm', or 'weighted'
AGGREGATION_STRATEGY = "fedavg"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# ============================================================================
# PATHS (under fl-time-series so outputs stay in this folder)
# ============================================================================

_FL_TS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_FL_TS_DIR, "checkpoints")
PLOT_DIR = os.path.join(_FL_TS_DIR, "plots")

# ============================================================================
# COMPUTED PARAMETERS (Don't modify)
# ============================================================================

# Input size from GPS + IMU features only
INPUT_SIZE = len(GPS_FEATURES) + len(IMU_FEATURES)

# For regression / time-series prediction we typically predict the next
# full state vector, so set output_size = INPUT_SIZE.
MODEL_CONFIG["input_size"] = INPUT_SIZE
MODEL_CONFIG["output_size"] = INPUT_SIZE


def print_config() -> None:
    """Print current configuration."""
    print("=" * 70)
    print("FEDERATED LEARNING SYSTEM CONFIGURATION - GPS/IMU REGRESSION (NO PACKETS)")
    print("=" * 70)
    print("\nMODEL:")
    print(f"  Model: {MODEL_PATH}.{MODEL_CLASS}")
    print(f"  Hidden Size: {MODEL_CONFIG['hidden_size']}")
    print(f"  Num Layers: {MODEL_CONFIG['num_layers']}")
    print(f"  Input Size: {INPUT_SIZE} features")
    print(f"  Output Size: {MODEL_CONFIG['output_size']} (regression)")
    print("\nDATA:")
    print(f"  GPS File: {GPS_FILE}")
    print(f"  IMU File: {IMU_FILE}")
    print(f"  GPS Features: {', '.join(GPS_FEATURES)}")
    print(f"  IMU Features: {', '.join(IMU_FEATURES)}")
    print(f"  Sampling Strategy: {SAMPLING_STRATEGY}")
    print("\nWINDOW:")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  Overlap: {OVERLAP}")
    print(f"  Windows per Round: {WINDOWS_PER_ROUND}")
    print("\nFEDERATED LEARNING:")
    print(f"  Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Min Clients: {MIN_CLIENTS}")
    print(f"  Aggregation: {AGGREGATION_STRATEGY}")
    print("\nTRAINING:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print("\nOUTPUT:")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print("=" * 70)

