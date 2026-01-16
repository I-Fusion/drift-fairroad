"""
Central Configuration for Federated Learning System

This is the MAIN configuration file. Set all your parameters here.
"""
import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model selection - Point this to your model file
# To use a custom model:
# 1. Create 'models/my_model.py' with a class that has __init__ and forward methods
# 2. Change MODEL_PATH to 'models.my_model'
# 3. Change MODEL_CLASS to your class name
MODEL_PATH = 'models.lstm_model'
MODEL_CLASS = 'LSTMModel'

# Model architecture parameters
MODEL_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Data file paths (will be used by all clients)
GPS_FILE = 'data/waypoint_injection/mission_2_wp_23_attack_add_wp_5_alt_0005_gps.csv'
IMU_FILE = 'data/waypoint_injection/mission_2_wp_23_attack_add_wp_5_alt_0005_imu.csv'

# Features to use from data
GPS_FEATURES = ['Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ']
IMU_FEATURES = ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']

# Timestamp column
TIMESTAMP_COL = 'TimeUS'

# Sampling strategy: 'downsample' or 'upsample'
# 'downsample': IMU data downsampled to match GPS rate (default)
# 'upsample': GPS data upsampled to match IMU rate
SAMPLING_STRATEGY = 'downsample'

# ============================================================================
# WINDOW CONFIGURATION
# ============================================================================

# Sliding window parameters
WINDOW_SIZE = 50        # Number of timesteps per window
OVERLAP = 25            # Number of overlapping timesteps
WINDOWS_PER_ROUND = 10  # Windows to process before syncing with server

# ============================================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================================

# Server configuration
SERVER_HOST = 'localhost'
SERVER_PORT = 5544
NUM_CLIENTS = 3
MIN_CLIENTS = 2  # Minimum clients needed to start aggregation

# Aggregation strategy: 'fedavg', 'fedavgm', or 'weighted'
AGGREGATION_STRATEGY = 'fedavg'

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# ============================================================================
# PATHS
# ============================================================================

# Output directory for checkpoints and plots
CHECKPOINT_DIR = 'checkpoints'
PLOT_DIR = 'plots'

# ============================================================================
# COMPUTED PARAMETERS (Don't modify)
# ============================================================================

# Calculate input size from features
INPUT_SIZE = len(GPS_FEATURES) + len(IMU_FEATURES)

# Add to model config
MODEL_CONFIG['input_size'] = INPUT_SIZE
MODEL_CONFIG['output_size'] = INPUT_SIZE


def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("FEDERATED LEARNING SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"\nMODEL:")
    print(f"  Model: {MODEL_PATH}.{MODEL_CLASS}")
    print(f"  Hidden Size: {MODEL_CONFIG['hidden_size']}")
    print(f"  Num Layers: {MODEL_CONFIG['num_layers']}")
    print(f"  Input Size: {INPUT_SIZE} features")
    print(f"\nDATA:")
    print(f"  GPS File: {GPS_FILE}")
    print(f"  IMU File: {IMU_FILE}")
    print(f"  GPS Features: {', '.join(GPS_FEATURES)}")
    print(f"  IMU Features: {', '.join(IMU_FEATURES)}")
    print(f"  Sampling Strategy: {SAMPLING_STRATEGY}")
    print(f"\nWINDOW:")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  Overlap: {OVERLAP}")
    print(f"  Windows per Round: {WINDOWS_PER_ROUND}")
    print(f"\nFEDERATED LEARNING:")
    print(f"  Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Min Clients: {MIN_CLIENTS}")
    print(f"  Aggregation: {AGGREGATION_STRATEGY}")
    print(f"\nTRAINING:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"\nOUTPUT:")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print("=" * 70)
