"""
Configuration for Federated Learning System - Time-Series

This configuration is for time-series tasks: packet-only (with labels) or GPS/IMU regression.
"""
import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model selection
MODEL_PATH = 'models.lstm_model'
MODEL_CLASS = 'LSTMModel'

# Model architecture parameters
MODEL_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2
}

# ============================================================================
# DATA CONFIGURATION - PACKET ONLY
# ============================================================================

# GPS and IMU files are not used (set to None)
GPS_FILE = None
IMU_FILE = None

# Packet data configuration
# This will be overridden by command-line arguments or client-specific files
PACKET_FILE = None  # Will be set per client from data_dir

# Packet features to use
PACKET_FEATURES = ['SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol']

# Timestamp columns
TIMESTAMP_COL = 'TimeUS'  # Not used for packet-only, but kept for compatibility
PACKET_TIMESTAMP_COL = 'Timestamp'  # Timestamp column in packet data
LABEL_COL = 'Label'  # Label column in packet data

# Sampling strategy (not used for packet-only, but kept for compatibility)
SAMPLING_STRATEGY = 'downsample'

# Enable supervised learning (use labels)
USE_LABELS = True

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
# PATHS (under fl-time-series so outputs stay in this folder)
# ============================================================================

_FL_TS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_FL_TS_DIR, 'checkpoints')
PLOT_DIR = os.path.join(_FL_TS_DIR, 'plots')

# ============================================================================
# COMPUTED PARAMETERS (Don't modify)
# ============================================================================

# Calculate input size from packet features only
INPUT_SIZE = len(PACKET_FEATURES)

# Add to model config
MODEL_CONFIG['input_size'] = INPUT_SIZE
MODEL_CONFIG['output_size'] = 1  # Binary classification output size


def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("FEDERATED LEARNING SYSTEM CONFIGURATION - PACKET ONLY")
    print("=" * 70)
    print(f"\nMODEL:")
    print(f"  Model: {MODEL_PATH}.{MODEL_CLASS}")
    print(f"  Hidden Size: {MODEL_CONFIG['hidden_size']}")
    print(f"  Num Layers: {MODEL_CONFIG['num_layers']}")
    print(f"  Input Size: {INPUT_SIZE} features (packet only)")
    print(f"  Output Size: {MODEL_CONFIG['output_size']} (binary classification)")
    print(f"\nDATA:")
    print(f"  Mode: Supervised Learning (Packet Sequences Only)")
    print(f"  GPS File: {GPS_FILE} (not used)")
    print(f"  IMU File: {IMU_FILE} (not used)")
    print(f"  Packet Features: {', '.join(PACKET_FEATURES)}")
    print(f"  Packet Timestamp Column: {PACKET_TIMESTAMP_COL}")
    print(f"  Label Column: {LABEL_COL}")
    print(f"  Use Labels: {USE_LABELS}")
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
    print(f"  Learning Mode: Time-Series (binary labels when USE_LABELS=True)")
    print(f"\nOUTPUT:")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print("=" * 70)
