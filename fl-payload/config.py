"""
Central Configuration for Federated Learning System

This is the MAIN configuration file. Set all your parameters here.
"""
import os

# ============================================================================
# TASK CONFIGURATION
# ============================================================================

# Task type: 'timeseries', 'object_detection', or 'autoencoder'
TASK_TYPE = 'autoencoder'  # Options: 'timeseries', 'object_detection', 'autoencoder'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model selection - Point this to your model file
# For timeseries: 'models.lstm_model' + 'LSTMModel'
# For object detection: 'models.yolo_model' + 'YOLOFederatedModel'
# For autoencoder: 'models.cae_model' + 'CAEFederatedModel'
if TASK_TYPE == 'timeseries':
    MODEL_PATH = 'models.lstm_model'
    MODEL_CLASS = 'LSTMModel'
    MODEL_CONFIG = {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2
    }
elif TASK_TYPE == 'object_detection':
    MODEL_PATH = 'models.yolo_model'
    MODEL_CLASS = 'YOLOFederatedModel'
    MODEL_CONFIG = {
        'model_path': 'payload/yolov8n.pt',
        'num_classes': 80,
        'img_size': 640,
        'pretrained': True
    }
elif TASK_TYPE == 'autoencoder':
    MODEL_PATH = 'models.cae_model'
    MODEL_CLASS = 'CAEFederatedModel'
    MODEL_CONFIG = {
        'latent_dim': 128,
        'learning_rate': 0.001
    }
    # To test whether model size causes timeouts, use the small CAE (~2–3 MB vs ~48 MB):
    MODEL_PATH = 'models.cae_model_small'
    MODEL_CONFIG = { 'latent_dim': 32, 'learning_rate': 0.001 }
else:
    raise ValueError(f"Unknown TASK_TYPE: {TASK_TYPE}")

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# TIMESERIES DATA (for TASK_TYPE='timeseries')
GPS_FILE = 'data/mission_2_wp_23_attack_add_wp_5_alt_0005_gps.csv'
IMU_FILE = 'data/mission_2_wp_23_attack_add_wp_5_alt_0005_imu.csv'
GPS_FEATURES = ['Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ']
IMU_FEATURES = ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']
TIMESTAMP_COL = 'TimeUS'
SAMPLING_STRATEGY = 'downsample'  # 'downsample' or 'upsample'

# OBJECT DETECTION DATA (for TASK_TYPE='object_detection')
IMAGE_DIR = '../data/payload/images'  # Directory containing images (relative to fl-payload/)
IMAGE_SIZE = 640  # YOLO input size
IMAGES_PER_ROUND = 5  # Number of images to process per round per client

# AUTOENCODER DATA (for TASK_TYPE='autoencoder')
CLEAN_IMAGE_DIR = '../data/payload/images/clean'  # Clean images directory
NOISY_IMAGE_DIR = '../data/payload/images/noise'  # Noisy/adversarial images directory
AUTOENCODER_IMAGE_SIZE = 224  # CAE input size (224x224)
EPOCHS_PER_ROUND = 1  # Number of epochs per FL round

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
# Delay (seconds) per client index before starting round 1 (0, CLIENT_STAGGER_SEC, 2*CLIENT_STAGGER_SEC, ...)
# Use a small value (e.g. 5–10) so all clients start soon; was 120 which made client_2/3 wait 2–4 minutes.
CLIENT_STAGGER_SEC = 10

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

# Calculate input size for timeseries tasks
if TASK_TYPE == 'timeseries':
    INPUT_SIZE = len(GPS_FEATURES) + len(IMU_FEATURES)
    MODEL_CONFIG['input_size'] = INPUT_SIZE
    MODEL_CONFIG['output_size'] = INPUT_SIZE
else:
    INPUT_SIZE = None  # Not applicable for object detection


def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("FEDERATED LEARNING SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"\nTASK: {TASK_TYPE.upper()}")
    print(f"\nMODEL:")
    print(f"  Model: {MODEL_PATH}.{MODEL_CLASS}")

    if TASK_TYPE == 'timeseries':
        print(f"  Hidden Size: {MODEL_CONFIG.get('hidden_size', 'N/A')}")
        print(f"  Num Layers: {MODEL_CONFIG.get('num_layers', 'N/A')}")
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
    elif TASK_TYPE == 'object_detection':
        print(f"  Pretrained Weights: {MODEL_CONFIG.get('model_path', 'N/A')}")
        print(f"  Num Classes: {MODEL_CONFIG.get('num_classes', 'N/A')}")
        print(f"  Image Size: {MODEL_CONFIG.get('img_size', 'N/A')}")
        print(f"\nDATA:")
        print(f"  Image Directory: {IMAGE_DIR}")
        print(f"  Image Size: {IMAGE_SIZE}")
        print(f"  Images per Round: {IMAGES_PER_ROUND}")
    elif TASK_TYPE == 'autoencoder':
        print(f"  Latent Dimension: {MODEL_CONFIG.get('latent_dim', 'N/A')}")
        print(f"  Learning Rate: {MODEL_CONFIG.get('learning_rate', 'N/A')}")
        print(f"\nDATA:")
        print(f"  Clean Images: {CLEAN_IMAGE_DIR}")
        print(f"  Noisy Images: {NOISY_IMAGE_DIR}")
        print(f"  Image Size: {AUTOENCODER_IMAGE_SIZE}x{AUTOENCODER_IMAGE_SIZE}")
        print(f"  Epochs per Round: {EPOCHS_PER_ROUND}")

    print(f"\nFEDERATED LEARNING:")
    print(f"  Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Min Clients: {MIN_CLIENTS}")
    print(f"  Client stagger: {CLIENT_STAGGER_SEC}s per client index")
    print(f"  Aggregation: {AGGREGATION_STRATEGY}")
    print(f"\nTRAINING:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"\nOUTPUT:")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print("=" * 70)
