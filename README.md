# Federated Learning System

Federated Learning system for training time series models on GPS and IMU sensor data and payload based data. 

## 🚀 Quick Start

```bash
# 1. Set up virtual environment (recommended)
python -m venv fl_env
source fl_env/bin/activate  # On Windows: fl_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the data CSV files in data/ folder
ls data/

# 4. Configure parameters (optional - defaults are set)
nano config.py

# 5. Run the entire FL system
python run_fl_system.py
```

The system will:
- Start 1 server + N clients automatically
- Load and preprocess data with configurable sampling strategy
- Train using sliding windows (single-pass)
- Sync periodically with server
- Track and plot loss for each client and average loss
- Print metrics for each round
- Save checkpoints and loss plots automatically


## Project Structure

### Essential Files (Need to Deploy)

```
FL/
├── config.py                  # ⭐ MAIN CONFIGURATION
├── run_fl_system.py           # ⭐ RUN THIS to start FL system
│
├── fl_client.py               # FL client implementation
├── fl_server.py               # FL server implementation
├── data_preprocessing.py      # GPS+IMU data fusion
├── aggregation.py             # Aggregation strategies
│
├── models/
│   ├── __init__.py
│   └── lstm_model.py          # Default LSTM model
│
├── data/                      # Place your CSV files here
│   ├── your_gps.csv
│   └── your_imu.csv
│
├── checkpoints/               # Auto-created for model checkpoints
├── plots/                     # Auto-created for loss plots
│
└── requirements.txt           # Python dependencies
```
---


## Installation

### Method 1: With Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv fl_env

# Activate virtual environment
# On macOS/Linux:
source fl_env/bin/activate
# On Windows:
fl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# When done, deactivate
deactivate
```

### Method 2: Global Installation

```bash
# Install dependencies globally
pip install -r requirements.txt

# Or manually
pip install torch numpy aiohttp pandas requests matplotlib
```


## Configuration

### Main Configuration File: `config.py`

This is the **ONLY** file you need to edit to configure the entire system.

```python
# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model selection - point to your model file
MODEL_PATH = 'models.lstm_model'
MODEL_CLASS = 'LSTMModel'

# Model parameters
MODEL_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Data files (used by all clients)
GPS_FILE = 'data/your_gps.csv'
IMU_FILE = 'data/your_imu.csv'

# Features to use
GPS_FEATURES = ['Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ']
IMU_FEATURES = ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']

# Sampling strategy: 'downsample' or 'upsample'
# 'downsample': IMU data downsampled to match GPS rate (fewer samples, default)
# 'upsample': GPS data upsampled to match IMU rate (more samples)
SAMPLING_STRATEGY = 'downsample'

# ============================================================================
# WINDOW CONFIGURATION
# ============================================================================

WINDOW_SIZE = 50        # Timesteps per window
OVERLAP = 25            # Overlapping timesteps
WINDOWS_PER_ROUND = 10  # Windows before syncing with server

# ============================================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================================

SERVER_HOST = 'localhost'
SERVER_PORT = 8080
NUM_CLIENTS = 3
MIN_CLIENTS = 2

# Aggregation: 'fedavg', 'fedavgm', or 'weighted'
AGGREGATION_STRATEGY = 'fedavg'

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

LEARNING_RATE = 0.001
BATCH_SIZE = 32

# ============================================================================
# PATHS
# ============================================================================

CHECKPOINT_DIR = 'checkpoints'
PLOT_DIR = 'plots'
```

### Parameter Guide

| Parameter | What It Controls | Recommended Values |
|-----------|-----------------|-------------------|
| `SAMPLING_STRATEGY` | Data fusion strategy ('downsample' or 'upsample') | 'downsample' (default) |
| `WINDOW_SIZE` | How many timesteps in each training window | 30-100 |
| `OVERLAP` | How many timesteps overlap between windows | 50-75% of window_size |
| `WINDOWS_PER_ROUND` | How many windows before syncing with server | 5-20 |
| `NUM_CLIENTS` | Number of FL clients to run | 2-5 |
| `MIN_CLIENTS` | Minimum clients required before training starts | 2-NUM_CLIENTS |
| `hidden_size` | Model capacity (LSTM units) | 32-128 |
| `num_layers` | Model depth (LSTM layers) | 2-3 |

### Sampling Strategy for GPS+IMU data

**Downsample (default):**
- IMU data is downsampled to match the slower GPS sampling rate
- Example: GPS @ 5 Hz, IMU @ 100 Hz → Output @ 5 Hz

**Upsample:**
- GPS data is upsampled to match the faster IMU sampling rate
- Example: GPS @ 5 Hz, IMU @ 100 Hz → Output @ 100 Hz

---

## Running the System

### Method 1: One-Command (Recommended)

```bash
python run_fl_system.py
```

This will:
1. Print configuration
2. Start FL server
3. Start all clients automatically
4. Monitor training progress
5. Print metrics for each round
6. Save checkpoints
7. Print completion summary

### Method 2: Manual (For Debugging)

**Terminal 1 - Server:**
```bash
python fl_server.py
```

**Terminal 2-4 - Clients:**
```bash
python fl_client.py --client-id client_1 --gps-file data/your_gps.csv --imu-file data/your_imu.csv --server-url http://localhost:5544
python fl_client.py --client-id client_2 --gps-file data/your_gps.csv --imu-file data/your_imu.csv --server-url http://localhost:5544
python fl_client.py --client-id client_3 --gps-file data/your_gps.csv --imu-file data/your_imu.csv --server-url http://localhost:5544
```

---

## Creating Custom Models

### Step 1: Create Your Model File

Create `models/my_model.py`:

```python
import torch
import torch.nn as nn

class MyCustomModel(nn.Module):
    """Your custom model for time series prediction."""

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Your architecture
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Your forward pass
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### Step 2: Update config.py

```python
# Change these two lines in config.py:
MODEL_PATH = 'models.my_model'      # Your model file
MODEL_CLASS = 'MyCustomModel'        # Your model class
```

### Step 3: Run

```bash
python run_fl_system.py
```

This should run the custom model for the FL training.

---

## Data Format

### Required Data Files

Place in `data/` folder:
- GPS CSV file
- IMU CSV file

### GPS CSV Format

Must include these columns:
- `TimeUS` - Timestamp in microseconds
- `Lat` - Latitude
- `Lng` - Longitude
- `Alt` - Altitude
- `Spd` - Speed
- `GCrs` - Ground course
- `VZ` - Vertical velocity

**Example:**
```csv
,Timestamp,,TimeUS,GPS_ID,Status,GMS,GWk,Nstats,HDop,Lat,Lng,Alt,Spd,GCrs,VZ,Yaw,U
2155,25:14.6,GPS,65785342,0,6,156332600,2401,10,1.21,-35.3632621,149.1652374,584.09,0,353.8629,0,0,1
```

### IMU CSV Format

Must include these columns:
- `TimeUS` - Timestamp in microseconds
- `I` - Sensor ID (will filter to I=0)
- `GyrX`, `GyrY`, `GyrZ` - Gyroscope readings
- `AccX`, `AccY`, `AccZ` - Accelerometer readings

**Example:**
```csv
,Timestamp,,TimeUS,I,GyrX,GyrY,GyrZ,AccX,AccY,AccZ,EG,EA,T,GH,AH,GHz,AHz
1933,25:14.5,IMU,65645398,0,0.000902,0.000975,0.000909,-0.000436,-0.001627,-9.817502,0,0,28.82777,1,1,1000,1000
```

### Data Preprocessing

The system automatically:
1. Loads GPS and IMU CSV files
2. Filters IMU data (keeps sensor I=0)
3. Applies sampling strategy (downsample or upsample) to align timestamps
4. Merges on timestamp using nearest neighbor matching
5. Normalizes using z-score (mean=0, std=1)
6. Creates sliding windows with overlap

**Total features:** 6 GPS + 6 IMU = 12 features

**Sampling Strategy:**
- **Downsample**: IMU → GPS rate (fewer, cleaner samples)
- **Upsample**: GPS → IMU rate (more samples, interpolated GPS)

---

## Output and Checkpoints

### During Training

```
======================================================================
FEDERATED LEARNING SYSTEM CONFIGURATION
======================================================================

MODEL:
  Model: models.lstm_model.LSTMModel
  Hidden Size: 64
  Num Layers: 2
  Input Size: 12 features

DATA:
  GPS File: data/mission_2_wp_23_attack_add_wp_5_alt_0005_gps.csv
  IMU File: data/mission_2_wp_23_attack_add_wp_5_alt_0005_imu.csv

WINDOW:
  Window Size: 50
  Overlap: 25
  Windows per Round: 10

FEDERATED LEARNING:
  Server: localhost:8080
  Clients: 3
  Aggregation: fedavg

======================================================================
MONITORING TRAINING PROGRESS
======================================================================

📊 Round 1:
   Registered: 3/3
   Ready: 3/3
   Strategy: fedavg

📊 Round 2:
   Registered: 3/3
   Ready: 3/3
   Strategy: fedavg

...

======================================================================
TRAINING COMPLETED!
======================================================================

📁 Saved Checkpoints (5):
   checkpoints/server_round_1.pt
   checkpoints/server_round_2.pt
   checkpoints/server_round_3.pt
   checkpoints/server_round_4.pt
   checkpoints/server_round_5.pt

📈 Training Loss Plot:
   plots/training_loss.png

✓ Training artifacts saved successfully!
```

### Loss Tracking and Plotting

The system automatically tracks loss from each client after every training round. Each client's loss curve is recorded and plotted along with an average of the overall loss.

```
plots/training_loss.png
```

### Loading Checkpoints

```python
import torch
from models.lstm_model import LSTMModel

# Load checkpoint
checkpoint = torch.load('checkpoints/server_round_5.pt')

# Create model
model = LSTMModel(input_size=12, hidden_size=64, num_layers=2)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from round {checkpoint['round']}")
```

### Checkpoint Cleanup

Before each training run, the system automatically:
- Checks for existing `.pt` files in `checkpoints/` directory
- Removes all old checkpoint files
- Logs the number of files cleaned up

This prevents:
- Confusion between old and new checkpoints
- Disk space issues from accumulating checkpoints
- Accidental loading of outdated models

To disable cleanup, comment out the `cleanup_old_checkpoints()` call in [run_fl_system.py](FL/run_fl_system.py:110).

---
