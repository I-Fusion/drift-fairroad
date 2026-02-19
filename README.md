# Federated Learning System

Federated Learning system for **time series** (GPS/IMU), **classification** (e.g. attack vs normal from packet sequences), and **payload/CAE** (image denoising) — with configurable aggregation and evaluation for cyber anomaly detection.

**Convention: run all commands from the project root directory** (the folder that contains `fl-time-series/`, `fl-payload/`, `data/`, and `docker/`). Paths in examples are relative to project root.

**Requirements:** Python **3.10** (used in Docker and recommended for local runs).

---

## 🚀 Quick Start

```bash
# From project root:

# 1. Set up virtual environment (recommended)
python -m venv fl_env
source fl_env/bin/activate  # On Windows: fl_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data in data/ (e.g. data/train/packets or data/train/gps-imu)

# 4. Run timeseries FL (classification or regression)
#    See fl-time-series/README.md; example for GPS/IMU regression:
python fl-time-series/run_fl_system_time_series.py --data-dir data/train/gps-imu --config config_regression --learning-mode regression
```

For packet classification, use `--config config_classification --learning-mode classification` and a packet data dir. See **fl-time-series/README.md** for full options.


## Project Structure

All commands below are run from **project root** (`drift-fairroad/`).

```
drift-fairroad/
├── data/                      # CSV files and payload images
├── requirements.txt
│
├── fl-payload/                # CAE (Convolutional Autoencoder) FL
│   ├── config.py
│   ├── run_fl_cae_system.py   # Run: cd fl-payload; python run_fl_cae_system.py
│   ├── evaluate_cae.py       # Eval: cd fl-payload; python evaluate_cae.py ...
│   ├── fl_server.py, fl_client_cae.py, aggregation.py
│   ├── models/cae_model.py, cae_model_small.py
│   ├── checkpoints/
│   └── README.md
│
├── fl-time-series/            # Classification & regression FL (packet / GPS–IMU)
│   ├── run_fl_system_time_series.py
│   ├── fl_server_time_series.py, fl_client_time_series.py
│   ├── config_classification.py, config_regression.py
│   ├── data_preprocessing_time_series.py, aggregation.py
│   ├── models/                # Timeseries model (lstm_model.py)
│   ├── checkpoints/           # Created by FL runs
│   ├── plots/                 # Loss plots from FL runs
│   ├── evaluation_results/    # From evaluate_fl_system.py
│   ├── evaluate_fl_system.py  # Eval: python fl-time-series/evaluate_fl_system.py ...
│   ├── EVALUATE_FL_SYSTEM.md
│   └── README.md
│
└── docker/
    ├── Dockerfile, Dockerfile.cae
    ├── docker-compose.yml, docker-compose.cae.yml
    └── DOCKER.md
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

### Docker (optional)

All Docker files are in **`docker/`**. Run from the **project root**:

```bash
# Timeseries FL
docker-compose -f docker/docker-compose.yml up --build

# CAE FL (image denoising)
docker-compose -f docker/docker-compose.cae.yml up --build
```

See **`docker/DOCKER.md`** for details (build context, volumes, troubleshooting).


## Running CAE FL (fl-payload)

From **project root**, run the CAE system with working directory set to `fl-payload` (so its relative paths resolve correctly). Use `;` on Windows PowerShell instead of `&&`:

```bash
cd fl-payload; python run_fl_cae_system.py --num-rounds 10
```

Evaluate (also from project root, run from inside `fl-payload`):

```bash
cd fl-payload; python evaluate_cae.py --checkpoint-dir checkpoints --clean-dir ../data/payload/images/clean --noisy-dir ../data/payload/images/noise --num-clients 3 --output-dir evaluation_results
```

See **`fl-payload/README.md`** for full options.


## FL Time-Series (fl-time-series)

The **`fl-time-series/`** directory provides federated learning for **classification** (e.g. attack vs normal from labeled packet sequences) and **regression** (e.g. GPS/IMU time-series prediction). It targets **cyber anomaly detection** (GPS spoofing, waypoint injection, jamming) with multiple clients training locally and a server aggregating model updates (FedAvg, FedAvgM, or weighted).

- **Classification**: Labeled packet sequences (e.g. SrcPort, DstPort, Length, MsgID, Protocol → Label). Sliding windows + LSTM (or configured model) for binary/multi-class detection.
- **Regression**: GPS/IMU time-series; same FL server/client layout, different data and loss.
- **Config**: `config_classification` (packet-only) or `config_regression` (GPS/IMU); selected at runtime with `--config`.

**Quick start** (from project root):

```bash
# Prepare packet data (if needed)
python raw_data_processing/prepare_fl_data_for_run_fl.py --packet-file data/network_packets/your_labeled.csv --output data/train/packets

# Run FL classification
python fl-time-series/run_fl_system_time_series.py --data-dir data/train/packets --config config_classification --learning-mode classification

# Run FL regression (GPS/IMU)
python fl-time-series/run_fl_system_time_series.py --data-dir data/train/gps-imu --config config_regression --learning-mode regression
```

Checkpoints go to `fl-time-series/checkpoints/` (and `fl-time-series/checkpoints/clients/` for client checkpoints). Evaluate with `fl-time-series/evaluate_fl_system.py` (see Evaluation section below). See **`fl-time-series/README.md`** for options and data format.


## Evaluation (timeseries / classification)

From **project root**, run the comprehensive evaluator (defaults: `fl-time-series/checkpoints`, `fl-time-series/evaluation_results`):

```bash
python fl-time-series/evaluate_fl_system.py --data-dir data/validate/packets --task classification --config config_classification
```

See **`fl-time-series/EVALUATE_FL_SYSTEM.md`** for all options and evaluation dimensions.


## Configuration

### Timeseries FL: `fl-time-series/config_classification.py` and `config_regression.py`

Use **`config_classification`** for packet (classification) or **`config_regression`** for GPS/IMU regression. Edit the corresponding file in `fl-time-series/` and pass `--config config_classification` or `--config config_regression` to the run script.

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
# PATHS (resolved to fl-time-series/checkpoints and fl-time-series/plots)
# ============================================================================

_FL_TS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_FL_TS_DIR, 'checkpoints')
PLOT_DIR = os.path.join(_FL_TS_DIR, 'plots')
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

Run all commands from **project root**. See **fl-time-series/README.md** for full options.

### Method 1: One-Command (Recommended)

```bash
# Timeseries FL (example: GPS/IMU regression)
python fl-time-series/run_fl_system_time_series.py --data-dir data/train/gps-imu --config config_regression --learning-mode regression
```

This will start server + clients, train, save checkpoints under `fl-time-series/checkpoints/`, and print metrics.

### Method 2: Manual (For Debugging)

**Terminal 1 - Server:**
```bash
python fl-time-series/fl_server_time_series.py --host localhost --port 5544 --num-clients 3
```

**Terminal 2-4 - Clients** (paths relative to project root; use your data dir and config):
```bash
python fl-time-series/fl_client_time_series.py --client-id client_1 --data-dir data/train/gps-imu --config config_regression --learning-mode regression --server-url http://localhost:5544
# ... client_2, client_3 with same args
```

---

## Creating Custom Models

### Step 1: Create Your Model File

Create **`fl-time-series/models/my_model.py`**:

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

### Step 2: Update config in `fl-time-series/`

In the config you use (e.g. **`fl-time-series/config_regression.py`**), set:

```python
MODEL_PATH = 'models.my_model'   # Your model file (in fl-time-series/models/)
MODEL_CLASS = 'MyCustomModel'    # Your model class
```

### Step 3: Run

From project root:

```bash
python fl-time-series/run_fl_system_time_series.py --data-dir <your-data-dir> --config config_regression --learning-mode regression
```

This runs the custom model for the FL training.

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
   fl-time-series/checkpoints/server_round_1.pt
   fl-time-series/checkpoints/server_round_2.pt
   ...
   fl-time-series/checkpoints/server_round_5.pt

📈 Training Loss Plot:
   fl-time-series/plots/training_loss.png

✓ Training artifacts saved successfully!
```

### Loss Tracking and Plotting

The system automatically tracks loss from each client after every training round. Each client's loss curve is recorded and plotted along with an average of the overall loss.

```
fl-time-series/plots/training_loss.png
```

### Loading Checkpoints

From project root (or add `fl-time-series` to path):

```python
import torch
from models.lstm_model import LSTMModel  # when run from fl-time-series or with path set

# Load checkpoint (path under fl-time-series)
checkpoint = torch.load('fl-time-series/checkpoints/server_round_5.pt')

# Create model
model = LSTMModel(input_size=12, hidden_size=64, num_layers=2)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from round {checkpoint['round']}")
```

### Checkpoint Cleanup

Before each training run, the system automatically:
- Checks for existing `.pt` files in `fl-time-series/checkpoints/` directory
- Removes all old checkpoint files
- Logs the number of files cleaned up

This prevents:
- Confusion between old and new checkpoints
- Disk space issues from accumulating checkpoints
- Accidental loading of outdated models

To disable cleanup, comment out the `cleanup_old_checkpoints()` call in `fl-time-series/run_fl_system_time_series.py`.

---
