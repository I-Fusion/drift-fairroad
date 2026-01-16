# Data Processing Pipeline Documentation

This folder contains scripts for processing GPS and IMU data and preparing it for Federated Learning (FL) with LSTM models. The pipeline processes raw sensor data, labels attack periods, and prepares sequences for distributed training.

## Overview

The data processing pipeline consists of four main stages:

1. **Time Period Detection**: Identify attack periods from network packet data
2. **Data Labeling**: Label GPS/IMU CSV data or packet .npy data based on detected attack time periods
3. **FL Data Preparation**: Create sequences and divide data among FL clients
4. **Packet Data Processing**: Filter and label packet data from .npy files (alternative to CSV labeling)

## Pipeline Workflow

**Main Pipeline (GPS/IMU CSV Data):**
```
Raw Data (.npy files) 
    ↓
[find_time_periods.py] → Time Periods CSV
    ↓
GPS/IMU CSV Data + Time Periods
    ↓
[label_csv_with_periods.py] → Labeled & Filtered CSV
    (Auto-detects file type, applies filtering)
    ↓
[prepare_fl_data.py] → FL Client Data (NPZ files)
```

**Alternative Pipeline (Packet .npy Data):**
```
Packet Data (.npy files) + Time Periods CSV
    ↓
[label_npy_packets.py] → Labeled Packet CSV
    (Filters port 5762, labels based on time periods)
    ↓
[prepare_fl_data.py] → FL Client Data (NPZ files)
```

## Scripts Description

### 1. `find_time_periods.py`

**Purpose**: Identifies distinct time periods in network packet data by detecting gaps between consecutive timestamps.

**Key Features**:
- Loads packet data from `.npy` files
- Detects time gaps larger than a threshold (default: 1 second)
- Identifies separate time periods based on these gaps
- Outputs time periods with start/end times and duration

**Input**:
- `.npy` file containing packet data with timestamps in the first column

**Output**:
- CSV file with time periods: `*_time_periods.csv`
- Contains columns: `period`, `start_time`, `end_time`, `duration_seconds`, `packet_count`

**Usage**:
```python
from find_time_periods import find_time_periods

periods, timestamps = find_time_periods(
    npy_file_path='data.npy',
    gap_threshold_seconds=1.0
)
```

**Parameters**:
- `gap_threshold_seconds`: Minimum gap (in seconds) to consider as a new time period

---

### 2. `label_csv_with_periods.py`

**Purpose**: Labels GPS or IMU CSV data by marking rows as attack (1) or normal (0) based on whether their timestamps fall within detected attack periods. Automatically filters and processes data based on file type.

**Key Features**:
- Supports both GPS and IMU CSV files with automatic type detection
- Parses multiple timestamp formats (full datetime, time-only, numeric seconds)
- Handles time alignment between CSV data and period data
- Labels rows: `1` if timestamp is within any attack period, `0` otherwise
- **GPS-specific filtering**:
  - Removes rows where 5th column (index 4) equals 1
  - Keeps only columns 2, 11, 12, 13 (indices 1, 10, 11, 12)
- **IMU-specific filtering**:
  - Removes columns 1, 3, 4, 12, 13, 15, 16 (indices 0, 2, 3, 11, 12, 14, 15)
- After filtering, timestamp is always in the first column (index 0)
- Output format: timestamp (col 0), features (cols 1-N), label (last column)

**Input**:
- GPS or IMU CSV file (timestamps in second column, index 1, before filtering)
- Time periods (from `.npy` file or CSV file)

**Output**:
- Labeled CSV file: `*_labeled.csv`
- Structure: timestamp (col 0), filtered features, label (last column)
- All rows and columns filtered according to file type

**File Type Detection**:
- Auto-detects from filename: files with 'gps' → GPS, files with 'imu' → IMU
- Can be manually specified using `file_type` parameter

**Usage**:
```python
from label_csv_with_periods import label_csv_with_periods, load_periods_from_csv

# Load time periods
periods = load_periods_from_csv('time_periods.csv')

# Label GPS data (auto-detected from filename)
labeled_df = label_csv_with_periods(
    csv_file_path='gps_data.csv',
    periods=periods
)

# Label IMU data (auto-detected from filename)
labeled_df = label_csv_with_periods(
    csv_file_path='imu_data.csv',
    periods=periods
)

# Manually specify file type
labeled_df = label_csv_with_periods(
    csv_file_path='data.csv',
    periods=periods,
    file_type='gps'  # or 'imu'
)

# Save labeled data
labeled_df.to_csv('gps_data_labeled.csv', index=False, header=False)
```

**Command Line**:
```bash
python label_csv_with_periods.py
```

**Note**: Update file paths in the script's `__main__` section before running. The script will automatically detect file type from the filename.

---

### 3. `label_npy_packets.py`

**Purpose**: Loads packet data from .npy files, filters rows containing port 5762, and labels rows based on whether their timestamps fall within attack time periods.

**Key Features**:
- Loads packet data from .npy files (handles multiple data formats)
- Filters out rows where port 5762 appears in either SrcPort or DstPort
- Labels rows: `1` if timestamp is within any attack period, `0` otherwise
- Supports various .npy file formats (structured arrays, 2D arrays, 1D arrays of objects)
- Automatically converts timestamps to datetime format

**Input**:
- `.npy` file containing packet data with columns: Timestamp, SrcPort, DstPort, Length, MsgID, Protocol
- Time periods CSV file (from `find_time_periods.py`)

**Output**:
- Labeled CSV file: `*_labeled.csv`
- Contains all original columns plus `Label` column (last column)
- Rows with port 5762 are removed

**Usage**:
```python
from label_npy_packets import load_and_label_npy

# Load and label packet data
labeled_df = load_and_label_npy(
    npy_file_path='packet_data.npy',
    periods_csv_path='time_periods.csv',
    output_path='packet_data_labeled.csv'  # optional
)
```

**Command Line**:
```bash
python label_npy_packets.py
```

**Note**: Update file paths in the script's `__main__` section before running.

**Data Format**:
The .npy file should contain packet data with the following columns (in order):
- `Timestamp`: Timestamp for each packet
- `SrcPort`: Source port number
- `DstPort`: Destination port number
- `Length`: Packet length
- `MsgID`: Message ID
- `Protocol`: Protocol type

**Filtering Logic**:
- Removes all rows where `SrcPort == 5762` OR `DstPort == 5762`
- This filters out packets on the attack port (5762)

**Labeling Logic**:
- Label = `1`: Timestamp falls within any time period from the CSV
- Label = `0`: Timestamp does not fall within any time period

---

### 4. `prepare_fl_data.py`

**Purpose**: Prepares labeled GPS data for Federated Learning by creating LSTM sequences and dividing data among clients with consecutive time periods.

**Key Features**:
- Creates sequences of specified length for LSTM training
- Labels sequences: `1` if any element in sequence has label `1`, else `0`
- Time-based train/test split (preserves chronological order)
- Divides data into consecutive time periods for each client
- Ensures all clients have equal (or near-equal) time duration
- Optional feature normalization using StandardScaler

**Input**:
- Labeled CSV file (labels in last column)

**Output**:
- Client data files: `client_000.npz`, `client_001.npz`, ...
- Test data file: `test_data.npz`
- Scaler file: `scaler.npz` (if normalization enabled)

**Usage**:

**Basic Usage**:
```bash
python prepare_fl_data.py \
    --input labeled_gps_data.csv \
    --output fl_data_output \
    --sequence-length 10 \
    --n-clients 5
```

**Advanced Usage with Normalization**:
```bash
python prepare_fl_data.py \
    --input labeled_gps_data.csv \
    --output fl_data_output \
    --sequence-length 20 \
    --stride 5 \
    --n-clients 10 \
    --test-size 0.3 \
    --distribution consecutive \
    --normalize \
    --random-state 42
```

**Parameters**:

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | str | Required | Path to labeled CSV file |
| `--output` | `-o` | str | Required | Output directory for prepared data |
| `--sequence-length` | `-s` | int | 10 | Sequence length for LSTM |
| `--stride` | | int | 1 | Step size for creating sequences |
| `--n-clients` | `-n` | int | 5 | Number of FL clients |
| `--test-size` | | float | 0.2 | Proportion of data for testing |
| `--distribution` | | str | consecutive | Distribution: `consecutive`, `iid`, or `non-iid` |
| `--normalize` | | flag | False | Normalize features using StandardScaler |
| `--random-split` | | flag | False | Use random split instead of time-based |
| `--random-state` | | int | 42 | Random seed for reproducibility |

**Distribution Strategies**:
- **`consecutive`** (default): Each client gets consecutive time periods with equal duration
- **`iid`**: Random shuffle (breaks time continuity, not recommended)
- **`non-iid`**: Sort by label (breaks time continuity, not recommended)

**Output Format**:

Each `.npz` file contains:
- `X`: Sequences array of shape `(n_samples, sequence_length, n_features)`
- `y`: Labels array of shape `(n_samples,)`
- `start_idx`, `end_idx`: (optional) Original indices for consecutive distribution

**Python API**:
```python
from prepare_fl_data import (
    load_labeled_data,
    create_sequences,
    split_train_test,
    divide_among_clients,
    normalize_features,
    save_client_data,
    save_test_data
)

# Load data
features, labels = load_labeled_data('labeled_gps_data.csv')

# Create sequences
X, y = create_sequences(features, labels, sequence_length=10, stride=1)

# Split train/test
X_train, X_test, y_train, y_test = split_train_test(
    X, y, test_size=0.2, time_based=True
)

# Divide among clients
client_data = divide_among_clients(
    X_train, y_train, n_clients=5, distribution='consecutive'
)

# Normalize (optional)
X_train_norm, X_test_norm, client_data_norm, scaler = normalize_features(
    X_train, X_test, client_data
)

# Save
save_client_data(client_data_norm, 'fl_data_output')
save_test_data(X_test_norm, y_test, 'fl_data_output')
```

---

### 5. `example_usage_prepare_fl.py`

**Purpose**: Provides example usage patterns and code snippets for the FL data preparation script.

**Contents**:
- Basic command-line usage examples
- Advanced usage with normalization
- Python API usage examples
- PyTorch data loading examples

---

### 6. `packets_analysis_pyshark.py`

**Purpose**: Analyzes network packets (MAVLink protocol) from pcap files. Used for extracting packet data and timestamps for time period detection.

**Note**: This script is part of the data collection/analysis phase and is typically used before the main pipeline.

---

## Data Format Specifications

### Input CSV Format (GPS or IMU Data)

The input CSV file should have:
- **No header** (or header will be auto-detected)
- **Timestamps in second column** (index 1, before filtering)
- **Features in other columns** (GPS: latitude, longitude, altitude, velocity, etc.; IMU: accelerometer, gyroscope, etc.)
- **Chronological order** (rows ordered by time)

**GPS File Example** (before filtering):
```
row_id, timestamp, col2, col3, col4, col5, ..., col11, col12, col13, ...
0, 1/12/2026 2:25:15 PM, val1, val2, 0, val4, ..., val11, val12, val13, ...
1, 1/12/2026 2:25:16 PM, val1, val2, 1, val4, ..., val11, val12, val13, ...
```

**IMU File Example** (before filtering):
```
col0, col1(timestamp), col2, col3, col4, ..., col11, col12, col13, col14, col15, col16, ...
val0, 1/12/2026 2:25:15 PM, val2, val3, val4, ..., val11, val12, val13, val14, val15, val16, ...
```

**Packet .npy File Format**:
The .npy file should contain packet data with the following structure:
- **Columns**: Timestamp, SrcPort, DstPort, Length, MsgID, Protocol
- **Format**: Can be structured numpy array, 2D array, or 1D array of objects
- **Timestamp**: Should be convertible to pandas datetime format
- **Ports**: Numeric values (will be converted if needed)

Example structure:
```
Timestamp: 2026-01-12 14:25:15.123
SrcPort: 14550
DstPort: 14550
Length: 263
MsgID: 33
Protocol: 0 (UDP) or 1 (TCP)
```

### Data Filtering

The `label_csv_with_periods` function applies file-type-specific filtering:

**GPS Files**:
- **Row filtering**: Removes rows where 5th column (index 4) equals 1
- **Column filtering**: Keeps only columns 2, 11, 12, 13 (indices 1, 10, 11, 12)
- Result: 4 columns (timestamp + 3 feature columns)

**IMU Files**:
- **Column filtering**: Removes columns 1, 3, 4, 12, 13, 15, 16 (indices 0, 2, 3, 11, 12, 14, 15)
- Result: Remaining columns (timestamp + feature columns)

### Labeled CSV Format

After labeling and filtering, the CSV has the following structure:
- **Timestamp in first column** (index 0)
- **Filtered features** (columns 1 to N-1)
- **Label in last column** (index N)

**GPS Example** (after filtering):
```
timestamp, col2, col11, col12, Label
1/12/2026 2:25:15 PM, val2, val11, val12, 0
1/12/2026 2:25:16 PM, val2, val11, val12, 1
```

**IMU Example** (after filtering):
```
timestamp, col2, col5, col6, col7, col8, col9, col10, col14, Label
1/12/2026 2:25:15 PM, val2, val5, val6, val7, val8, val9, val10, val14, 0
1/12/2026 2:25:16 PM, val2, val5, val6, val7, val8, val9, val10, val14, 1
```

**Note**: The exact number of feature columns depends on the original file structure and filtering applied.

**Packet Data Example** (after filtering and labeling):
```
Timestamp, SrcPort, DstPort, Length, MsgID, Protocol, Label
2026-01-12 14:25:15.123, 14550, 14550, 263, 33, 0, 0
2026-01-12 14:25:16.456, 14550, 14550, 264, 33, 0, 1
```

**Note**: 
- Rows with port 5762 in SrcPort or DstPort are removed
- All original columns are preserved (except filtered rows)
- Label column is added at the end

### FL Data Format (NPZ Files)

Each client's data file (`client_XXX.npz`) contains:
- **X**: NumPy array of shape `(n_sequences, sequence_length, n_features)`
- **y**: NumPy array of shape `(n_sequences,)` with binary labels (0 or 1)

Test data file (`test_data.npz`) has the same structure.

---

## Complete Workflow Example

### Step 1: Find Time Periods
```bash
# Edit find_time_periods.py to set npy_file_path
python find_time_periods.py
# Output: mission_2_wp_23_attack_add_wp_5_alt_0005_tcp_port_5762_time_periods.csv
```

### Step 2: Label GPS or IMU Data
```bash
# Edit label_csv_with_periods.py to set file paths
# The script auto-detects file type from filename ('gps' or 'imu')
python label_csv_with_periods.py
# Output: mission_2_wp_23_attack_add_wp_5_alt_0005_gps_labeled.csv
# or: mission_2_wp_23_attack_add_wp_5_alt_0005_imu_labeled.csv
```

**Note**: The script automatically:
- Detects file type (GPS or IMU) from filename
- Applies appropriate filtering (removes rows/columns based on file type)
- Places timestamp in first column after filtering

**Alternative Step 2: Label Packet Data from .npy File**
```bash
# Edit label_npy_packets.py to set file paths
python label_npy_packets.py
# Output: mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv
```

**Note**: This script:
- Loads packet data from .npy file
- Removes rows with port 5762 in SrcPort or DstPort
- Labels rows based on time periods
- Saves labeled CSV with all original columns plus Label

### Step 3: Prepare FL Data
```bash
python prepare_fl_data.py \
    --input mission_2_wp_23_attack_add_wp_5_alt_0005_gps_labeled.csv \
    --output fl_data_output \
    --sequence-length 10 \
    --n-clients 5 \
    --normalize
```

### Step 4: Load Data for Training (PyTorch Example)
```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Load client data
client_data = np.load('fl_data_output/client_000.npz')
X_client = client_data['X']  # (n_samples, seq_len, n_features)
y_client = client_data['y']  # (n_samples,)

# Create PyTorch dataset
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = LSTMDataset(X_client, y_client)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## Key Design Decisions

### 1. Consecutive Time Periods for Clients

The default `consecutive` distribution ensures:
- Each client represents a distinct time period
- Realistic federated learning scenario (different devices/time periods)
- Time continuity is preserved
- All clients have equal duration (within 1 sequence)

### 2. Sequence Labeling Strategy

A sequence is labeled as `1` (attack) if **any** element in the sequence has label `1`. This is appropriate for:
- Attack detection scenarios where any attack in a window is significant
- LSTM models that need to detect attacks within sequences

### 3. Time-Based Train/Test Split

By default, the test set is taken from the end of the time series to:
- Preserve chronological order
- Simulate realistic deployment (train on past, test on future)
- Avoid data leakage from future to past

### 4. Equal Duration for Clients

All clients get the same number of sequences (or within 1) to:
- Ensure fair comparison in FL
- Balance computational load
- Maintain consistent training across clients

---

## Dependencies

Required Python packages:
- `numpy`
- `pandas`
- `scikit-learn` (for StandardScaler and train_test_split)
- `pyshark` (for packets_analysis_pyshark.py only)

Install dependencies:
```bash
pip install numpy pandas scikit-learn pyshark
```

---

## Notes and Best Practices

1. **Data Order**: Ensure CSV data (GPS or IMU) is in chronological order before processing
2. **File Type Detection**: The labeling script auto-detects file type from filename. Include 'gps' or 'imu' in the filename, or manually specify using the `file_type` parameter
3. **Data Filtering**: 
   - GPS files: Rows with 5th column == 1 are removed, only columns 2, 11, 12, 13 are kept
   - IMU files: Columns 1, 3, 4, 12, 13, 15, 16 are removed
   - After filtering, timestamp is always in the first column
4. **Sequence Length**: Choose sequence length based on your LSTM model architecture and temporal patterns
5. **Stride**: Use stride=1 for maximum data (overlapping sequences) or larger stride to reduce redundancy
6. **Normalization**: Always normalize features if using different scales (GPS coordinates, velocities, IMU sensor values, etc.)
7. **Test Size**: Use 20-30% for testing, depending on dataset size
8. **Client Count**: Choose number of clients based on your FL scenario and data size

---

## Troubleshooting

### Issue: "Not enough samples to create sequences"
**Solution**: Reduce `--sequence-length` or use more data

### Issue: "Date mismatch detected" in labeling
**Solution**: The script automatically aligns dates, but verify your timestamp formats

### Issue: Wrong file type detected
**Solution**: Include 'gps' or 'imu' in the filename, or manually specify `file_type='gps'` or `file_type='imu'` in the function call

### Issue: Columns missing after filtering
**Solution**: Ensure your input CSV has the expected number of columns:
- GPS files should have at least 13 columns (for columns 2, 11, 12, 13)
- IMU files should have at least 16 columns (for proper column removal)

### Issue: Clients have very different label distributions
**Solution**: This is expected with consecutive distribution. Consider using larger sequence length or more clients

### Issue: Memory errors with large datasets
**Solution**: 
- Reduce `--stride` to create fewer sequences
- Process data in batches
- Use smaller `--sequence-length`

---

## File Structure

```
wp_inject_attack_label/
├── find_time_periods.py              # Time period detection
├── label_csv_with_periods.py         # GPS/IMU CSV data labeling
├── label_npy_packets.py              # Packet .npy data labeling
├── prepare_fl_data.py                # FL data preparation
├── example_usage_prepare_fl.py       # Usage examples
├── packets_analysis_pyshark.py       # Packet analysis (optional)
└── DATA_PROCESSING_PIPELINE.md       # This file
```

---

## Contact and Support

For issues or questions about the data processing pipeline, please refer to the script docstrings or modify the scripts according to your specific requirements.
