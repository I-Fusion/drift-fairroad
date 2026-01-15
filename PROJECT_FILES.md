# Project Files - What to Include for Deployment

This document lists all files needed to deploy the Federated Learning system.

## ✅ Essential Files (MUST INCLUDE)

### Core System Files

```
FL/
├── config.py                      # Main configuration file
├── run_fl_system.py               # Single-command execution script
├── fl_client.py                   # FL client implementation
├── fl_server.py                   # FL server implementation
├── data_preprocessing.py          # GPS+IMU data fusion module
├── aggregation.py                 # Aggregation strategies
├── models/
│   ├── __init__.py               # Models package init
│   └── lstm_model.py             # Default LSTM model
└── requirements.txt               # Python dependencies
```

**Total:** 9 files

### Data Files (User Provides)

```
├── data/
│   ├── your_gps_file.csv         # Your GPS data
│   └── your_imu_file.csv         # Your IMU data
```

### Auto-Generated (During Runtime)

```
├── checkpoints/                   # Created automatically
│   └── server_round_*.pt         # Model checkpoints
```

## 📚 Documentation Files (OPTIONAL)

```
├── README.md                      # User guide
├── PROJECT_FILES.md               # This file
└── ARCHITECTURE.md                # Technical documentation
```

## ❌ Files NOT Needed (Can Delete)

### Legacy Files

```
├── base_model.py                  # Removed - no longer needed
├── client.py                      # Old version - use fl_client.py
├── server.py                      # Old version - use fl_server.py
├── model.py                       # Old version - use models/lstm_model.py
├── data_utils.py                  # Old version - use data_preprocessing.py
├── server_client.py               # Old entry point - use run_fl_system.py
```

### Old Documentation

```
├── README_OLD.md                  # Superseded by README.md
├── README_NEW.md                  # Consolidated into README.md
├── MIGRATION_GUIDE.md             # Not needed anymore
├── QUICKSTART.md                  # Consolidated into README.md
├── START_HERE.md                  # Consolidated into README.md
├── SYSTEM_SUMMARY.md              # Consolidated into README.md
```

### Docker Files (Optional - If Not Using Docker)

```
├── Dockerfile                     # Only if using Docker
├── docker-compose.yml             # Only if using Docker
├── .dockerignore                  # Only if using Docker
```

### Generated Files

```
├── fl_architecture.png            # Generated diagram
├── fl_sequence_diagram.png        # Generated diagram
├── generate_architecture_diagram.py  # Diagram generator
```

### Misc

```
├── .gitignore                     # For git only
```

## 📦 Minimal Deployment Package

For a clean deployment, include only these:

```
FL/
├── config.py
├── run_fl_system.py
├── fl_client.py
├── fl_server.py
├── data_preprocessing.py
├── aggregation.py
├── models/
│   ├── __init__.py
│   └── lstm_model.py
├── requirements.txt
├── README.md
└── data/
    ├── your_gps.csv
    └── your_imu.csv
```

**That's 11 files total** (9 code + 1 doc + 1 requirements)

## 🚀 Deployment Steps

### 1. Create Project Directory

```bash
mkdir my_fl_project
cd my_fl_project
```

### 2. Copy Essential Files

```bash
# Copy core files
cp config.py run_fl_system.py fl_client.py fl_server.py data_preprocessing.py aggregation.py requirements.txt ./

# Copy models directory
cp -r models/ ./

# Copy documentation (optional)
cp README.md ./
```

### 3. Add Your Data

```bash
mkdir data
cp /path/to/your_gps.csv data/
cp /path/to/your_imu.csv data/
```

### 4. Update Configuration

```bash
nano config.py
# Update GPS_FILE and IMU_FILE paths
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run

```bash
python run_fl_system.py
```

## 📝 File Dependencies

### Dependency Graph

```
run_fl_system.py
├── config.py
├── fl_server.py
│   ├── config.py
│   ├── aggregation.py
│   └── models/lstm_model.py
└── fl_client.py
    ├── config.py
    ├── data_preprocessing.py
    └── models/lstm_model.py
```

### What Each File Does

| File | Purpose | Dependencies |
|------|---------|--------------|
| `config.py` | Central configuration | None |
| `run_fl_system.py` | Main execution script | config, fl_server, fl_client |
| `fl_server.py` | FL server | config, aggregation, models/* |
| `fl_client.py` | FL client | config, data_preprocessing, models/* |
| `data_preprocessing.py` | Data fusion | None (standalone) |
| `aggregation.py` | Aggregation strategies | None (standalone) |
| `models/lstm_model.py` | LSTM model | torch |
| `requirements.txt` | Python packages | None |

## 🔧 Customization Files

To customize the system:

### To Change Model

**Add:** `models/your_model.py`
**Modify:** `config.py` (MODEL_PATH, MODEL_CLASS)

### To Change Aggregation

**Modify:** `aggregation.py` (add new method)
**Modify:** `config.py` (AGGREGATION_STRATEGY)

### To Change Data Format

**Modify:** `data_preprocessing.py` (DataPreprocessor class)
**Modify:** `config.py` (GPS_FEATURES, IMU_FEATURES)

## 📊 File Sizes (Approximate)

```
config.py                    ~3 KB
run_fl_system.py             ~4 KB
fl_client.py                 ~7 KB
fl_server.py                 ~7 KB
data_preprocessing.py        ~7 KB
aggregation.py               ~5 KB
models/lstm_model.py         ~3 KB
requirements.txt             <1 KB
README.md                    ~15 KB

Total: ~52 KB (code only)
```

## ✨ Summary

### Absolute Minimum to Run:
- 9 code files
- 1 requirements.txt
- Your 2 data files (GPS + IMU)

### Recommended to Include:
- README.md (user guide)
- PROJECT_FILES.md (this file)

### Can Delete Safely:
- All files listed in "Files NOT Needed" section
- Architecture diagrams (unless needed for documentation)
- Docker files (if not using Docker)
- Legacy code files

---

**Keep it simple. Deploy lean.** 🚀
