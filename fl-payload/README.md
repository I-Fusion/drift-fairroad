# Federated Learning - Convolutional Autoencoder (FL-CAE)

## Overview

This directory contains a **Federated Learning system with Convolutional Autoencoder** for image denoising and reconstruction using adversarial/noisy inputs. The system includes comprehensive evaluation aligned with the root `evaluate_fl_system.py` dimensions: reconstruction quality, drift resistance, convergence stability, efficiency, and non-IID resilience.

**Run from project root:** use `cd fl-payload; python ...` so that paths in this directory (e.g. `../data/`) resolve correctly. (On Windows PowerShell use `;`; on bash/zsh you can use `&&` instead.)

---

## Quick Start

### 1. Run FL Training

From **project root** (use `;` on Windows PowerShell, or `&&` on bash/zsh):

```bash
cd fl-payload; python run_fl_cae_system.py --num-rounds 10
```

This will:
- Start the FL server on `localhost:5544`
- Launch 3 clients (configurable in `config.py`)
- Train for 10 rounds with FedAvg aggregation
- Save checkpoints to `checkpoints/server_round_*.pt`
- Generate a training loss plot

### 2. Evaluate Model

From **project root** (paths relative to `fl-payload/` when run from inside that directory). Use `;` on PowerShell, `&&` on bash/zsh.

**Single checkpoint evaluation:**
```bash
cd fl-payload; python evaluate_cae.py \
    --model-path checkpoints/server_round_10.pt \
    --clean-dir ../data/payload/images/clean \
    --noisy-dir ../data/payload/images/noise \
    --num-clients 3 \
    --output-dir evaluation_results
```

**Multi-round evaluation (comprehensive):**
```bash
cd fl-payload; python evaluate_cae.py \
    --checkpoint-dir checkpoints \
    --clean-dir ../data/payload/images/clean \
    --noisy-dir ../data/payload/images/noise \
    --num-clients 3 \
    --output-dir evaluation_results
```

The multi-round evaluation generates:
- `evaluation_report.json` - Complete metrics in JSON format
- `evaluation_report.txt` - Human-readable report
- `cae_metrics_over_rounds.png` - MSE and SSIM over rounds
- `cae_client_heterogeneity.png` - Per-client performance (last round)
- `cae_drift_over_rounds.png` - Drift analysis (round 0 vs later rounds)

---

## System Architecture

- **Model**: Convolutional Autoencoder
  - **Full model**: ~12M parameters, 128-dim latent space (`models/cae_model.py`)
  - **Small model** (for testing): ~0.76M parameters, 32-dim latent space (`models/cae_model_small.py`)
- **Task**: Reconstruct clean images from noisy/adversarial inputs
- **Dataset**: 500 clean/noisy image pairs distributed across 3 clients
- **FL Algorithm**: FedAvg (Federated Averaging)
- **Metrics**: MSE (Mean Squared Error) + SSIM (Structural Similarity Index)

---

## Key Files

### Core System
- `config.py` - Central configuration (set `TASK_TYPE='autoencoder'`)
- `fl_server.py` - FL server for aggregation (with async executor for responsiveness)
- `fl_client_cae.py` - FL client for CAE training (with retry logic and staggered starts)
- `aggregation.py` - FedAvg implementation

### Model & Data
- `models/cae_model.py` - Full Convolutional Autoencoder architecture (~12M params)
- `models/cae_model_small.py` - Small CAE for testing timeouts (~0.76M params)
- `data_preprocessing_cae.py` - Clean/noisy image pair loader

### Training & Evaluation
- `run_fl_cae_system.py` - Complete FL system orchestrator (starts server, clients, monitors)
- `evaluate_cae.py` - **Comprehensive evaluation** with all FL dimensions:
  - Reconstruction quality (MSE/SSIM, best/final rounds)
  - Drift resistance (performance on fixed validation set over rounds)
  - Convergence stability (round where metrics stabilize)
  - Efficiency (model size, checkpoint sizes, communication overhead)
  - Non-IID resilience (client heterogeneity analysis)
- `test_cae_system.py` - Component testing

---

## Configuration

Edit `config.py`:

```python
# Task type
TASK_TYPE = 'autoencoder'

# Model selection
MODEL_PATH = 'models.cae_model'  # or 'models.cae_model_small' for testing
MODEL_CLASS = 'CAEFederatedModel'
MODEL_CONFIG = {
    'latent_dim': 128,  # 32 for small model
    'learning_rate': 0.001
}

# FL parameters
NUM_CLIENTS = 3
MIN_CLIENTS = 2  # Minimum submissions per round to aggregate
CLIENT_STAGGER_SEC = 10  # Delay per client index before round 1 (0, 10, 20s for 3 clients)
AGGREGATION_STRATEGY = 'fedavg'
BATCH_SIZE = 32
EPOCHS_PER_ROUND = 1

# Server
SERVER_HOST = 'localhost'
SERVER_PORT = 5544
```

**Note**: The server aggregates when at least `MIN_CLIENTS` have submitted (not necessarily all registered clients), allowing progress even if one client fails.

---

## Evaluation Dimensions

The `evaluate_cae.py` script provides comprehensive evaluation aligned with the root `evaluate_fl_system.py`:

### 1. Reconstruction Quality & Responsiveness
- **MSE and SSIM** (overall and per-client)
- **Best round** identification (lowest MSE, highest SSIM)
- **Final round** performance

### 2. Robustness & Non-IID Resilience
- **Client heterogeneity**: Mean, std, min, max of MSE/SSIM across clients
- **Non-IID resilience score**: Coefficient of variation and resilience metric (1 - cv)
- Per-client performance visualization

### 3. Drift Resistance & Model Stability
- **Drift analysis**: MSE/SSIM on fixed validation set at round 0 vs each round
- **Convergence stability**: Round at which metrics stabilize (within epsilon of final)
- Variance in tail rounds (last N rounds)

### 4. Mission Continuity
- Placeholder for reconstruction quality thresholds (N/A for false alarms)

### 5. Efficiency & Practicality
- **Model size**: Total parameters, model size (MB)
- **Checkpoint sizes**: Mean checkpoint file size
- **Communication overhead**: Estimated MB per round (checkpoint size × 2 × num_clients)

---

## Evaluation Output

### JSON Report (`evaluation_report.json`)
Structured data with:
- `summary`: Best/final rounds and metrics
- `reconstruction_quality`: Final MSE/SSIM, best rounds
- `drift_resistance`: MSE/SSIM drift from round 0
- `convergence_stability`: Stabilization round and variance
- `efficiency`: Model size, checkpoint sizes, communication overhead
- `non_iid_resilience`: Client heterogeneity metrics
- `client_heterogeneity`: Per-client statistics
- `evaluation_results`: Detailed per-round, per-client results

### Text Report (`evaluation_report.txt`)
Human-readable summary with all key metrics.

### Plots (multi-round evaluation only)
- **`cae_metrics_over_rounds.png`**: MSE and SSIM trends over rounds
- **`cae_client_heterogeneity.png`**: Bar charts of per-client MSE/SSIM (last round)
- **`cae_drift_over_rounds.png`**: Drift visualization (positive = worse than round 0)

---

## Command-Line Options

### `evaluate_cae.py`

**Required (one of):**
- `--model-path PATH` - Single checkpoint path
- `--checkpoint-dir DIR` - Directory with `server_round_*.pt` for multi-round evaluation

**Data:**
- `--clean-dir DIR` - Clean images directory (default: `../data/payload/images/clean`)
- `--noisy-dir DIR` - Noisy images directory (default: `../data/payload/images/noise`)
- `--num-clients N` - Number of clients (default: 3)

**Output:**
- `--output-dir DIR` - Output directory (default: `evaluation_results`)

**Options:**
- `--rounds R1 R2 ...` - Specific rounds to evaluate (default: all)
- `--no-client-eval` - Skip per-client evaluation (faster)
- `--save-reconstructions` - Save sample reconstruction images

### `run_fl_cae_system.py`

- `--num-rounds N` - Number of FL rounds (default: 10)

---

## Debugging & Testing

### Using the Small Model

To test whether model size causes timeouts, switch to the small model in `config.py`:

```python
MODEL_PATH = 'models.cae_model_small'
MODEL_CONFIG = { 'latent_dim': 32, 'learning_rate': 0.001 }
```

The small model (~0.76M params, ~3MB) is much faster to transfer than the full model (~12M params, ~48MB).

### Debug Output

All scripts include debug prints prefixed with:
- `[FL-SERVER]` - Server operations (register, get_model, submit_update, aggregation)
- `[FL-CLIENT client_X]` - Client operations (register, get_model, train, submit)
- `[RUNNER]` - System runner operations (start server/clients, monitor)

These help trace the FL flow and identify bottlenecks.

---

## Requirements

```bash
pip install torch torchvision
pip install scikit-image  # For SSIM metric
pip install aiohttp        # For FL communication
pip install matplotlib     # For plots
pip install tqdm           # For progress bars
pip install numpy          # For numerical operations
```

---

## Troubleshooting

### Timeouts
- **Server**: Uses thread executor for pickle/aggregation to keep event loop responsive
- **Client**: Long timeouts (900s total, 600s read) and retry logic (5 attempts)
- **Stagger**: Clients start with delays (10s per index) to avoid simultaneous requests

### Aggregation Not Happening
- Server aggregates when **at least `MIN_CLIENTS`** have submitted (default: 2)
- If one client fails, others can still make progress
- Check server logs for "all X submitted -> aggregating"

### Evaluation Errors
- Ensure checkpoint directory contains `server_round_*.pt` files
- For multi-round evaluation, use `--checkpoint-dir` (not `--model-path`)
- Check that clean/noisy image directories exist and contain matching pairs

---

## Example Workflow

```bash
# 1. Train FL system (10 rounds)
python run_fl_cae_system.py --num-rounds 10

# 2. Comprehensive evaluation (all rounds)
python evaluate_cae.py \
    --checkpoint-dir checkpoints \
    --clean-dir ../data/payload/images/clean \
    --noisy-dir ../data/payload/images/noise \
    --num-clients 3 \
    --output-dir evaluation_results

# 3. View results
cat evaluation_results/evaluation_report.txt
# Or open evaluation_results/evaluation_report.json
```

---

## Notes

- **Server responsiveness**: Heavy operations (pickle, aggregation, checkpoint save) run in thread executor so `/status` endpoint stays responsive
- **Client resilience**: Clients retry failed submissions (5 attempts) and continue even if one submit fails
- **Checkpoint format**: Server checkpoints contain only `model_state_dict` and `round` (no optimizer state)
- **Model compatibility**: Both `cae_model.py` and `cae_model_small.py` can load server checkpoints (optimizer state optional)
