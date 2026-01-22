# Federated Learning - Convolutional Autoencoder (FL-CAE)

## Overview

This directory contains a **Federated Learning system with Convolutional Autoencoder** for image denoising and reconstruction using adversarial/noisy inputs.


---

## Quick Start

### 1. Run FL Training
```bash
python run_fl_cae_system.py --num-rounds 10
```

### 2. Evaluate Model
```bash
python evaluate_cae.py \
    --model-path checkpoints/server_round_10.pt \
    --clean-dir ../data/payload/images/clean \
    --noisy-dir ../data/payload/images/noise \
    --num-clients 3
```

---

## System Architecture

- **Model**: Convolutional Autoencoder (12M parameters, 128-dim latent space)
- **Task**: Reconstruct clean images from noisy/adversarial inputs
- **Dataset**: 500 clean/noisy image pairs distributed across 3 clients
- **FL Algorithm**: FedAvg (Federated Averaging)
- **Metrics**: MSE (Mean Squared Error) + SSIM (Structural Similarity Index)

---

## Key Files

### Core System
- `config.py` - Central configuration (set `TASK_TYPE='autoencoder'`)
- `fl_server.py` - FL server for aggregation
- `fl_client_cae.py` - FL client for CAE training
- `aggregation.py` - FedAvg implementation

### Model & Data
- `models/cae_model.py` - Convolutional Autoencoder architecture
- `data_preprocessing_cae.py` - Clean/noisy image pair loader

### Training & Evaluation
- `run_fl_cae_system.py` - Complete FL system orchestrator
- `evaluate_cae.py` - MSE and SSIM evaluation
- `test_cae_system.py` - Component testing

## Configuration

Edit `config.py`:

```python
# Task type
TASK_TYPE = 'autoencoder'

# Model parameters
MODEL_CONFIG = {
    'latent_dim': 128,
    'learning_rate': 0.001
}

# FL parameters
NUM_CLIENTS = 3
MIN_CLIENTS = 2
AGGREGATION_STRATEGY = 'fedavg'
BATCH_SIZE = 32
EPOCHS_PER_ROUND = 1
```

---

## Requirements

```bash
pip install torch torchvision
pip install scikit-image  # For SSIM metric
pip install aiohttp        # For FL communication
pip install matplotlib     # For plots
pip install tqdm           # For progress bars
```

---
