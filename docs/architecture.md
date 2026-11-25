# ACFL Architecture Specification

## Overview

ACFL (Adversarially-Conditioned Federated Learning) is a server-side defense framework that extends the standard federated learning training loop at three points, designed as modular plugins compatible with the drift-fairroad client-server architecture.

```
┌─────────────────────────────────────────────────────────────────┐
│                     ACFL Defense Framework                       │
├─────────────────────────────────────────────────────────────────┤
│  Client 1 ──► Update ──┐                                        │
│  Client 2 ──► Update ──┼──► Layer 1: Update Validation          │
│  Client N ──► Update ──┘         │                              │
│                                  ▼                              │
│                    Layer 2: Trust-Weighted Aggregation          │
│                                  │                              │
│                                  ▼                              │
│                    Global Model (Hardened)                      │
│                                  │                              │
│                                  ▼                              │
│                    Layer 3: Spoofing Detection (Inference)      │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
ACFL/
├── acfl/
│   ├── config.py              # ACFLConfig dataclass
│   ├── weight_utils.py        # Weight flattening, robust statistics
│   ├── layer1_validation.py   # Counterfactual update validation
│   ├── layer2_aggregation.py  # Byzantine-tolerant aggregation
│   ├── layer3_detection.py    # Control signal spoofing detector
│   ├── defense.py             # ACFLDefense orchestrator
│   └── acfl_server.py         # drift-fairroad server integration
├── drift-fairroad/            # Evaluation testbed (v2)
├── run_acfl_system.py         # End-to-end runner
├── tests/                     # Unit tests
└── docs/                      # Architecture and threat model
```

## Layer 1: Counterfactual Update Validation

Operates at the server before aggregation. Three complementary signals:

| Component | Method | Purpose |
|-----------|--------|---------|
| Subspace Projection | Randomized SVD (rSVD) | Detect backdoored updates with anomalous projection structure |
| Round-Consistency | Temporal trajectory scoring | Detect abrupt inter-round discontinuities from poisoning |
| Counterfactual Probes | Attack-scenario probe evaluation | Measure operational harm on GNSS/ADS-B/MAVLink scenarios |

Output: per-client trust scores in `[min_trust_weight, 1.0]` via soft-filtering.

## Layer 2: Mission-Context Trust-Weighted Aggregation

Two-stage robust aggregation:

**Stage 1 — Coordinate-Wise Robust Pre-Filtering**
- Median and trimmed-mean statistics per coordinate
- Dynamic threshold from empirical benign-round history
- Soft-filter: blend suspicious updates toward robust statistic

**Stage 2 — Mission-Context Trust Weighting**
- Combines statistical consistency with cyber-physical evidence:
  - GNSS/IMU consistency
  - MAVLink command-state consistency
  - Communication link health
  - Local validation loss
  - Temporal participation history

## Layer 3: Spoofing Countermeasures

Uses the hardened global model as an operational detector at inference time:
- Model-based anomaly scoring
- Cross-modal inconsistency between commanded trajectory and GPS/IMU state
- Triggers `safe_hold` or `continue` based on configurable threshold

## Integration Points

ACFL integrates with drift-fairroad via:

1. **`ACFLServerTimeSeries`** — Drop-in replacement for `FLServerTimeSeries`
2. **`/detect_spoofing`** — Layer 3 inference endpoint
3. **Client headers** — Mission-context metadata via `X-GNSS-IMU-Consistency`, etc.
4. **Checkpoint exports** — `acfl_trust_scores.json`, `acfl_round_reports.json`

## Configuration

All parameters are in `ACFLConfig` (`acfl/config.py`). Key tunables:

- `subspace_dim`, `subspace_anomaly_threshold` — Layer 1 spatial analysis
- `trim_fraction`, `coordinate_deviation_threshold` — Layer 2 robust filtering
- `spoofing_detection_threshold` — Layer 3 detection sensitivity

## Ablation Support

Each layer can be independently disabled for controlled experiments:

```bash
python run_acfl_system.py --no-layer1 ...  # Layer 2+3 only
python run_acfl_system.py --no-layer2 ...  # Layer 1 only
python run_acfl_system.py --no-layer3 ...  # Training defense only
```
