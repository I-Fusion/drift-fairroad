# ACFL — Adversarially-Conditioned Federated Learning

Clean deliverable of the ACFL defense for the
[I-Fusion/drift-fairroad](https://github.com/I-Fusion/drift-fairroad)
evaluation testbed (IEEE AESS Challenge 3).

This folder is intended to be **copied into the root of `drift-fairroad`**
(alongside `fl-time-series/`, `data/`, and `docker/`). It does not replace
the existing FL stack; it adds a poisoning-resilient aggregation path and
a spoofing detection layer on top of it.

## What this adds

| Path | Role |
|------|------|
| `acfl/` | Defense package (Layers 1–3) |
| `run_acfl_system.py` | End-to-end FL runner that uses `acfl.acfl_server` |
| `evaluate_acfl.py` | Trust-score / round-report evaluation helpers |
| `docs/` | Architecture and threat model |
| `tests/` | Unit tests for the ACFL package |
| `requirements-acfl.txt` | Extra Python deps beyond `requirements.txt` |

### Defense layers

1. **Layer 1 — Counterfactual update validation**  
   Subspace / round-consistency checks plus attack-probe harm scoring
   before aggregation.
2. **Layer 2 — Trust-weighted aggregation**  
   Byzantine-tolerant mix of robust statistics and cyber-physical trust.
3. **Layer 3 — Spoofing countermeasures**  
   Hardened global model for waypoint injection / GNSS spoof detection.

## Install into drift-fairroad

```bash
# From a clone of I-Fusion/drift-fairroad (v2 recommended):
git clone -b v2 https://github.com/I-Fusion/drift-fairroad.git
cd drift-fairroad

# Copy this deliverable into the repo root (merge, do not nest):
rsync -a /path/to/ACFL_deliverable/ ./

python -m venv fl_env
source fl_env/bin/activate
pip install -r requirements.txt
pip install -r requirements-acfl.txt
```

## Quick start

Run from the **drift-fairroad repository root** (after the copy above):

```bash
# Packet classification
python run_acfl_system.py \
    --data-dir data/train/packets \
    --config config_classification \
    --learning-mode classification \
    --num-rounds 5

# GPS/IMU regression
python run_acfl_system.py \
    --data-dir data/train/gps-imu \
    --config config_regression \
    --learning-mode regression \
    --num-rounds 5
```

Ablations:

```bash
python run_acfl_system.py --no-layer2 --no-layer3 ...   # Layer 1 only
python run_acfl_system.py --no-layer1 ...               # Layer 2 path
python run_acfl_system.py --malicious-clients client_2,client_3 ...
```

## Tests

```bash
pytest tests/ -v
```

## Outputs

During training ACFL writes under `fl-time-series/checkpoints/`:

- `acfl_trust_scores.json` — per-round client trust
- `acfl_round_reports.json` — validation / aggregation diagnostics

## Scope of this deliverable

**Included:** ACFL package, runner, evaluator, architecture/threat docs,
core unit tests.

**Not included:** large SITL datasets, journal manuscript sources,
long attack-sweep driver scripts, local virtualenvs, and experimental
result caches. Those stay in the development workspace and are not
required to run the defense on the drift-fairroad testbed.
