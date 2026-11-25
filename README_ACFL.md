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
| `fl-time-series/models/tiny_packet_cnn.py` | Drone-scale TinyPacketCNN (~14k params) |
| `fl-time-series/config_classification_drone.py` | Packet FL profile that selects TinyPacketCNN |
| `fl-time-series/checkpoints/acfl_examples/` | Example trained ACFL / FedAvg `.pt` checkpoints |
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

Run from the **drift-fairroad repository root** (after the copy above).

Drone / ACFL paper profile (**TinyPacketCNN**):

```bash
# Ensure fl-time-series is on PYTHONPATH, or cd into it first
python run_acfl_system.py \
    --data-dir data/train/packets \
    --config config_classification_drone \
    --learning-mode classification \
    --num-rounds 12
```

Legacy LSTM packet classification / GPS-IMU regression:

```bash
# Packet classification (LSTM)
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

## Generate experiment / mission data

SITL collection and dataset builders (run from this repo root; needs ArduPilot SITL):

```bash
# Waypoint pools + SITL flights → labeled packet CSVs
bash collect_validate_missions.sh
bash collect_drone_missions.sh

# Merge run folders into agent-centric FL packet layout
python3 build_agent_mission_dataset.py \
  --train-runs data/sitl_dataset/runs \
  --extra-runs data/sitl_validate/runs \
  --extra-runs data/sitl_drone_extra/runs \
  --output data/sitl_dataset/fl/packets --clean

# Layer-1 counterfactual probes (.pt)
python3 generate_probe_set.py --mode classification \
  --packet-file data/network_packets/mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv \
  --output fl-time-series/checkpoints/acfl_probes.pt

# Optional: bootstrap more labeled rows from existing templates
python3 generate_synthetic_data.py --mode packets --help
```

Low-level harness: `simulations/collect_dataset.py`, `simulations/scripts/generate_missions.py`,
`run_simulation.py`, `attacks_sitl.py`, `convert_logs.py`, `packet_pipeline.py`
(see `simulations/SIMULATION_HARNESS.md`).

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
