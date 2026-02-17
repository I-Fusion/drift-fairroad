# Federated Learning Evaluation – `evaluate_fl_system.py`

This document describes how to use `evaluate_fl_system.py` to evaluate trained federated learning (FL) models for **classification** (e.g. attack vs normal from packet sequences) and **regression** (GPS/IMU time-series). It supports full evaluation of detection quality, robustness, drift, mission continuity, and efficiency for cyber anomaly detection (GPS spoofing, jamming, etc.).

---

## 1. Overview

`evaluate_fl_system.py` can:

- **Load checkpoints** from the server (and optionally client checkpoints)
- **Load test data** from prepared client CSVs or `test_data.npz`
- **Evaluate** classification or regression performance
- **Plot** metrics over rounds and client heterogeneity
- **Generate reports** (JSON and human-readable text) into an output directory

**Typical usage** — run from **project root**. Checkpoint and output dirs default to `fl-time-series/checkpoints` and `fl-time-series/evaluation_results`:

```bash
python fl-time-series/evaluate_fl_system.py --data-dir data/validate/packets --task classification --config config_classification
```

To override: `--checkpoint-dir <path>` and/or `--output-dir <path>`.

On Windows (PowerShell), use backtick for line continuation:

```powershell
python fl-time-series/evaluate_fl_system.py `
  --data-dir data/validate/packets `
  --task classification `
  --config config_classification
```

---

## 2. Required Inputs

- **`--checkpoint-dir`** (optional; default: `fl-time-series/checkpoints`)  
  Directory containing server checkpoints. Files should be named `server_round_XX.pt` (from `fl-time-series/run_fl_system_time_series.py`).

- **`--data-dir`** (optional but recommended)  
  Directory with evaluation data:
  - **Packet classification**: `packet_client_000.csv`, `packet_client_001.csv`, … (e.g. from `prepare_fl_data_for_run_fl.py` → `data/validate/packets`).
  - **GPS/IMU regression**: `gps_client_000.csv`, `imu_client_000.csv`, etc.

- **`--config`** (required in practice; default: `config_classification`)  
  Python config module name. In this repo, use:
  - **Classification**: `config_classification` (packet features, labels, windowing).
  - **Regression**: `config_regression` (GPS/IMU).

---

## 3. Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint-dir` | Directory with `server_round_*.pt` (default: `fl-time-series/checkpoints`). |
| `--data-dir` | Directory with client data files. |
| `--task` | `classification`, `regression`, or `auto` (default: `auto`). |
| `--test-split` | Fraction of data used for test (default: `0.2`). |
| `--output-dir` | Where to save reports and plots (default: `fl-time-series/evaluation_results`). |
| `--device` | `cuda` or `cpu` (default: auto-detect). |
| `--rounds` | Specific rounds to evaluate (e.g. `--rounds 10 20 30`); default: all. |
| `--config` | Config module (e.g. `config_classification`, `config_regression`). |
| `--evaluate-clients` / `--no-client-eval` | Enable or disable per-client model evaluation. |
| `--attack-onset-file` | JSON: single list (`attack_onset_indices`) or per-client keys (`client_1`, `client_2`, …). |
| `--attack-onset-dir` | Directory with one JSON per client (e.g. `attack_onset_client_1.json`). |
| `--max-false-alarms-per-hour` | Operational limit for **mission continuity**. |
| `--window-duration-sec` | Seconds per window (for false alarms per hour). |
| `--poisoning-results` | JSON with clean vs poisoned metrics (**degradation under poisoning**). |
| `--backdoor-results` | JSON with `backdoor_success_rate` (**backdoor robustness**). |

---

## 4. Evaluation Dimensions (Cyber Anomaly Detection & FL Robustness)

The evaluator reports:

1. **Detection accuracy and responsiveness**  
   AUROC, precision, recall, F1, **false alarm rate** (FP/(FP+TN)), and optionally **time to detect** (from `--attack-onset-file`).

2. **Robustness to malicious or poisoned updates**  
   **Degradation under poisoning** (from `--poisoning-results`), **backdoor success rate** (from `--backdoor-results`), and **non-IID resilience** (from client heterogeneity).

3. **Drift resistance and model stability**  
   **Drift** (accuracy on fixed validation set at round 0 vs later rounds; plot: `drift_over_rounds.png`), **convergence stability** (round at which metric stabilizes, variance in last N rounds).

4. **Mission continuity**  
   False alarm rate; with `--max-false-alarms-per-hour` and `--window-duration-sec`, reports whether **false alarms per hour** are within limit.

5. **Efficiency and practicality**  
   Model size (parameters, MB), checkpoint size, **communication overhead per round** (MB).

---

## 5. Optional JSON Inputs (Advanced)

### Time-to-detect (`--attack-onset-file` or `--attack-onset-dir`)

**Single list (legacy):** one JSON file with test-set indices for one test set:

```json
{
  "attack_onset_indices": [100, 500, 1200]
}
```

**Per-client (one JSON file):** keys must match test set names (`client_1`, `client_2`, …):

```json
{
  "client_1": [0, 42, 100],
  "client_2": [10, 55],
  "client_3": [3, 20]
}
```

**Per-client (directory):** use `--attack-onset-dir` with one JSON per client, e.g. `attack_onset_client_1.json`, `attack_onset_client_2.json`. Each file contains `{"attack_onset_indices": [...]}`.

The evaluator reports mean/min delay (in windows) from each onset to the first positive prediction, per client when per-client data is provided.

### Degradation under poisoning (`--poisoning-results`)

JSON with accuracy (or `final_accuracy`) for clean and poisoned runs. Example:

```json
{
  "clean": { "accuracy": 0.92 },
  "poisoned_0.2": { "accuracy": 0.85 },
  "poisoned_0.5": { "accuracy": 0.78 }
}
```

The report will show accuracy drop vs clean for each key.

### Backdoor robustness (`--backdoor-results`)

JSON with backdoor success rate (and optional trigger accuracy). Example:

```json
{
  "backdoor_success_rate": 0.15,
  "trigger_accuracy": 0.98
}
```

---

## 6. Typical Usage Patterns

### 6.1. Packet classification (`config_classification`)

1. **Train** (from project root):

```bash
python fl-time-series/run_fl_system_time_series.py --data-dir data/train/packets --config config_classification --learning-mode classification
```

2. **Evaluate**:

```bash
python fl-time-series/evaluate_fl_system.py --data-dir data/train/packets --task classification --config config_classification
```

This loads all `server_round_*.pt` checkpoints from `fl-time-series/checkpoints/`, builds sliding windows from `packet_client_*.csv` using `config_classification` (WINDOW_SIZE, OVERLAP, PACKET_FEATURES, LABEL_COL), and writes metrics and plots to `fl-time-series/evaluation_results/`.

### 6.2. GPS/IMU regression (`config_regression`)

```bash
python fl-time-series/evaluate_fl_system.py --data-dir data/validate/gps-imu --task regression --config config_regression
```

Uses `gps_client_*.csv` / `imu_client_*.csv`, builds time-series windows, and reports RMSE, MAE, R².

### 6.3. With mission continuity and time-to-detect

```bash
python fl-time-series/evaluate_fl_system.py --data-dir data/train/packets --task classification --config config_classification --max-false-alarms-per-hour 5 --window-duration-sec 1.0 --attack-onset-file path/to/attack_onsets.json
```

---

## 7. Outputs

With `--output-dir` (default: `fl-time-series/evaluation_results/`):

- **`evaluation_report.json`**: Task, paths, per-round metrics, detection responsiveness, drift, convergence, mission continuity, efficiency, non-IID resilience, robustness (poisoning/backdoor if provided).
- **`evaluation_report.txt`**: Human-readable version of the above.
- **Plots**:  
  `classification_metrics_<test_name>.png`, `regression_metrics_<test_name>.png`,  
  `client_heterogeneity_classification.png` / `client_heterogeneity_regression.png`,  
  `client_performance_distribution.png`,  
  `confusion_matrices_<test_name>.png` (classification),  
  `predictions_vs_actual_<test_name>.png` (regression),  
  `drift_over_rounds.png` (classification).

---

## 8. Notes and Tips

- **Import path**: The script runs from `fl-time-series/` and adds that directory and project root to `sys.path`; run from the **project root** (e.g. `python fl-time-series/evaluate_fl_system.py ...`).
- **Data format**: Packet CSVs must match `config_classification.PACKET_FEATURES` and `LABEL_COL` (see `data/network_packets/labeled_packet_sequences.md`).
- **Client alignment**: Using `prepare_fl_data_for_run_fl.py` with equalized packet counts keeps clients aligned in rounds (see Option 3 in that script).

**See also**: `fl-time-series/README.md` (training), `evaluate_fl_system.py` docstrings in this directory.
