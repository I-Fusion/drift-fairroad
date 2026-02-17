# Federated Learning – Time-Series (Classification & Regression)

This directory (**fl-time-series**) contains the **federated learning (FL) system** for the drift-fairroad project: **classification** (e.g. attack vs normal from labeled packet sequences) and **regression** (e.g. GPS/IMU time-series prediction). It is used for cyber anomaly detection (GPS spoofing, waypoint injection, jamming) with multiple clients training locally and a server aggregating model updates.

---

## Overview

- **Classification**: Labeled packet sequences (e.g. `SrcPort`, `DstPort`, `Length`, `MsgID`, `Protocol` → `Label` 0/1). Sliding windows over sequences; LSTM (or configured model) for binary/multi-class detection.
- **Regression**: GPS/IMU time-series; same FL server/client layout, different data and loss.
- **Server** (`fl_server_time_series.py`): Coordinates rounds, aggregates client weights (FedAvg, FedAvgM, or weighted), saves checkpoints.
- **Clients** (`fl_client_time_series.py`): Load per-client data, train on windows, submit updates. Support for packet-only or GPS/IMU data.
- **Run script** (`run_fl_system_time_series.py`): Starts server and clients from the project root; uses a single config module (e.g. `config_classification` or `config_regression`).

---

## Directory Contents

| File | Description |
|------|-------------|
| `run_fl_system_time_series.py` | Main entry: starts FL server and client processes. |
| `fl_server_time_series.py` | FL server: registration, model distribution, aggregation, checkpoints. |
| `fl_client_time_series.py` | FL client: data load, training, submit updates. |
| `config_classification.py` | Config for **packet-only classification** (features, labels, window, server). |
| `config_regression.py` | Config for **GPS/IMU regression**. |
| `data_preprocessing_time_series.py` | Builds sliding-window data from packet or GPS/IMU CSVs. |
| `aggregation.py` | Aggregation strategies (FedAvg, FedAvgM, weighted). |

---

## Configuration

- **Classification** (packet): `config_classification` — `PACKET_FEATURES`, `LABEL_COL`, `WINDOW_SIZE`, `OVERLAP`, `NUM_CLIENTS`, `MIN_CLIENTS`, etc. See `config_classification.py`.
- **Regression** (GPS/IMU): `config_regression` — GPS/IMU features, no labels, same window/FL params.
- **Config module** is selected at runtime with `--config`; server and clients share it (e.g. via `FL_CONFIG_MODULE`).

---

## Quick Start

**From the project root** (not from inside `fl-time-series`):

1. **Prepare data** (e.g. packet client splits):
   ```bash
   python raw_data_processing/prepare_fl_data_for_run_fl.py --packet-file data/network_packets/mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv --output data/train/packets
   ```

2. **Run FL classification**:
   ```bash
   python fl-time-series/run_fl_system_time_series.py --data-dir data/train/packets --config config_classification --learning-mode classification
   ```

3. **Run FL regression** (if you have GPS/IMU data):
   ```bash
   python fl-time-series/run_fl_system_time_series.py --data-dir data/train/gps-imu --config config_regression --learning-mode regression
   ```

Checkpoints are written to `fl-time-series/checkpoints/` (server) and `fl-time-series/checkpoints/clients/` (client checkpoints). Plots go to `fl-time-series/plots/`. Use `evaluate_fl_system.py` from project root; see `EVALUATE_FL_SYSTEM.md`.

---

## Key Options (`run_fl_system_time_series.py`)

- `--data-dir`: Directory with per-client data (e.g. `packet_client_000.csv`, or GPS/IMU client files).
- `--config`: Config module name (`config_classification`, `config_regression`, or custom).
- `--learning-mode`: `classification` or `regression` (can be auto-detected from config/data).
- `--num-clients`, `--min-clients`: Override config if needed.

---

## Data and Evaluation

- **Packet format**: See `data/network_packets/labeled_packet_sequences.md` for CSV format (Timestamp, SrcPort, DstPort, Length, MsgID, Protocol, Label).
- **Evaluation**: From project root, `python fl-time-series/evaluate_fl_system.py --data-dir data/train/packets --task classification --config config_classification` (checkpoint and output dirs default to `fl-time-series/checkpoints` and `fl-time-series/evaluation_results`). Covers detection metrics (AUROC, precision, recall, F1, false alarm rate), drift, mission continuity, and efficiency — see **`EVALUATE_FL_SYSTEM.md`** in this directory.
