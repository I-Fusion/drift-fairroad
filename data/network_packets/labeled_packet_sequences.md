# Labeled Packet Sequences

This directory contains **labeled network packet sequence** CSVs used for federated learning (FL) **classification** in the drift-fairroad system. Each row is one packet (or packet-level event); sequences are turned into sliding windows for training and evaluating anomaly detection (e.g. attack vs normal).

---

## Purpose

- **FL classification**: Train and evaluate binary (or multi-class) models that separate **normal** from **attack** traffic (e.g. GPS spoofing, waypoint injection, jamming-related traffic).
- **Pipeline**: Data is prepared with `raw_data_processing/prepare_fl_data_for_run_fl.py` (packet-only mode), then used by `fl-classification/run_fl_system_classification.py` and `evaluate_fl_system.py` with configs such as `config_classification` / `config_packets_only`.

---

## CSV Format

Each file must have a **header row** and the following columns (names must match config):

| Column      | Description                    | Used as        |
|------------|--------------------------------|----------------|
| **Timestamp** | Packet time (e.g. `2026-01-12 14:24:21.940780`) | Time order; config: `PACKET_TIMESTAMP_COL` |
| **SrcPort**   | Source port                    | Feature        |
| **DstPort**   | Destination port               | Feature        |
| **Length**    | Packet length (numeric)        | Feature        |
| **MsgID**     | Message ID                     | Feature        |
| **Protocol**  | Protocol identifier (numeric)  | Feature        |
| **Label**     | 0 = normal, 1 = attack (or class id) | Target; config: `LABEL_COL` |

Config in `fl-classification/config_classification.py`:

- **`PACKET_FEATURES`**: `['SrcPort', 'DstPort', 'Length', 'MsgID', 'Protocol']`
- **`PACKET_TIMESTAMP_COL`**: `'Timestamp'`
- **`LABEL_COL`**: `'Label'`

---

## Example Row

```text
Timestamp,SrcPort,DstPort,Length,MsgID,Protocol,Label
2026-01-12 14:24:21.940780,52931,14550,48.0,163,0,0
```

- Normal row: `Label = 0`.
- Attack row: `Label = 1` (or other class if you extend to multi-class).

Missing numeric values (e.g. empty `Length`) are handled by the preprocessor; keep timestamps and labels consistent for correct windowing.

---

## Data Files in This Directory

| File | Description |
|------|-------------|
| `mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv` | Labeled packets for mission 2, waypoint 23, attack “add wp 5 alt 0005”. |
| `mission_2_wp_23_attack_add_wp_7_alt_0001_labeled.csv` | Labeled packets for mission 2, waypoint 23, attack “add wp 7 alt 0001”. |

Naming pattern: `mission_*_attack_*_labeled.csv` — one CSV per scenario; multiple CSVs can be split across clients by the preparation script.

---

## Using This Data

1. **Prepare client splits** (from project root):
   ```bash
   python raw_data_processing/prepare_fl_data_for_run_fl.py --packet-file data/network_packets/mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv --output data/train/packets
   ```
   Use `--packet-file` for each file or point to a directory if the script supports it; output is written to `data/train/packets` (or your chosen path) as `packet_client_000.csv`, etc.

2. **Train FL classification**:
   ```bash
   python fl-classification/run_fl_system_classification.py --data-dir data/train/packets --config config_classification --learning-mode classification
   ```

3. **Evaluate** (detection accuracy, AUROC, false alarm rate, etc.):
   ```bash
   python evaluate_fl_system.py --checkpoint-dir checkpoints --data-dir data/train/packets --task classification --config config_classification
   ```

Ensure `PACKET_FEATURES`, `PACKET_TIMESTAMP_COL`, and `LABEL_COL` in your config match the column names in these CSVs.
