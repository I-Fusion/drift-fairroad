# GPS Spoofing Data

This directory contains **GPS spoofing / GNSS anomaly** datasets for the drift-fairroad federated learning (FL) system. The data is used to train and evaluate models that detect cyber anomalies (e.g. GPS spoofing, waypoint injection) in aerospace telemetry.

---

## Purpose

- **Training & evaluation**: Provide labeled or unlabeled GPS/IMU (and derived EKF) logs for FL classification or regression.
- **Anomaly detection**: Support evaluation metrics such as detection accuracy (AUROC, precision, recall, F1), false alarm rate, and time-to-detect in `evaluate_fl_system.py`.
- **Consistency with project**: Same style of mission/run naming as `data/waypoint_injection`; can be prepared with `raw_data_processing/prepare_fl_data_for_run_fl.py` for client splits.

---

## Primary EKF3 Message Names (XKF)

ArduPilot Extended Kalman Filter (EKF3) outputs used in the XKF logs:

| Message | Description |
|--------|-------------|
| **XKF1** | Attitude and velocity: Roll, Pitch, Yaw, VN, VE, VD. |
| **XKF2** | Position and altitude, IMU weighting: Lat, Lng, Alt. |
| **XKF3** | Sensor innovations for GPS, barometer, magnetometer, and airspeed. |
| **XKF4** | Innovation variances and health ratios: SV, SP, SH, SMX, SMY, SMZ. |

These messages are useful for detecting spoofing (e.g. innovation/consistency checks) and for feature engineering in FL models.

---

## Data Files in This Directory

| File | Description |
|------|-------------|
| `mission_2_wp_23_gps_input_increment_wind_10_gps.csv` | Raw GPS log (timestamp, Lat, Lng, Alt, fix type, etc.). |
| `mission_2_wp_23_gps_input_increment_wind_10_imu.csv` | IMU log (gyro, accelerometer, etc.). |
| `mission_2_wp_23_gps_input_increment_wind_10_pos.csv` | Position/fused position output. |
| `mission_2_wp_23_gps_input_increment_wind_10_xkf1.csv` | EKF3 attitude/velocity (XKF1). |
| `mission_2_wp_23_gps_input_increment_wind_10_xkf2.csv` | EKF3 position/altitude (XKF2). |
| `mission_2_wp_23_gps_input_increment_wind_10_xkf3.csv` | EKF3 sensor innovations (XKF3). |
| `mission_2_wp_23_gps_input_increment_wind_10_xkf4.csv` | EKF3 innovation variances and health (XKF4). |

Naming: `mission_2_wp_23` = mission/waypoint set; `gps_input_increment_wind_10` = scenario (e.g. GPS input increment with wind level 10).

---

## Using This Data in the Pipeline

1. **Prepare client splits** (if using packet or GPS/IMU FL):  
   From project root, run e.g.  
   `python raw_data_processing/prepare_fl_data_for_run_fl.py --gps-file data/gps_spoofing/mission_2_wp_23_gps_input_increment_wind_10_gps.csv --imu-file data/gps_spoofing/mission_2_wp_23_gps_input_increment_wind_10_imu.csv --output data/train/gps-spoofing`

2. **Evaluation**: Point `evaluate_fl_system.py` at the prepared data dir and checkpoint dir to compute detection metrics (AUROC, false alarm rate, etc.) and drift/stability as in `EVALUATE_FL_SYSTEM.md`.

3. **Timestamp column**: If CSVs use a different time column (e.g. first column or `Timestamp`), set `TIMESTAMP_COL` in your config to match.
