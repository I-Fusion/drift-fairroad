# ArduPilot SITL Data Collection Harness

Reproduces the drift-fairroad data-collection method (see
`AESS_Challenge_Simulation.pdf`) as an automated, multi-drone pipeline:
generate missions -> fly them in ArduCopter SITL -> inject attacks ->
capture network packets + dataflash logs -> label by attack time-periods ->
assemble a federated-learning dataset. It scales to an arbitrary number of
drones (no longer limited to 3 agents).

## Components

| File | Role |
|------|------|
| `setup_sitl.sh` | One-time: clone + build ArduPilot SITL, install tshark, install pymavlink/MAVProxy into `acfl_env`. |
| `scripts/generate_missions.py` | Generate varied `.waypoints` missions (polygon/grid/star/circle/random) around home. |
| `scripts/attacks_sitl.py` | Parametrized waypoint-injection and GPS-spoofing attacks against an instance's attacker port. |
| `scripts/run_simulation.py` | Run one SITL flight: launch, mission, arm/takeoff/auto/RTL, schedule attack, capture pcap, locate `.BIN`. |
| `scripts/convert_logs.py` | Convert dataflash `.BIN` to GPS/IMU CSVs in the drift-fairroad schema. |
| `scripts/packet_pipeline.py` | pcap -> packet rows -> attack periods -> labeled packets + labeled GPS/IMU. |
| `collect_dataset.py` | Orchestrate many drones in batches; post-process; assemble an N-client FL dataset. |

## Pipeline

```
generate_missions -> collect_dataset --(per drone, batched)--> run_simulation
   run_simulation -> capture.pcapng + sitl/logs/*.BIN
   convert_logs   -> gps.csv, imu.csv
   packet_pipeline-> packets_labeled.csv, gps_labeled.csv, imu_labeled.csv
   collect_dataset-> fl/{packets,gps-imu}/*_client_XXX.csv + manifest.json
```

## Instance / port convention

Drone `I` uses:
- control / serial0: `5760 + 10*I` (pymavlink GCS connection + normal telemetry)
- attacker / serial1: `5762 + 10*I` (attack traffic; its packets mark attack periods)

Both ports are captured. Packets on the attacker port define the attack
time-windows; the (non-attacker) telemetry packets are then labeled
attack/normal by whether they fall inside a window -- exactly the
drift-fairroad labeling method.

## Setup (one time)

```bash
bash drift-fairroad/simulations/setup_sitl.sh
```

This needs `sudo` for apt (ArduPilot prerequisites and tshark) and downloads/
builds ~2-3 GB; budget 20-40 minutes. After it finishes:

```bash
export ARDUPILOT_DIR=$HOME/ardupilot
```

If you only need the GPS/IMU (regression) stream you can skip tshark, but packet
(classification) data requires it.

## Usage

Generate missions (optional; the orchestrator can do this automatically):

```bash
./acfl_env/bin/python drift-fairroad/simulations/scripts/generate_missions.py \
    --count 10 --output drift-fairroad/simulations/missions/generated
```

One flight:

```bash
./acfl_env/bin/python drift-fairroad/simulations/scripts/run_simulation.py \
    --instance 0 --attack waypoint --speedup 15 \
    --mission drift-fairroad/simulations/missions/mission_2_wp_23.waypoints \
    --run-dir runs/run0
```

Full multi-drone dataset (e.g. 6 drones, 2 at a time):

```bash
./acfl_env/bin/python drift-fairroad/simulations/collect_dataset.py \
    --num-drones 6 --concurrency 2 \
    --attacks waypoint gps none --winds 0 6 10 \
    --speedup 15 \
    --output runs/dataset1 --fl-output runs/dataset1/fl
```

Re-run only post-processing on existing flights:

```bash
./acfl_env/bin/python drift-fairroad/simulations/collect_dataset.py \
    --num-drones 6 --no-sim --output runs/dataset1 --fl-output runs/dataset1/fl
```

## Output layout

```
runs/dataset1/
  missions/                      generated mission files
  runs/drone_000/
    sitl/logs/*.BIN              dataflash log
    capture.pcapng               captured packets (control + attacker port)
    run_meta.json                ports, attack windows, paths
    processed/
      gps.csv, imu.csv           converted sensor streams
      packets_labeled.csv        Timestamp,SrcPort,DstPort,Length,MsgID,Protocol,Label
      gps_labeled.csv, imu_labeled.csv
  fl/
    packets/packet_client_000.csv ...    one client per drone
    gps-imu/{gps,imu}_client_000.csv ...
  manifest.json
```

The `fl/packets` directory is a drop-in `--data-dir` for the drift-fairroad
classification FL system and the ACFL defense; `fl/gps-imu` is for the
regression task. Each drone is one federated client, so the number of clients
equals `--num-drones`.

## Scenarios

`--attacks` and `--winds` are assigned round-robin across drones:
- `waypoint`: injects malicious waypoint(s) mid-mission via the attacker port (uses `config_waypoint_inject.param`).
- `gps`: streams spoofed `GPS_INPUT` for a window (uses `config_gps_injection.param`, `EK3_CHECK_SCALE=200`).
- `none`: clean flight (all-normal labels) for benign baseline data.
- `--winds` sets `SIM_WIND_SPD` (0 / 6 / 10 m/s as in the reference study).

## Resource notes

This machine has 4 cores / limited free RAM, so keep `--concurrency` low
(2-3) and raise `--speedup` (10-20x) to compress wall-clock time. Each drone
flies independently, so large datasets are produced by running many drones
across sequential batches.

## Caveats

- Loopback packet capture needs capture privileges; if tshark capture fails,
  re-login after `setup_sitl.sh` (group change) or run with `sudo`.
- The serial1 attacker-port offset is the standard ArduPilot SITL mapping; if a
  future ArduPilot changes it, adjust the offset in `run_simulation.py`.
