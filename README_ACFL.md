# ACFL — Adversarially-Conditioned Federated Learning

Poisoning-resilient FL defense for the
[I-Fusion/drift-fairroad](https://github.com/I-Fusion/drift-fairroad)
evaluation testbed (IEEE AESS Challenge 3), plus the harness needed to
**reproduce the manuscript main results**.

This tree is meant to live at the **root of `drift-fairroad`** (alongside
`fl-time-series/`, `data/`, and `docker/`).

## What this adds

| Path | Role |
|------|------|
| `acfl/` | Defense package (Layers 1–3) + robust baselines |
| `run_acfl_system.py` | End-to-end FL runner (`acfl.acfl_server`) |
| `benchmarks/` | Journal / ablation / Layer‑3 harnesses (`run_drone_*.py`, `run_comprehensive_validation.py`) |
| `evaluate_acfl.py` / `evaluate_layer3_sitl.py` | Trust-score and Layer‑3 AUROC helpers |
| `fl-time-series/models/tiny_packet_cnn.py` | Drone-scale TinyPacketCNN (~14k params) |
| `fl-time-series/config_classification_drone.py` | Packet FL profile that selects TinyPacketCNN |
| `fl-time-series/checkpoints/acfl_examples/` | Example ACFL / FedAvg `.pt` checkpoints |
| `collect_sitl_dataset.sh` | Primary 10-mission SITL collection (paper recipe) |
| `collect_validate_missions.sh` / `collect_drone_missions.sh` | +4 / +6 supplemental missions |
| `build_agent_mission_dataset.py` | Agent–mission FL packet layout |
| `scripts/run_acfl_mp_sitl_demo.py` | Fly FedAvg vs ACFL 3-UAV Mission Planner scenes |
| `scripts/merge_mp_multivehicle_tlog.py` | Merge multi-vehicle tlogs (+ EKF/SIM dual sysids, NVF metrics) |
| `scripts/pack_acfl_mp_demo.py` | Pack MP-ready tlogs / captions under `results/demo_mission_planner/` |
| `scripts/extract_acfl_mp_tracks.py` | Extract EKF vs SIM tracks from packed runs |
| `scripts/sync_manuscript_numbers.py` | Regenerate LaTeX tables from JSON summaries |
| `docs/manuscript/` | Journal manuscript sources |
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

## Install

```bash
git clone -b v2 https://github.com/I-Fusion/drift-fairroad.git
cd drift-fairroad
# If using this branch / PR, check out acfl-deliverable instead of copying.

python -m venv fl_env
source fl_env/bin/activate
pip install -r requirements.txt
pip install -r requirements-acfl.txt
```

## Quick start (stock example packets)

```bash
python run_acfl_system.py \
    --data-dir data/train/packets \
    --config config_classification_drone \
    --learning-mode classification \
    --num-clients 3 --min-clients 2 --max-fl-rounds 12
```

## Reproduce manuscript main results

Raw SITL flight data is **not** shipped. Generate it locally, then run the
journal benchmark. Wall-clock for full SITL collection is hours; the FL
matrix (3 attacks × 5 defenders × 10 seeds × 12 rounds) is also long — start
with a smoke subset if needed.

### 1. One-time ArduPilot SITL setup

```bash
bash simulations/setup_sitl.sh
export ARDUPILOT_DIR=$HOME/ardupilot
```

See `simulations/SIMULATION_HARNESS.md` for ports, labeling, and debugging.

### 2. Collect the 20-mission paper dataset (scripts only — no raw data in git)

Matches the recorded primary recipe (seed **11**, **10** drones, winds
`{0,6,10}`, attacks `waypoint` / `gps` / `none`, 15× speedup, 300 m radius),
then adds validate (+4) and drone-extra (+6) missions:

```bash
bash collect_sitl_dataset.sh          # → data/sitl_dataset/
bash collect_validate_missions.sh     # → data/sitl_validate/
bash collect_drone_missions.sh        # → data/sitl_drone_extra/

python3 build_agent_mission_dataset.py \
  --train-runs data/sitl_dataset/runs \
  --extra-runs data/sitl_validate/runs \
  --extra-runs data/sitl_drone_extra/runs \
  --output data/sitl_dataset/fl/packets \
  --num-agents 10 --clean
```

Expected layout: `data/sitl_dataset/fl/packets/agent_XXX/mission_*.csv`
(~98k labeled packets across 10 agents × 2 missions).

Optional dataset stats:

```bash
python3 scripts/compute_dataset_statistics.py \
  --latex-out docs/manuscript/dataset_stats.tex
```

### 3. Layer‑1 probe set

The journal harness builds probes automatically when missing. To build them
yourself (pick any labeled mission CSV under the agent layout):

```bash
python3 generate_probe_set.py --mode classification \
  --packet-file data/sitl_dataset/fl/packets/agent_000/mission_drone_000.csv \
  --output fl-time-series/checkpoints/acfl_probes_sitl.pt
```

### 4. Main benchmark matrix

Defenders: FedAvg, coordinate median, trimmed mean, Krum, ACFL (full).  
Attacks: gradient scaling (10×), Byzantine, backdoor (35% poison / 5×).  
Seeds: 42–51. Config: `config_classification_drone` (TinyPacketCNN), 3 clients / quorum 2, 12 rounds.

```bash
# Full paper matrix (run from repo root)
python3 benchmarks/run_drone_journal_benchmark.py --seeds 42,43,44,45,46,47,48,49,50,51

# Smoke (3 seeds)
python3 benchmarks/run_drone_journal_benchmark.py --seeds 42,43,44
```

Outputs:

- `evaluation_results/acfl_tuning/drone_journal_benchmark.csv`
- `evaluation_results/acfl_tuning/drone_journal_benchmark.json`
- `evaluation_results/acfl_tuning/drone_journal_benchmark_summary.json`
- Per-scenario checkpoints under `evaluation_results/comprehensive_validation/journal_*`

### 5. Sync manuscript tables / figures (optional)

```bash
python3 scripts/sync_manuscript_numbers.py
python3 scripts/compute_significance_tests.py
python3 docs/manuscript/generate_figures.py

cd docs/manuscript
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Related manuscript suites

| Script | Paper section |
|--------|----------------|
| `benchmarks/run_drone_ablation.py` | Layer ablations |
| `benchmarks/run_drone_threat_gaps.py` | Label flip / model replacement |
| `benchmarks/run_drone_attack_intensity_sweep.py` | Attack intensity |
| `benchmarks/run_drone_probe_config_ablation.py` | Probe-config ablation |
| `benchmarks/run_drone_5client_*_benchmark.py` | 5-client swarm |
| `benchmarks/run_drone_combined_layer3_benchmark.py` / `benchmarks/run_drone_full_stack_benchmark.py` | Joint Layer‑1/2/3 |

## Mission Planner demos (FedAvg vs ACFL)

Fly separate controller logs per scene so Mission Planner can show **EKF
(sysid 1/2/3)** vs **SIM truth (sysid 11/12/13)**. Generated packs land under
`results/demo_mission_planner/` (gitignored — regenerate locally).

**Not included:** dual-pane / storyboard MP4 renderers, VO/soundtrack scripts,
or pre-baked demo packs.

### Prerequisites

```bash
bash simulations/setup_sitl.sh
export ARDUPILOT_DIR=$HOME/ardupilot
# Detectors: prefers journal checkpoints if you have run the FL matrix;
# otherwise falls back to fl-time-series/checkpoints/acfl_examples/*.pt
```

### Fly and pack

```bash
# All scenes × both controllers (fedavg, acfl)
python3 scripts/run_acfl_mp_sitl_demo.py --scene all

# One scene / one controller
python3 scripts/run_acfl_mp_sitl_demo.py --scene scene_a_gps_spoof --controller acfl

# Merge tlogs (dual sysids + NAMED_VALUE_FLOAT metrics) and pack for MP
python3 scripts/pack_acfl_mp_demo.py
```

### Recommended play order

For each scene, open the **fedavg** tlog first, then the **acfl** tlog (same
map setup) so miss→break vs recover-on-detect is obvious:

1. `scene_c_clean` — baseline; EKF ≈ SIM
2. `scene_a_gps_spoof` — multi-wave GPS
3. `scene_b_waypoint_collude` — waypoint + GPS
4. `scene_d_fleet_collapse` — all-3 spoof
5. `scene_e_reference_uav` — 2-of-3 spoof; clean craft stays locked

Packed files live at:

```text
results/demo_mission_planner/mission_planner/<scene>/{fedavg,acfl}/
  multivehicle_sync.tlog   # play this
  mission.waypoints        # load for map overlay
  README.md                # per-run notes
```

### Play telemetry logs in Mission Planner

Use a recent [Mission Planner](https://ardupilot.org/planner/) build (Windows
is the usual path; Mono/Linux builds work if tlog replay is available).

1. **Load waypoints (optional but recommended)**  
   Open **Flight Plan → Load WP** and select `mission.waypoints` from the same
   folder as the tlog. Mission Planner often does **not** draw embedded
   `MISSION_ITEM_INT` messages from the tlog automatically. Switch back to
   **Flight Data** so the route stays overlaid while icons move.

2. **Open the merged tlog**  
   **Flight Data → Ctrl+F → Telemetry Logs** (or **Data → Telemetry Logs**) →
   browse to `multivehicle_sync.tlog` → open / play.

3. **Show all vehicles**  
   Enable MAVLink system IDs **1, 2, 3** and **11, 12, 13** in the vehicle /
   map list (naming varies by MP version — look for “Vehicles” or the map
   legend). Without 11–13 you only see EKF and cannot judge spoof miss distance.

   | Sysid | Track |
   |------:|-------|
   | 1 / 2 / 3 | EKF `GLOBAL_POSITION_INT` (UAV 0 / 1 / 2) |
   | 11 / 12 / 13 | SIM truth projected as `GLOBAL_POSITION_INT` |

4. **Add live metrics to the HUD**  
   Merged tlogs emit `NAMED_VALUE_FLOAT` (~1 Hz) on each EKF sysid:

   | Name | Meaning |
   |------|---------|
   | `EKF_SIM_M` | EKF vs SIM horizontal error (m) |
   | `MISS_CNT` | Cumulative miss events |
   | `HOLD_CNT` | Cumulative detect / hold events |
   | `DET_CNT` | Cumulative recoveries |
   | `FAVG_SCR` / `ACFL_SCR` | Latest anomaly scores |
   | `ATK_ON` | Attack active (0 / 1) |

   In Mission Planner: **Status** tab → **Quick** (double-click a name to pin),
   HUD **right-click → User Items**, and/or the **Tuning** graph under the map.
   Pin at least `EKF_SIM_M`, `MISS_CNT`, and `HOLD_CNT`.

5. **Play and compare**  
   - **Clean:** icons 1↔11, 2↔12, 3↔13 stay stacked; `EKF_SIM_M` stays small.  
   - **FedAvg miss:** EKF icon slides away from its SIM twin; `EKF_SIM_M`
     climbs; watch for miss / break outcomes in the packed README.  
   - **ACFL hold:** spoof stops / mission restores; EKF re-converges toward SIM;
     `HOLD_CNT` / `DET_CNT` rise while max `EKF_SIM_M` stays bounded.

6. **Playback tips**  
   - Use MP’s play / pause / speed controls; start near attack onset (~15 s in
     most attack scenes).  
   - Keep map zoom wide enough to see all three UAVs and their SIM twins.  
   - Prefer one controller log at a time; open the matching `fedavg` and `acfl`
     folders side-by-side on disk so you can swap quickly.  
   - If only one vehicle appears, re-check that multi-vehicle display is on and
     sysids 11–13 are not filtered out.

## Tests

```bash
pytest tests/ -v
```

## Outputs during a single FL run

Under `fl-time-series/checkpoints/`:

- `acfl_trust_scores.json` — per-round client trust
- `acfl_round_reports.json` — validation / aggregation diagnostics

## Scope

**Included:** ACFL package, patched FL client/server (agent–mission loading,
oversampling, attack hooks, robust aggregators), TinyPacketCNN, journal and
ablation harnesses, SITL **collection** scripts, Mission Planner **fly/pack**
scripts, manuscript sources, unit tests.

**Not included:** large generated SITL run trees (`data/sitl_*`),
`evaluation_results/` caches, `results/demo_mission_planner/` packs, dual-pane
film / soundtrack tooling, and local virtualenvs. Regenerate runnable artifacts
with the steps above.
