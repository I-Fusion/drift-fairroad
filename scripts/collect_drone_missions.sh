#!/usr/bin/env bash
# Collect additional SITL missions sized for drone-scale storage/compute budgets.
#
# Defaults: 6 short missions, concurrency 2, ~15–25 min wall time.
# Each mission ≈ 3–7 MB processed CSVs; 6 missions ≈ 30–45 MB added to dataset.
#
# After collection, missions are merged into agents (not separate eval agents):
#   python3 scripts/build_agent_mission_dataset.py \
#     --train-runs data/sitl_dataset/runs \
#     --extra-runs data/sitl_drone_extra/runs \
#     --output data/sitl_dataset/fl/packets --clean
#
# Usage (from drift-fairroad repo root):
#   bash scripts/collect_drone_missions.sh
#   NUM_DRONE_MISSIONS=8 bash scripts/collect_drone_missions.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${REPO_ROOT}/acfl_env/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/acfl_env/bin/python"
elif [[ -x "${REPO_ROOT}/fl_env/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/fl_env/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
COLLECT="${REPO_ROOT}/simulations/collect_dataset.py"
GEN="${REPO_ROOT}/simulations/scripts/generate_missions.py"

DATA_ROOT="${REPO_ROOT}/data/sitl_drone_extra"
MISSION_POOL="${DATA_ROOT}/mission_pool"
FL_OUT="${DATA_ROOT}/fl"
NUM_DRONES="${NUM_DRONE_MISSIONS:-6}"
START_INSTANCE="${DRONE_START_INSTANCE:-14}"
CONCURRENCY="${DRONE_CONCURRENCY:-2}"
NO_SIM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sim) NO_SIM="--no-sim"; shift ;;
    --num) NUM_DRONES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${DATA_ROOT}" "${MISSION_POOL}"
LOG="${DATA_ROOT}/collection.log"

if [[ -z "$(find "${MISSION_POOL}" -maxdepth 1 -name '*.waypoints' -print -quit 2>/dev/null)" ]]; then
  echo ">>> Generating ${NUM_DRONES} compact missions (seed=77)..."
  "${PYTHON}" "${GEN}" --count "${NUM_DRONES}" --seed 77 \
    --radius 250 --min-wp 8 --max-wp 14 \
    --output "${MISSION_POOL}"
fi

echo ">>> Collecting ${NUM_DRONES} drone-scale missions (instance ${START_INSTANCE}+, concurrency ${CONCURRENCY})..."
"${PYTHON}" "${COLLECT}" \
  --num-drones "${NUM_DRONES}" \
  --start-instance "${START_INSTANCE}" \
  --fl-index-start 0 \
  --concurrency "${CONCURRENCY}" \
  --attacks waypoint gps none waypoint gps none \
  --winds 0 6 10 0 6 10 \
  --missions-dir "${MISSION_POOL}" \
  --speedup 15 \
  --attack-count 4 \
  --gps-duration 20 \
  --mission-timeout 150 \
  --seed 77 \
  --output "${DATA_ROOT}" \
  --fl-output "${FL_OUT}" \
  --manifest-name drone_extra_manifest.json \
  ${NO_SIM} 2>&1 | tee -a "${LOG}"

echo ""
echo "Drone extra dataset ready:"
echo "  runs: ${DATA_ROOT}/runs/"
echo "  Merge into agents:"
echo "    python3 scripts/build_agent_mission_dataset.py \\"
echo "      --train-runs data/sitl_dataset/runs \\"
echo "      --extra-runs data/sitl_validate/runs \\"
echo "      --extra-runs ${DATA_ROOT}/runs \\"
echo "      --output data/sitl_dataset/fl/packets --clean"
