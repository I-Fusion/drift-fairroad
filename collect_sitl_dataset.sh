#!/usr/bin/env bash
# Collect the primary 10-mission SITL packet dataset used by the ACFL manuscript.
#
# Matches the recorded collection recipe (seed=11, 10 drones, wind {0,6,10},
# attacks waypoint/gps/none, 15x speedup, 300 m radius). Raw flight artifacts
# land under data/sitl_dataset/; they are NOT shipped in this repository.
#
# Prerequisites:
#   bash simulations/setup_sitl.sh
#   export ARDUPILOT_DIR=$HOME/ardupilot
#
# Usage (from drift-fairroad repo root):
#   bash collect_sitl_dataset.sh
#   bash collect_sitl_dataset.sh --no-sim   # re-postprocess existing runs only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "${REPO_ROOT}/acfl_env/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/acfl_env/bin/python"
elif [[ -x "${REPO_ROOT}/fl_env/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/fl_env/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
COLLECT="${REPO_ROOT}/simulations/collect_dataset.py"
GEN="${REPO_ROOT}/simulations/scripts/generate_missions.py"

DATA_ROOT="${REPO_ROOT}/data/sitl_dataset"
MISSION_POOL="${DATA_ROOT}/mission_pool"
FL_OUT="${DATA_ROOT}/fl"
NUM_DRONES="${NUM_SITL_DRONES:-10}"
START_INSTANCE="${SITL_START_INSTANCE:-0}"
CONCURRENCY="${SITL_CONCURRENCY:-2}"
SEED="${SITL_SEED:-11}"
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
  echo ">>> Generating primary mission pool (seed=${SEED}, count=${NUM_DRONES})..."
  "${PYTHON}" "${GEN}" --count "${NUM_DRONES}" --seed "${SEED}" \
    --radius 300 --output "${MISSION_POOL}"
fi

echo ">>> Collecting ${NUM_DRONES} primary SITL drones (instance ${START_INSTANCE}+)..."
# shellcheck disable=SC2086
"${PYTHON}" "${COLLECT}" \
  --num-drones "${NUM_DRONES}" \
  --start-instance "${START_INSTANCE}" \
  --fl-index-start 0 \
  --concurrency "${CONCURRENCY}" \
  --attacks waypoint gps none \
  --winds 0 6 10 \
  --missions-dir "${MISSION_POOL}" \
  --mission-pool "${NUM_DRONES}" \
  --area-radius 300 \
  --speedup 15 \
  --attack-count 5 \
  --gps-duration 25 \
  --seed "${SEED}" \
  --output "${DATA_ROOT}" \
  --fl-output "${FL_OUT}" \
  ${NO_SIM} \
  2>&1 | tee -a "${LOG}"

echo
echo ">>> Primary collection done. Next steps:"
echo "  1) bash collect_validate_missions.sh    # +4 missions"
echo "  2) bash collect_drone_missions.sh       # +6 missions"
echo "  3) python3 build_agent_mission_dataset.py \\"
echo "       --train-runs data/sitl_dataset/runs \\"
echo "       --extra-runs data/sitl_validate/runs \\"
echo "       --extra-runs data/sitl_drone_extra/runs \\"
echo "       --output data/sitl_dataset/fl/packets --num-agents 10 --clean"
