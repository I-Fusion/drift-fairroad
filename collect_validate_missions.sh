#!/usr/bin/env bash
# Collect additional SITL missions; they are ingested into agents (not held-out agents).
# After collection, run: python3 build_agent_mission_dataset.py --extra-runs .../sitl_validate/runs
#
# Usage (from drift-fairroad repo root):
#   bash collect_validate_missions.sh
#   bash collect_validate_missions.sh --no-sim   # re-postprocess existing runs only
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

DATA_ROOT="${REPO_ROOT}/data/sitl_validate"
MISSION_POOL="${DATA_ROOT}/mission_pool"
FL_OUT="${DATA_ROOT}/fl"
NUM_DRONES="${NUM_VALIDATE_DRONES:-4}"
START_INSTANCE="${VALIDATE_START_INSTANCE:-10}"
CONCURRENCY="${VALIDATE_CONCURRENCY:-2}"
NO_SIM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sim) NO_SIM="--no-sim"; shift ;;
    --num-drones) NUM_DRONES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${DATA_ROOT}" "${MISSION_POOL}"
LOG="${DATA_ROOT}/collection.log"

if [[ -z "$(find "${MISSION_POOL}" -maxdepth 1 -name '*.waypoints' -print -quit 2>/dev/null)" ]]; then
  echo ">>> Generating validate mission pool (seed=42)..."
  "${PYTHON}" "${GEN}" --count "${NUM_DRONES}" --seed 42 \
    --radius 300 --output "${MISSION_POOL}"
fi

echo ">>> Collecting ${NUM_DRONES} validate drones (instance ${START_INSTANCE}+)..."
"${PYTHON}" "${COLLECT}" \
  --num-drones "${NUM_DRONES}" \
  --start-instance "${START_INSTANCE}" \
  --fl-index-start 0 \
  --concurrency "${CONCURRENCY}" \
  --attacks waypoint gps waypoint gps \
  --winds 0 6 0 6 \
  --missions-dir "${MISSION_POOL}" \
  --speedup 15 \
  --attack-count 5 \
  --gps-duration 25 \
  --mission-timeout 200 \
  --seed 42 \
  --output "${DATA_ROOT}" \
  --fl-output "${FL_OUT}" \
  --manifest-name validate_manifest.json \
  ${NO_SIM} 2>&1 | tee -a "${LOG}"

echo ""
echo "Validate dataset ready:"
echo "  runs:  ${DATA_ROOT}/runs/drone_${START_INSTANCE}.."
echo "  Ingest into agents: python3 build_agent_mission_dataset.py --extra-runs ${DATA_ROOT}/runs"
echo "  manifest: ${DATA_ROOT}/validate_manifest.json"
