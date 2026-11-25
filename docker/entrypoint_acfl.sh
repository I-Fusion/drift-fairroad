#!/bin/sh
# ACFL Docker entrypoint — mirrors docker/Dockerfile role switching.
# Env vars are documented in docker/DOCKER.md (ACFL section).
set -e

ROLE="${FL_ROLE:-client}"
HOST="${FL_SERVER_HOST:-0.0.0.0}"
PORT="${FL_SERVER_PORT:-8090}"
NUM_CLIENTS="${FL_NUM_CLIENTS:-3}"
MIN_CLIENTS="${FL_MIN_CLIENTS:-2}"
CONFIG="${FL_CONFIG_MODULE:-config_classification_drone}"
LEARNING_MODE="${FL_LEARNING_MODE:-classification}"
SERVER_URL="${FL_SERVER_URL:-http://acfl-server:8090}"
CLIENT_ID="${FL_CLIENT_ID:-client_1}"
DATA_DIR="${FL_DATA_DIR:-/app/data/train/packets}"
PACKET_FILE="${FL_PACKET_FILE:-}"
MAX_ROUNDS="${FL_MAX_ROUNDS:-}"
FL_SEED="${FL_SEED:-}"

export FL_CONFIG_MODULE="$CONFIG"
if [ -n "$MAX_ROUNDS" ]; then
  export FL_MAX_ROUNDS="$MAX_ROUNDS"
fi
if [ -n "$FL_SEED" ]; then
  export FL_SEED
fi

if [ "$ROLE" = "server" ]; then
  set -- python -u acfl/acfl_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --num-clients "$NUM_CLIENTS" \
    --min-clients "$MIN_CLIENTS" \
    --learning-mode "$LEARNING_MODE"

  [ "${ACFL_NO_LAYER1:-0}" = "1" ] && set -- "$@" --no-layer1
  [ "${ACFL_NO_LAYER2:-0}" = "1" ] && set -- "$@" --no-layer2
  [ "${ACFL_NO_LAYER3:-0}" = "1" ] && set -- "$@" --no-layer3
  [ -n "${ACFL_MALICIOUS_CLIENTS:-}" ] && set -- "$@" --malicious-clients "$ACFL_MALICIOUS_CLIENTS"
  [ -n "${ACFL_PROBE_FILE:-}" ] && set -- "$@" --probe-file "$ACFL_PROBE_FILE"
  [ -n "${ACFL_EVALUATION_ATTACK_TYPE:-}" ] && set -- "$@" --evaluation-attack-type "$ACFL_EVALUATION_ATTACK_TYPE"

  exec "$@"
fi

# Client role
if [ -n "${FL_CLIENT_STAGGER_SEC:-}" ] && [ "$FL_CLIENT_STAGGER_SEC" -gt 0 ] 2>/dev/null; then
  # Stagger by client index when FL_CLIENT_ID=client_N
  idx=$(echo "$CLIENT_ID" | sed -n 's/.*_\([0-9][0-9]*\)$/\1/p')
  idx=${idx:-1}
  sleep $(( (idx - 1) * FL_CLIENT_STAGGER_SEC ))
fi

set -- python -u fl_client_time_series.py \
  --client-id "$CLIENT_ID" \
  --server-url "$SERVER_URL" \
  --data-dir "$DATA_DIR"

if [ -n "$PACKET_FILE" ]; then
  set -- "$@" --packet-file "$PACKET_FILE"
fi
if [ "$LEARNING_MODE" = "classification" ] || [ "${FL_USE_LABELS:-1}" = "1" ]; then
  set -- "$@" --use-labels
fi

# Optional poisoning on malicious clients (same env names as run_acfl_system.py)
if [ -n "${ACFL_ATTACK_CLIENTS:-}" ] && [ -n "${ACFL_ATTACK_TYPE:-}" ]; then
  export ACFL_ATTACK_CLIENTS
  export ACFL_ATTACK_TYPE
  export ACFL_ATTACK_SCALE="${ACFL_ATTACK_SCALE:-10.0}"
  export ACFL_ATTACK_NOISE="${ACFL_ATTACK_NOISE:-1.0}"
  [ -n "${ACFL_BACKDOOR_POISON_FRACTION:-}" ] && export ACFL_BACKDOOR_POISON_FRACTION
  [ -n "${ACFL_LABEL_FLIP_FRACTION:-}" ] && export ACFL_LABEL_FLIP_FRACTION
fi

exec "$@"
