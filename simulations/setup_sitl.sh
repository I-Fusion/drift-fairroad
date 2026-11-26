#!/usr/bin/env bash
#
# setup_sitl.sh - one-time environment setup for the ArduPilot SITL data
# collection harness.
#
# This script:
#   1. Clones ArduPilot (with submodules) into $ARDUPILOT_DIR (default ~/ardupilot)
#   2. Installs ArduPilot's Ubuntu build prerequisites
#   3. Installs tshark/wireshark for packet capture (and enables non-root capture)
#   4. Builds the ArduCopter SITL binary (./waf copter)
#   5. Installs pymavlink + MAVProxy into the project venv (acfl_env)
#
# Several steps need sudo (apt). You may be prompted for your password.
#
# Usage:
#   bash drift-fairroad/simulations/setup_sitl.sh
#
# Environment overrides:
#   ARDUPILOT_DIR  - where to clone/build ArduPilot (default: $HOME/ardupilot)
#   VENV_DIR       - project virtualenv to install python deps into
#   SKIP_PREREQS=1 - skip the (slow) apt prerequisite install
#   SKIP_BUILD=1   - skip the waf build (e.g. if already built)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ACFL project root is two levels up: simulations/ -> drift-fairroad/ -> ACFL/
ACFL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
VENV_DIR="${VENV_DIR:-$ACFL_ROOT/acfl_env}"
ARDUPILOT_REPO="https://github.com/ArduPilot/ardupilot.git"

echo "================================================================"
echo "ArduPilot SITL harness setup"
echo "  ACFL_ROOT      = $ACFL_ROOT"
echo "  ARDUPILOT_DIR  = $ARDUPILOT_DIR"
echo "  VENV_DIR       = $VENV_DIR"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Clone ArduPilot
# ---------------------------------------------------------------------------
if [ -d "$ARDUPILOT_DIR/.git" ]; then
    echo "[1/5] ArduPilot already present at $ARDUPILOT_DIR; updating submodules..."
    git -C "$ARDUPILOT_DIR" submodule update --init --recursive
else
    echo "[1/5] Cloning ArduPilot into $ARDUPILOT_DIR (this is large, ~2 GB)..."
    git clone --recurse-submodules "$ARDUPILOT_REPO" "$ARDUPILOT_DIR"
fi

# ---------------------------------------------------------------------------
# 2. ArduPilot build prerequisites
# ---------------------------------------------------------------------------
if [ "${SKIP_PREREQS:-0}" = "1" ]; then
    echo "[2/5] SKIP_PREREQS=1 -> skipping apt prerequisite install."
else
    echo "[2/5] Installing ArduPilot prerequisites (may prompt for sudo)..."
    # The official installer is interactive-friendly and idempotent.
    # USER is required by the script; default to current user if unset.
    export USER="${USER:-$(id -un)}"
    bash "$ARDUPILOT_DIR/Tools/environment_install/install-prereqs-ubuntu.sh" -y
fi

# ---------------------------------------------------------------------------
# 3. tshark / wireshark (packet capture)
# ---------------------------------------------------------------------------
echo "[3/5] Installing tshark for packet capture..."
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
# Preseed so wireshark-common does not open an interactive dialog.
echo "wireshark-common wireshark-common/install-setuid boolean true" | sudo debconf-set-selections
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tshark wireshark-common

# Allow non-root capture: add current user to the wireshark group and make
# dumpcap capturing-capable. (Capturing on loopback otherwise needs root.)
if getent group wireshark >/dev/null 2>&1; then
    sudo usermod -aG wireshark "$(id -un)" || true
fi
DUMPCAP="$(command -v dumpcap || echo /usr/bin/dumpcap)"
if [ -x "$DUMPCAP" ]; then
    sudo setcap cap_net_raw,cap_net_admin=eip "$DUMPCAP" || true
fi
echo "    NOTE: group membership changes take effect on next login. If capture"
echo "    fails with a permissions error, run the harness with sudo or re-login."

# ---------------------------------------------------------------------------
# 4. Build ArduCopter SITL
# ---------------------------------------------------------------------------
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "[4/5] SKIP_BUILD=1 -> skipping waf build."
else
    echo "[4/5] Building ArduCopter SITL (./waf copter)..."
    (
        cd "$ARDUPILOT_DIR"
        # Ensure ardupilot's own tools are on PATH for this build (waf uses them).
        export PATH="$ARDUPILOT_DIR/Tools/autotest:$PATH"
        ./waf configure --board sitl
        ./waf copter
    )
fi

ARDUCOPTER_BIN="$ARDUPILOT_DIR/build/sitl/bin/arducopter"
if [ -x "$ARDUCOPTER_BIN" ]; then
    echo "    Built: $ARDUCOPTER_BIN"
else
    echo "    WARNING: expected binary not found at $ARDUCOPTER_BIN"
fi

# ---------------------------------------------------------------------------
# 5. Python deps into the project venv
# ---------------------------------------------------------------------------
echo "[5/5] Installing pymavlink + MAVProxy into $VENV_DIR ..."
if [ -x "$VENV_DIR/bin/pip" ]; then
    "$VENV_DIR/bin/pip" install --upgrade pymavlink MAVProxy
else
    echo "    WARNING: venv pip not found at $VENV_DIR/bin/pip; installing with python3 -m pip"
    python3 -m pip install --upgrade pymavlink MAVProxy
fi

echo "================================================================"
echo "Setup complete."
echo "  ArduCopter binary : $ARDUCOPTER_BIN"
echo "  sim_vehicle.py    : $ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py"
echo ""
echo "Export this so the harness can find ArduPilot:"
echo "  export ARDUPILOT_DIR=$ARDUPILOT_DIR"
echo "================================================================"
