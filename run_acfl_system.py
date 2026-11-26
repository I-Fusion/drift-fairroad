#!/usr/bin/env python3
"""
Run ACFL-enabled Federated Learning with drift-fairroad testbed.

Usage (from the drift-fairroad repository root):
    python3 run_acfl_system.py --data-dir data/train/packets \\
        --config config_classification --learning-mode classification

    python3 run_acfl_system.py --data-dir data/train/gps-imu \\
        --config config_regression --learning-mode regression
"""
import argparse
import glob
import importlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

import requests

# Deliverable layout: this tree is intended to live at the root of
# I-Fusion/drift-fairroad (sibling to fl-time-series/ and data/).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ACFL_ROOT = REPO_ROOT
DRIFT_ROOT = REPO_ROOT
FL_TS_DIR = os.path.join(REPO_ROOT, "fl-time-series")

for path in [REPO_ROOT, FL_TS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

processes: List[subprocess.Popen] = []


def cleanup_processes():
    logger.info("\nCleaning up processes...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    logger.info("All processes terminated.")


def signal_handler(sig, frame):
    logger.info("\nInterrupted by user")
    cleanup_processes()
    sys.exit(0)


def wait_for_server(host: str, port: int, max_retries: int = 30) -> bool:
    url = f"http://{host}:{port}/status"
    for i in range(max_retries):
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                status = resp.json()
                logger.info("Server ready at %s", url)
                if "acfl" in status:
                    logger.info(
                        "  ACFL layers: L1=%s L2=%s L3=%s",
                        status["acfl"]["layer1"],
                        status["acfl"]["layer2"],
                        status["acfl"]["layer3"],
                    )
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
        if i % 5 == 0:
            logger.info("Waiting for server... (%d/%d)", i, max_retries)
    return False


def wait_for_client_registered(
    host: str, port: int, expected_count: int, max_wait: int = 180
) -> bool:
    """Poll until at least expected_count clients have registered."""
    url = f"http://{host}:{port}/status"
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                registered = resp.json().get("registered_clients", 0)
                if registered >= expected_count:
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    return False


def count_clients_in_data_dir(data_dir: str) -> int:
    """Count FL agents (not missions) in a packet data directory."""
    pkt_dir = data_dir
    if os.path.basename(data_dir) != "packets":
        candidate = os.path.join(data_dir, "packets")
        if os.path.isdir(candidate):
            pkt_dir = candidate
    try:
        sys.path.insert(0, FL_TS_DIR)
        from agent_mission_data import count_agents
        n = count_agents(pkt_dir)
        if n > 0:
            return n
    except Exception:
        pass
    patterns = [
        "packet_client_*.csv",
        "gps_client_*.csv",
        "imu_client_*.csv",
    ]
    counts = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        if files:
            counts.append(len(files))
    return max(counts) if counts else 0


def get_client_files(
    client_id: str,
    data_dir: Optional[str],
    gps_file: Optional[str],
    imu_file: Optional[str],
    packet_file: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve per-client data files (matches drift-fairroad run script)."""
    client_num = None
    try:
        parts = client_id.split("_")
        if len(parts) > 1:
            client_num = int(parts[-1]) - 1
    except (ValueError, IndexError):
        pass

    client_gps = gps_file
    client_imu = imu_file
    client_packet = packet_file

    if data_dir and client_num is not None:
        gps_path = os.path.join(data_dir, f"gps_client_{client_num:03d}.csv")
        imu_path = os.path.join(data_dir, f"imu_client_{client_num:03d}.csv")
        packet_path = os.path.join(data_dir, f"packet_client_{client_num:03d}.csv")

        if os.path.exists(gps_path):
            client_gps = gps_path
        if os.path.exists(imu_path):
            client_imu = imu_path
        if os.path.exists(packet_path):
            client_packet = packet_path

    return client_gps, client_imu, client_packet


def monitor_training(host: str, port: int, target_rounds: Optional[int] = None):
    """Poll server status during training."""
    url = f"http://{host}:{port}/status"
    last_round = -1
    while True:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                time.sleep(2)
                continue
            status = resp.json()
            current_round = status.get("current_round", 0)
            if current_round != last_round:
                trust = status.get("acfl", {}).get("latest_trust_scores", {})
                logger.info(
                    "Round %d complete | clients ready: %d/%d | trust: %s",
                    current_round,
                    status.get("ready_clients", 0),
                    status.get("registered_clients", 0),
                    {k: f"{v:.3f}" for k, v in trust.items()} if trust else "n/a",
                )
                last_round = current_round

            if target_rounds and current_round >= target_rounds:
                logger.info("Reached target round %d", target_rounds)
                break

            if status.get("registered_clients", 0) > 0:
                client_procs_running = any(
                    p.poll() is None for p in processes[1:]
                )
                if not client_procs_running and current_round > 0:
                    break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="Run ACFL Federated Learning System")
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument(
        "--config",
        default="config_classification",
        choices=["config_classification", "config_classification_drone", "config_regression"],
    )
    parser.add_argument(
        "--learning-mode",
        default="classification",
        choices=["classification", "regression"],
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Stop monitoring after N server rounds",
    )
    parser.add_argument(
        "--max-fl-rounds",
        type=int,
        default=None,
        help="Hard cap on client FL rounds (sets FL_MAX_ROUNDS env)",
    )
    parser.add_argument(
        "--fedavg",
        action="store_true",
        help="Use vanilla FedAvg server (no ACFL defense) as baseline",
    )
    parser.add_argument(
        "--baseline-aggregation",
        default="",
        choices=["", "fedavg", "coordinate_median", "trimmed_mean", "krum"],
        help="Use vanilla FL server with a robust aggregation baseline (no ACFL)",
    )
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5544)
    parser.add_argument("--no-layer1", action="store_true")
    parser.add_argument("--no-layer2", action="store_true")
    parser.add_argument("--no-layer3", action="store_true")
    parser.add_argument(
        "--malicious-clients",
        default="",
        help="Comma-separated client IDs that are malicious (sets ground truth "
        "and, if --attack-type is set, makes them actually poison updates)",
    )
    parser.add_argument(
        "--attack-type",
        default="none",
        choices=[
            "none",
            "gradient_scaling",
            "byzantine",
            "backdoor",
            "label_flip",
            "model_replacement",
        ],
        help="Poisoning attack applied by malicious clients",
    )
    parser.add_argument("--attack-scale", type=float, default=10.0)
    parser.add_argument("--attack-noise", type=float, default=1.0)
    parser.add_argument(
        "--backdoor-poison-fraction",
        type=float,
        default=0.35,
        help="Fraction of benign windows poisoned on backdoor clients",
    )
    parser.add_argument(
        "--label-flip-fraction",
        type=float,
        default=0.5,
        help="Fraction of local training labels flipped on label_flip clients",
    )
    parser.add_argument(
        "--fl-seed",
        type=int,
        default=None,
        help="Fixed RNG seed for reproducible FL training (server + clients)",
    )
    parser.add_argument(
        "--probe-file",
        default=None,
        help="Probe set (.pt) from generate_probe_set.py for Layer 1 "
        "counterfactual validation",
    )
    parser.add_argument(
        "--acfl-trust-weight-power",
        type=float,
        default=None,
        help="Override ACFL trust exponent for aggregation weighting",
    )
    parser.add_argument(
        "--acfl-min-trust-weight",
        type=float,
        default=None,
        help="Override ACFL minimum aggregation trust weight",
    )
    parser.add_argument(
        "--acfl-subspace-weight",
        type=float,
        default=None,
        help="Override Layer 1 subspace score weight (renormalized with others)",
    )
    parser.add_argument(
        "--acfl-temporal-weight",
        type=float,
        default=None,
        help="Override Layer 1 temporal consistency weight",
    )
    parser.add_argument(
        "--acfl-counterfactual-weight",
        type=float,
        default=None,
        help="Override Layer 1 counterfactual probe weight (0 skips probe eval)",
    )
    parser.add_argument(
        "--acfl-coordinate-deviation-threshold",
        type=float,
        default=None,
        help="Override Layer 2 coordinate deviation threshold",
    )
    parser.add_argument(
        "--acfl-norm-ratio-threshold",
        type=float,
        default=None,
        help="Override Layer 1/2 norm-ratio threshold",
    )
    parser.add_argument(
        "--acfl-norm-ratio-threshold-two-clients",
        type=float,
        default=None,
        help="Override norm-ratio threshold for two-client rounds",
    )
    parser.add_argument(
        "--acfl-backdoor-probe-soft-cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use backdoor_trigger probe harm for aggregation soft cap (backdoor runs)",
    )
    parser.add_argument(
        "--acfl-backdoor-trigger-hard-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hard-clamp aggregation trust on backdoor_trigger harm / low P(attack)",
    )
    parser.add_argument(
        "--acfl-backdoor-probe-hard-reject",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hard-reject backdoor trigger-harm outliers from aggregation",
    )
    parser.add_argument(
        "--acfl-layer1-layer2-gating",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Separate telemetry L1 from aggregation trust (neutral/min-blend)",
    )
    parser.add_argument(
        "--acfl-adaptive-l1l2-gating",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Blend raw L1 and gated trust with context gate g_t",
    )
    parser.add_argument(
        "--acfl-parallel-trust-arbitration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Dual-path gated/ungated aggregation with probe-set arbitration",
    )
    parser.add_argument(
        "--acfl-pta-method",
        choices=["hard", "soft", "conservative"],
        default=None,
        help="PTA selector: hard winner, soft mixture, or conservative fallback",
    )
    parser.add_argument(
        "--acfl-pta-default-branch",
        choices=["ungated", "gated"],
        default=None,
        help="PTA production default / tie-break branch",
    )
    parser.add_argument(
        "--acfl-pta-tau",
        type=float,
        default=None,
        help="Conservative PTA margin before switching branches",
    )
    parser.add_argument(
        "--acfl-pta-lambda-r",
        type=float,
        default=None,
        help="PTA residual-influence weight λ_R",
    )
    parser.add_argument(
        "--acfl-pta-lambda-s",
        type=float,
        default=None,
        help="PTA update-size weight λ_S",
    )
    parser.add_argument(
        "--acfl-pta-lambda-skip",
        type=float,
        default=None,
        help="PTA extra J cost when a branch holds the current global",
    )
    parser.add_argument(
        "--acfl-backdoor-geometry-hard-reject",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Backdoor multi-Krum shortlist-veto geometry gate",
    )
    parser.add_argument(
        "--acfl-label-flip-geometry-hard-reject",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Label-flip multi-Krum shortlist-clean geometry gate",
    )
    parser.add_argument(
        "--acfl-use-layer1-for-model-replacement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Let Layer 1 probe trust modulate model-replacement aggregation",
    )
    parser.add_argument(
        "--acfl-update-norm-hard-rejection-for-model-replacement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto hard-reject extreme norms for model_replacement evaluation",
    )
    parser.add_argument(
        "--acfl-post-agg-trigger-scrub",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Post-aggregation SGD scrub on trigger/benign probes (backdoor runs)",
    )
    parser.add_argument(
        "--acfl-probe-max-windows-per-scenario",
        type=int,
        default=None,
        help="Cap probe windows per scenario (0 = use full probe set)",
    )
    parser.add_argument(
        "--acfl-probe-attack-scoped",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Evaluate only attack-relevant probe scenarios",
    )
    parser.add_argument("--server-only", action="store_true")
    parser.add_argument("--use-labels", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not os.path.isdir(args.data_dir):
        logger.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    data_dir = os.path.abspath(args.data_dir)
    available_clients = count_clients_in_data_dir(data_dir)
    num_clients = args.num_clients
    if available_clients > 0:
        num_clients = min(num_clients, available_clients)
        logger.info("Found %d client data file(s) in %s", available_clients, data_dir)
    elif not glob.glob(os.path.join(data_dir, "*.csv")):
        logger.error("No CSV data files found in %s", data_dir)
        sys.exit(1)

    os.environ["FL_CONFIG_MODULE"] = args.config
    if args.max_fl_rounds:
        os.environ["FL_MAX_ROUNDS"] = str(args.max_fl_rounds)
    if args.fl_seed is not None:
        os.environ["FL_SEED"] = str(args.fl_seed)

    server_script = (
        os.path.join(DRIFT_ROOT, "fl-time-series", "fl_server_time_series.py")
        if args.fedavg or args.baseline_aggregation
        else os.path.join(ACFL_ROOT, "acfl", "acfl_server.py")
    )
    server_cmd = [
        sys.executable,
        server_script,
        "--host", args.host,
        "--port", str(args.port),
        "--num-clients", str(num_clients),
        "--min-clients", str(args.min_clients),
        "--learning-mode", args.learning_mode,
    ]
    if args.fedavg or args.baseline_aggregation:
        agg = args.baseline_aggregation or "fedavg"
        server_cmd.extend(["--aggregation", agg])
    if not args.fedavg and not args.baseline_aggregation:
        if args.no_layer1:
            server_cmd.append("--no-layer1")
        if args.no_layer2:
            server_cmd.append("--no-layer2")
        if args.no_layer3:
            server_cmd.append("--no-layer3")
        if args.malicious_clients:
            server_cmd.extend(["--malicious-clients", args.malicious_clients])
        if args.attack_type != "none":
            server_cmd.extend(["--evaluation-attack-type", args.attack_type])
        if args.probe_file:
            server_cmd.extend(["--probe-file", os.path.abspath(args.probe_file)])
        if args.acfl_backdoor_probe_soft_cap is not None:
            flag = (
                "--acfl-backdoor-probe-soft-cap"
                if args.acfl_backdoor_probe_soft_cap
                else "--no-acfl-backdoor-probe-soft-cap"
            )
            server_cmd.append(flag)
        if args.acfl_backdoor_trigger_hard_gate is not None:
            flag = (
                "--acfl-backdoor-trigger-hard-gate"
                if args.acfl_backdoor_trigger_hard_gate
                else "--no-acfl-backdoor-trigger-hard-gate"
            )
            server_cmd.append(flag)
        if args.acfl_backdoor_probe_hard_reject is not None:
            flag = (
                "--acfl-backdoor-probe-hard-reject"
                if args.acfl_backdoor_probe_hard_reject
                else "--no-acfl-backdoor-probe-hard-reject"
            )
            server_cmd.append(flag)
        if args.acfl_layer1_layer2_gating is not None:
            flag = (
                "--acfl-layer1-layer2-gating"
                if args.acfl_layer1_layer2_gating
                else "--no-acfl-layer1-layer2-gating"
            )
            server_cmd.append(flag)
        if args.acfl_adaptive_l1l2_gating is not None:
            flag = (
                "--acfl-adaptive-l1l2-gating"
                if args.acfl_adaptive_l1l2_gating
                else "--no-acfl-adaptive-l1l2-gating"
            )
            server_cmd.append(flag)
        if args.acfl_parallel_trust_arbitration is not None:
            flag = (
                "--acfl-parallel-trust-arbitration"
                if args.acfl_parallel_trust_arbitration
                else "--no-acfl-parallel-trust-arbitration"
            )
            server_cmd.append(flag)
        if args.acfl_pta_method is not None:
            server_cmd.extend(["--acfl-pta-method", str(args.acfl_pta_method)])
        if args.acfl_pta_default_branch is not None:
            server_cmd.extend(
                ["--acfl-pta-default-branch", str(args.acfl_pta_default_branch)]
            )
        if args.acfl_pta_tau is not None:
            server_cmd.extend(["--acfl-pta-tau", str(args.acfl_pta_tau)])
        if args.acfl_pta_lambda_r is not None:
            server_cmd.extend(["--acfl-pta-lambda-r", str(args.acfl_pta_lambda_r)])
        if args.acfl_pta_lambda_s is not None:
            server_cmd.extend(["--acfl-pta-lambda-s", str(args.acfl_pta_lambda_s)])
        if args.acfl_pta_lambda_skip is not None:
            server_cmd.extend(
                ["--acfl-pta-lambda-skip", str(args.acfl_pta_lambda_skip)]
            )
        if args.acfl_backdoor_geometry_hard_reject is not None:
            flag = (
                "--acfl-backdoor-geometry-hard-reject"
                if args.acfl_backdoor_geometry_hard_reject
                else "--no-acfl-backdoor-geometry-hard-reject"
            )
            server_cmd.append(flag)
        if args.acfl_label_flip_geometry_hard_reject is not None:
            flag = (
                "--acfl-label-flip-geometry-hard-reject"
                if args.acfl_label_flip_geometry_hard_reject
                else "--no-acfl-label-flip-geometry-hard-reject"
            )
            server_cmd.append(flag)
        if args.acfl_use_layer1_for_model_replacement is not None:
            flag = (
                "--acfl-use-layer1-for-model-replacement"
                if args.acfl_use_layer1_for_model_replacement
                else "--no-acfl-use-layer1-for-model-replacement"
            )
            server_cmd.append(flag)
        if args.acfl_update_norm_hard_rejection_for_model_replacement is not None:
            flag = (
                "--acfl-update-norm-hard-rejection-for-model-replacement"
                if args.acfl_update_norm_hard_rejection_for_model_replacement
                else "--no-acfl-update-norm-hard-rejection-for-model-replacement"
            )
            server_cmd.append(flag)
        if args.acfl_post_agg_trigger_scrub is not None:
            flag = (
                "--acfl-post-agg-trigger-scrub"
                if args.acfl_post_agg_trigger_scrub
                else "--no-acfl-post-agg-trigger-scrub"
            )
            server_cmd.append(flag)
        if args.acfl_probe_max_windows_per_scenario is not None:
            server_cmd.extend([
                "--acfl-probe-max-windows-per-scenario",
                str(args.acfl_probe_max_windows_per_scenario),
            ])
        if args.acfl_probe_attack_scoped is not None:
            flag = (
                "--acfl-probe-attack-scoped"
                if args.acfl_probe_attack_scoped
                else "--no-acfl-probe-attack-scoped"
            )
            server_cmd.append(flag)
        acfl_overrides = {
            "--acfl-trust-weight-power": args.acfl_trust_weight_power,
            "--acfl-min-trust-weight": args.acfl_min_trust_weight,
            "--acfl-subspace-weight": args.acfl_subspace_weight,
            "--acfl-temporal-weight": args.acfl_temporal_weight,
            "--acfl-counterfactual-weight": args.acfl_counterfactual_weight,
            "--acfl-coordinate-deviation-threshold": args.acfl_coordinate_deviation_threshold,
            "--acfl-norm-ratio-threshold": args.acfl_norm_ratio_threshold,
            "--acfl-norm-ratio-threshold-two-clients": args.acfl_norm_ratio_threshold_two_clients,
        }
        for flag, value in acfl_overrides.items():
            if value is not None:
                server_cmd.extend([flag, str(value)])

    server_env = os.environ.copy()
    server_env["FL_CONFIG_MODULE"] = args.config

    logger.info(
        "Starting %s server...",
        (
            f"baseline/{args.baseline_aggregation or 'fedavg'}"
            if (args.fedavg or args.baseline_aggregation)
            else "ACFL"
        ),
    )
    server_proc = subprocess.Popen(server_cmd, cwd=FL_TS_DIR, env=server_env)
    processes.append(server_proc)

    if not wait_for_server(args.host, args.port):
        logger.error("Server failed to start")
        cleanup_processes()
        sys.exit(1)

    if args.server_only:
        logger.info("Server running (server-only mode). Press Ctrl+C to stop.")
        server_proc.wait()
        return

    logger.info("Launching %d FL clients...", num_clients)
    client_script = os.path.join(FL_TS_DIR, "fl_client_time_series.py")

    for i in range(1, num_clients + 1):
        client_id = f"client_{i}"
        client_gps, client_imu, client_packet = get_client_files(
            client_id, data_dir, None, None, None
        )

        client_cmd = [
            sys.executable, "-u", client_script,
            "--client-id", client_id,
            "--server-url", f"http://{args.host}:{args.port}",
            "--data-dir", data_dir,
        ]

        if client_gps:
            client_cmd.extend(["--gps-file", client_gps])
        if client_imu:
            client_cmd.extend(["--imu-file", client_imu])
        if client_packet:
            client_cmd.extend(["--packet-file", client_packet])

        use_labels = args.use_labels or args.learning_mode == "classification"
        if use_labels and (client_packet or args.learning_mode == "classification"):
            client_cmd.append("--use-labels")

        client_env = os.environ.copy()
        client_env["FL_CONFIG_MODULE"] = args.config
        if args.malicious_clients and args.attack_type != "none":
            client_env["ACFL_ATTACK_CLIENTS"] = args.malicious_clients
            client_env["ACFL_ATTACK_TYPE"] = args.attack_type
            client_env["ACFL_ATTACK_SCALE"] = str(args.attack_scale)
            client_env["ACFL_ATTACK_NOISE"] = str(args.attack_noise)
            if args.attack_type == "backdoor":
                client_env["ACFL_BACKDOOR_POISON_FRACTION"] = str(
                    args.backdoor_poison_fraction
                )
            if args.attack_type == "label_flip":
                client_env["ACFL_LABEL_FLIP_FRACTION"] = str(
                    args.label_flip_fraction
                )

        proc = subprocess.Popen(
            client_cmd,
            cwd=FL_TS_DIR,
            env=client_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        proc.client_id = client_id
        processes.append(proc)

        def stream_output(p, cid):
            for line in iter(p.stdout.readline, ""):
                if line:
                    logger.info("[%s] %s", cid, line.rstrip())

        threading.Thread(
            target=stream_output, args=(proc, client_id), daemon=True
        ).start()

        logger.info("Waiting for %s to register...", client_id)
        if not wait_for_client_registered(args.host, args.port, i, max_wait=180):
            logger.warning("%s did not register within 180s", client_id)

    monitor_thread = threading.Thread(
        target=monitor_training,
        args=(args.host, args.port, args.num_rounds),
        daemon=True,
    )
    monitor_thread.start()

    logger.info("Waiting for training to complete...")
    max_wait = 600
    for proc in processes[1:]:
        try:
            proc.wait(timeout=max_wait)
        except subprocess.TimeoutExpired:
            logger.warning("%s timed out after %ds", proc.client_id, max_wait)
            proc.kill()

    try:
        resp = requests.get(f"http://{args.host}:{args.port}/status", timeout=10)
        if resp.status_code == 200:
            status = resp.json()
            logger.info(
                "Final round: %d | registered: %d",
                status.get("current_round", 0),
                status.get("registered_clients", 0),
            )
            if "acfl" in status:
                logger.info("\nACFL Defense Summary:")
                for k, v in status["acfl"]["summary"].items():
                    logger.info("  %s: %s", k, v)
    except Exception as e:
        logger.warning("Could not fetch final status: %s", e)

    cleanup_processes()
    logger.info("ACFL training complete.")


if __name__ == "__main__":
    main()
