#!/usr/bin/env python3
"""
Comprehensive ACFL validation on the SITL-generated dataset.

Runs sequentially (memory-safe on limited hardware):
  1. Unit tests (pytest)
  2. Probe-set generation from raw SITL packet data
  3. FL scenarios on raw temporal packets (window oversampling + pos_weight for training)
  4. Per-scenario ACFL + drift-fairroad evaluation metrics
  5. Aggregated JSON report

Usage:
    ./acfl_env/bin/python benchmarks/run_comprehensive_validation.py
    ./acfl_env/bin/python benchmarks/run_comprehensive_validation.py --quick   # fewer scenarios
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import requests

# Allow `from run_comprehensive_validation import ...` across this package
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)


def _resolve_repo_roots(here: str):
    """Locate repo root from this file (root or benchmarks/)."""
    start = os.path.dirname(os.path.abspath(here))
    root = start
    for _ in range(5):
        if os.path.isdir(os.path.join(root, "fl-time-series")):
            return root, root
        nested = os.path.join(root, "drift-fairroad")
        if os.path.isdir(os.path.join(nested, "fl-time-series")):
            return root, nested
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    # Fallback: parent of benchmarks/ if present
    if os.path.basename(start) == "benchmarks":
        root = os.path.dirname(start)
        if os.path.isdir(os.path.join(root, "fl-time-series")):
            return root, root
    return start, start


def _resolve_python(repo_root: str) -> str:
    for cand in (
        os.path.join(repo_root, "acfl_env", "bin", "python"),
        os.path.join(repo_root, "fl_env", "bin", "python"),
        sys.executable,
    ):
        if cand == sys.executable or os.path.isfile(cand):
            return cand
    return sys.executable

ACFL_ROOT, DRIFT_ROOT = _resolve_repo_roots(__file__)
FL_TS_DIR = os.path.join(DRIFT_ROOT, "fl-time-series")
CHECKPOINT_DIR = os.path.join(FL_TS_DIR, "checkpoints")
SITL_PKT = os.path.join(DRIFT_ROOT, "data", "sitl_dataset", "fl", "packets")
SITL_GI = os.path.join(DRIFT_ROOT, "data", "sitl_dataset", "fl", "gps-imu")
SITL_EXTRA_RUNS = os.path.join(DRIFT_ROOT, "data", "sitl_validate", "runs")
OUT_ROOT = os.path.join(ACFL_ROOT, "evaluation_results", "comprehensive_validation")

PYTHON = _resolve_python(ACFL_ROOT)


@dataclass
class Scenario:
    name: str
    data_dir: str
    config: str
    learning_mode: str
    malicious_clients: str = ""
    attack_type: str = "none"
    attack_scale: float = 10.0
    attack_noise: Optional[float] = None
    no_layer1: bool = False
    no_layer2: bool = False
    no_layer3: bool = False
    probe_file: Optional[str] = None
    fedavg: bool = False
    baseline_aggregation: Optional[str] = None
    num_clients: int = 7
    min_clients: int = 3
    target_rounds: int = 5
    max_fl_rounds: int = 5
    acfl_trust_weight_power: Optional[float] = None
    acfl_min_trust_weight: Optional[float] = None
    acfl_subspace_weight: Optional[float] = None
    acfl_temporal_weight: Optional[float] = None
    acfl_counterfactual_weight: Optional[float] = None
    acfl_coordinate_deviation_threshold: Optional[float] = None
    acfl_norm_ratio_threshold: Optional[float] = None
    acfl_norm_ratio_threshold_two_clients: Optional[float] = None
    acfl_backdoor_probe_soft_cap: Optional[bool] = None
    acfl_backdoor_trigger_hard_gate: Optional[bool] = None
    acfl_backdoor_probe_hard_reject: Optional[bool] = None
    acfl_post_agg_trigger_scrub: Optional[bool] = None
    acfl_use_layer1_for_model_replacement: Optional[bool] = None
    acfl_update_norm_hard_rejection_for_model_replacement: Optional[bool] = None
    acfl_layer1_layer2_gating: Optional[bool] = None
    acfl_adaptive_l1l2_gating: Optional[bool] = None
    acfl_parallel_trust_arbitration: Optional[bool] = None
    acfl_pta_method: Optional[str] = None
    acfl_pta_default_branch: Optional[str] = None
    acfl_pta_tau: Optional[float] = None
    acfl_pta_lambda_r: Optional[float] = None
    acfl_pta_lambda_s: Optional[float] = None
    acfl_pta_lambda_skip: Optional[float] = None
    acfl_backdoor_geometry_hard_reject: Optional[bool] = None
    acfl_label_flip_geometry_hard_reject: Optional[bool] = None
    # Probe cost/quality knobs (None = use config defaults; 0 = unlimited windows).
    acfl_probe_max_windows_per_scenario: Optional[int] = None
    acfl_probe_attack_scoped: Optional[bool] = None
    backdoor_poison_fraction: Optional[float] = None
    label_flip_fraction: Optional[float] = None
    fl_seed: Optional[int] = None
    quick_fl_eval: bool = False


def _count_agents(packets_dir: str) -> int:
    sys.path.insert(0, FL_TS_DIR)
    from agent_mission_data import count_agents
    return count_agents(packets_dir)


def _apply_fl_client_scale(scenarios: List[Scenario], num_clients: int, min_clients: int) -> List[Scenario]:
    """Cap FL participants to available agents (not missions)."""
    available = _count_agents(SITL_PKT)
    if available > 0:
        num_clients = min(num_clients, available)
    min_clients = min(min_clients, num_clients)
    out = []
    for sc in scenarios:
        if sc.learning_mode == "classification":
            out.append(replace(sc, num_clients=num_clients, min_clients=min_clients))
        else:
            out.append(sc)
    return out


def sync_agent_missions() -> Dict:
    """Rebuild agent/mission layout from SITL runs (extra missions attach to agents)."""
    extra = []
    for rel in (
        "data/sitl_validate/runs",
        "data/sitl_drone_extra/runs",
    ):
        p = os.path.join(DRIFT_ROOT, rel)
        if os.path.isdir(p):
            extra.append(p)
    cmd = [
        PYTHON, os.path.join(ACFL_ROOT, "scripts", "build_agent_mission_dataset.py"),
        "--train-runs", os.path.join(DRIFT_ROOT, "data", "sitl_dataset", "runs"),
        "--output", SITL_PKT,
        "--num-agents", "10",
        "--clean",
    ]
    for e in extra:
        cmd.extend(["--extra-runs", e])
    rc = _run(cmd, timeout=120)
    return {"ok": rc == 0, "packets_dir": SITL_PKT, "extra_runs": extra}


def build_scenarios(quick: bool = False) -> List[Scenario]:
    pkt = SITL_PKT
    full = [
        Scenario("classification_benign", pkt, "config_classification",
                 "classification", target_rounds=5, max_fl_rounds=5),
        Scenario("classification_fedavg_poisoned", pkt, "config_classification",
                 "classification", malicious_clients="client_2",
                 attack_type="gradient_scaling", fedavg=True, target_rounds=5, max_fl_rounds=5),
        Scenario("classification_gradient_scaling", pkt, "config_classification",
                 "classification", malicious_clients="client_2",
                 attack_type="gradient_scaling", target_rounds=5, max_fl_rounds=5),
        Scenario("classification_byzantine", pkt, "config_classification",
                 "classification", malicious_clients="client_2",
                 attack_type="byzantine", attack_scale=1.0, target_rounds=5, max_fl_rounds=5),
        Scenario("classification_ablation_no_l1", pkt, "config_classification",
                 "classification", malicious_clients="client_2",
                 attack_type="gradient_scaling", no_layer1=True, target_rounds=5, max_fl_rounds=5),
        Scenario("classification_ablation_no_l2", pkt, "config_classification",
                 "classification", malicious_clients="client_2",
                 attack_type="gradient_scaling", no_layer2=True, target_rounds=5, max_fl_rounds=5),
        Scenario("classification_with_probes", pkt, "config_classification",
                 "classification", malicious_clients="client_2",
                 attack_type="gradient_scaling",
                 probe_file=os.path.join(CHECKPOINT_DIR, "acfl_probes_sitl.pt"),
                 target_rounds=5, max_fl_rounds=5),
        Scenario("regression_benign", SITL_GI, "config_regression",
                 "regression", target_rounds=5, max_fl_rounds=5),
        Scenario("regression_poisoned", SITL_GI, "config_regression",
                 "regression", malicious_clients="client_2",
                 attack_type="gradient_scaling", target_rounds=5, max_fl_rounds=5),
    ]
    if quick:
        return [full[0], full[1], full[2], full[4], full[7]]
    return full


def build_drone_scenarios(quick: bool = False) -> List[Scenario]:
    """Drone/edge profile: TinyPacketCNN, 3 clients, 12 rounds, fp16 comms."""
    pkt = SITL_PKT
    rounds = 12
    base = dict(
        data_dir=pkt,
        config="config_classification_drone",
        learning_mode="classification",
        num_clients=3,
        min_clients=2,
        target_rounds=rounds,
        max_fl_rounds=rounds,
    )
    full = [
        Scenario("drone_classification_benign", **base),
        Scenario("drone_classification_fedavg_poisoned", **base,
                 malicious_clients="client_2", attack_type="gradient_scaling", fedavg=True),
        Scenario("drone_classification_gradient_scaling", **base,
                 malicious_clients="client_2", attack_type="gradient_scaling"),
        Scenario("drone_classification_ablation_no_l1", **base,
                 malicious_clients="client_2", attack_type="gradient_scaling", no_layer1=True),
    ]
    if quick:
        return [full[0], full[1], full[2], full[3]]
    return full


def _run(cmd: List[str], cwd: str = ACFL_ROOT, timeout: int = 600) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, cwd=cwd, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return -1


def run_pytest() -> Dict:
    rc = _run([PYTHON, "-m", "pytest", "tests/", "-v", "--tb=short"], timeout=120)
    return {"passed": rc == 0, "returncode": rc}


def generate_probes() -> Dict:
    sys.path.insert(0, FL_TS_DIR)
    from agent_mission_data import list_mission_files_for_agent

    missions = list_mission_files_for_agent(SITL_PKT, 0)
    pkt = missions[0] if missions else os.path.join(SITL_PKT, "packet_client_000.csv")
    out = os.path.join(CHECKPOINT_DIR, "acfl_probes_sitl.pt")
    if not os.path.isfile(pkt):
        return {"ok": False, "error": f"missing {pkt}"}
    rc = _run([
        PYTHON, os.path.join(ACFL_ROOT, "scripts", "generate_probe_set.py"), "--mode", "classification",
        "--packet-file", pkt, "--output", out,
    ], timeout=120)
    return {"ok": rc == 0, "probe_file": out if rc == 0 else None}


def _clear_checkpoints():
    for name in ("acfl_trust_scores.json", "acfl_round_reports.json"):
        p = os.path.join(CHECKPOINT_DIR, name)
        if os.path.isfile(p):
            os.remove(p)


def _clear_fl_checkpoints():
    """Remove stale server/client checkpoints so each scenario starts clean."""
    for p in glob.glob(os.path.join(CHECKPOINT_DIR, "server_round_*.pt")):
        if os.path.isfile(p):
            os.remove(p)
    clients_dir = os.path.join(CHECKPOINT_DIR, "clients")
    if os.path.isdir(clients_dir):
        for p in glob.glob(os.path.join(clients_dir, "*.pt")):
            if os.path.isfile(p):
                os.remove(p)


def _archive_fl_checkpoints(sc_out: str, final_round: int) -> Optional[str]:
    """Copy this scenario's FL checkpoints into the scenario output directory."""
    if final_round <= 0:
        return None
    dst_dir = os.path.join(sc_out, "fl_checkpoints")
    clients_dst = os.path.join(dst_dir, "clients")
    os.makedirs(clients_dst, exist_ok=True)
    copied = 0
    for r in range(1, final_round + 1):
        src = os.path.join(CHECKPOINT_DIR, f"server_round_{r}.pt")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, f"server_round_{r}.pt"))
            copied += 1
    for p in glob.glob(os.path.join(CHECKPOINT_DIR, "clients", "*.pt")):
        shutil.copy2(p, os.path.join(clients_dst, os.path.basename(p)))
    return dst_dir if copied else None


def _archive_server_checkpoint(sc_out: str, final_round: int,
                               fl_ckpt_dir: Optional[str] = None) -> Optional[str]:
    """Copy the scenario's final server checkpoint for Layer 3 inference."""
    if final_round <= 0:
        return None
    candidates = []
    if fl_ckpt_dir:
        candidates.append(os.path.join(fl_ckpt_dir, f"server_round_{final_round}.pt"))
    candidates.append(os.path.join(CHECKPOINT_DIR, f"server_round_{final_round}.pt"))
    if not any(os.path.isfile(c) for c in candidates):
        pool = glob.glob(os.path.join(fl_ckpt_dir or CHECKPOINT_DIR, "server_round_*.pt"))
        pool += glob.glob(os.path.join(CHECKPOINT_DIR, "server_round_*.pt"))
        matches = sorted(
            set(pool),
            key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
        )
        candidates.append(matches[-1] if matches else None)
    src = next((c for c in candidates if c and os.path.isfile(c)), None)
    if not src:
        return None
    dst = os.path.join(sc_out, "server_checkpoint.pt")
    shutil.copy2(src, dst)
    return dst


def _wait_for_rounds(host: str, port: int, target: int, timeout_s: int = 300) -> int:
    url = f"http://{host}:{port}/status"
    deadline = time.time() + timeout_s
    last = -1
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                rnd = r.json().get("current_round", 0)
                if rnd != last:
                    print(f"  server round {rnd}/{target}")
                    last = rnd
                if rnd >= target:
                    return rnd
        except requests.RequestException:
            pass
        time.sleep(2)
    return last


def _terminate(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (PermissionError, ProcessLookupError, OSError):
            pass
    except (PermissionError, ProcessLookupError, OSError):
        # Sandbox / privilege mismatch: parent cannot signal child. Leave
        # cleanup to the OS / operator rather than aborting evaluation.
        pass


def run_scenario(
    sc: Scenario,
    host: str = "localhost",
    port: int = 5544,
    scenario_timeout_s: Optional[int] = None,
    out_root: Optional[str] = None,
) -> Dict:
    if not os.path.isdir(sc.data_dir):
        return {"ok": False, "error": f"data dir missing: {sc.data_dir}"}

    scenario_root = out_root or OUT_ROOT
    sc_out = os.path.join(scenario_root, sc.name)
    os.makedirs(sc_out, exist_ok=True)
    _clear_checkpoints()
    _clear_fl_checkpoints()

    cmd = [
        PYTHON, "-u", "run_acfl_system.py",
        "--data-dir", sc.data_dir,
        "--config", sc.config,
        "--learning-mode", sc.learning_mode,
        "--num-clients", str(sc.num_clients),
        "--min-clients", str(sc.min_clients),
        "--num-rounds", str(sc.target_rounds),
        "--max-fl-rounds", str(sc.max_fl_rounds),
        "--host", host, "--port", str(port),
    ]
    if sc.baseline_aggregation:
        cmd.extend(["--baseline-aggregation", sc.baseline_aggregation])
    elif sc.fedavg:
        cmd.append("--fedavg")
    if sc.malicious_clients:
        cmd.extend(["--malicious-clients", sc.malicious_clients])
    if sc.attack_type != "none":
        cmd.extend(["--attack-type", sc.attack_type,
                    "--attack-scale", str(sc.attack_scale)])
        if sc.attack_noise is not None:
            cmd.extend(["--attack-noise", str(sc.attack_noise)])
        if sc.attack_type == "backdoor" and sc.backdoor_poison_fraction is not None:
            cmd.extend([
                "--backdoor-poison-fraction",
                str(sc.backdoor_poison_fraction),
            ])
        if sc.attack_type == "label_flip" and sc.label_flip_fraction is not None:
            cmd.extend([
                "--label-flip-fraction",
                str(sc.label_flip_fraction),
            ])
    if sc.no_layer1:
        cmd.append("--no-layer1")
    if sc.no_layer2:
        cmd.append("--no-layer2")
    if sc.no_layer3:
        cmd.append("--no-layer3")
    if sc.probe_file and os.path.isfile(sc.probe_file):
        cmd.extend(["--probe-file", sc.probe_file])
    if sc.fl_seed is not None:
        cmd.extend(["--fl-seed", str(sc.fl_seed)])
    if not sc.fedavg and not sc.baseline_aggregation:
        acfl_overrides = {
            "--acfl-trust-weight-power": sc.acfl_trust_weight_power,
            "--acfl-min-trust-weight": sc.acfl_min_trust_weight,
            "--acfl-subspace-weight": sc.acfl_subspace_weight,
            "--acfl-temporal-weight": sc.acfl_temporal_weight,
            "--acfl-counterfactual-weight": sc.acfl_counterfactual_weight,
            "--acfl-coordinate-deviation-threshold": sc.acfl_coordinate_deviation_threshold,
            "--acfl-norm-ratio-threshold": sc.acfl_norm_ratio_threshold,
            "--acfl-norm-ratio-threshold-two-clients": sc.acfl_norm_ratio_threshold_two_clients,
        }
        for flag, value in acfl_overrides.items():
            if value is not None:
                cmd.extend([flag, str(value)])
        if sc.acfl_backdoor_probe_soft_cap is not None:
            flag = (
                "--acfl-backdoor-probe-soft-cap"
                if sc.acfl_backdoor_probe_soft_cap
                else "--no-acfl-backdoor-probe-soft-cap"
            )
            cmd.append(flag)
        if sc.acfl_backdoor_trigger_hard_gate is not None:
            flag = (
                "--acfl-backdoor-trigger-hard-gate"
                if sc.acfl_backdoor_trigger_hard_gate
                else "--no-acfl-backdoor-trigger-hard-gate"
            )
            cmd.append(flag)
        if sc.acfl_backdoor_probe_hard_reject is not None:
            flag = (
                "--acfl-backdoor-probe-hard-reject"
                if sc.acfl_backdoor_probe_hard_reject
                else "--no-acfl-backdoor-probe-hard-reject"
            )
            cmd.append(flag)
        if sc.acfl_post_agg_trigger_scrub is not None:
            flag = (
                "--acfl-post-agg-trigger-scrub"
                if sc.acfl_post_agg_trigger_scrub
                else "--no-acfl-post-agg-trigger-scrub"
            )
            cmd.append(flag)
        if sc.acfl_use_layer1_for_model_replacement is not None:
            flag = (
                "--acfl-use-layer1-for-model-replacement"
                if sc.acfl_use_layer1_for_model_replacement
                else "--no-acfl-use-layer1-for-model-replacement"
            )
            cmd.append(flag)
        if sc.acfl_update_norm_hard_rejection_for_model_replacement is not None:
            flag = (
                "--acfl-update-norm-hard-rejection-for-model-replacement"
                if sc.acfl_update_norm_hard_rejection_for_model_replacement
                else "--no-acfl-update-norm-hard-rejection-for-model-replacement"
            )
            cmd.append(flag)
        if sc.acfl_layer1_layer2_gating is not None:
            flag = (
                "--acfl-layer1-layer2-gating"
                if sc.acfl_layer1_layer2_gating
                else "--no-acfl-layer1-layer2-gating"
            )
            cmd.append(flag)
        if sc.acfl_adaptive_l1l2_gating is not None:
            flag = (
                "--acfl-adaptive-l1l2-gating"
                if sc.acfl_adaptive_l1l2_gating
                else "--no-acfl-adaptive-l1l2-gating"
            )
            cmd.append(flag)
        if sc.acfl_parallel_trust_arbitration is not None:
            flag = (
                "--acfl-parallel-trust-arbitration"
                if sc.acfl_parallel_trust_arbitration
                else "--no-acfl-parallel-trust-arbitration"
            )
            cmd.append(flag)
        if sc.acfl_pta_method is not None:
            cmd.extend(["--acfl-pta-method", str(sc.acfl_pta_method)])
        if sc.acfl_pta_default_branch is not None:
            cmd.extend(
                ["--acfl-pta-default-branch", str(sc.acfl_pta_default_branch)]
            )
        if sc.acfl_pta_tau is not None:
            cmd.extend(["--acfl-pta-tau", str(sc.acfl_pta_tau)])
        if sc.acfl_pta_lambda_r is not None:
            cmd.extend(["--acfl-pta-lambda-r", str(sc.acfl_pta_lambda_r)])
        if sc.acfl_pta_lambda_s is not None:
            cmd.extend(["--acfl-pta-lambda-s", str(sc.acfl_pta_lambda_s)])
        if sc.acfl_pta_lambda_skip is not None:
            cmd.extend(["--acfl-pta-lambda-skip", str(sc.acfl_pta_lambda_skip)])
        if sc.acfl_backdoor_geometry_hard_reject is not None:
            flag = (
                "--acfl-backdoor-geometry-hard-reject"
                if sc.acfl_backdoor_geometry_hard_reject
                else "--no-acfl-backdoor-geometry-hard-reject"
            )
            cmd.append(flag)
        if sc.acfl_label_flip_geometry_hard_reject is not None:
            flag = (
                "--acfl-label-flip-geometry-hard-reject"
                if sc.acfl_label_flip_geometry_hard_reject
                else "--no-acfl-label-flip-geometry-hard-reject"
            )
            cmd.append(flag)
        if sc.acfl_probe_max_windows_per_scenario is not None:
            cmd.extend([
                "--acfl-probe-max-windows-per-scenario",
                str(sc.acfl_probe_max_windows_per_scenario),
            ])
        if sc.acfl_probe_attack_scoped is not None:
            flag = (
                "--acfl-probe-attack-scoped"
                if sc.acfl_probe_attack_scoped
                else "--no-acfl-probe-attack-scoped"
            )
            cmd.append(flag)

    print(f"\n{'='*60}\nSCENARIO: {sc.name}\n{'='*60}")
    log_path = os.path.join(sc_out, "run.log")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=ACFL_ROOT, stdout=logf, stderr=subprocess.STDOUT,
        )
        if scenario_timeout_s is not None:
            deadline = time.time() + scenario_timeout_s
        else:
            deadline = time.time() + (900 if sc.num_clients > 5 else 600)
        final_round = 0
        while proc.poll() is None and time.time() < deadline:
            final_round = _wait_for_rounds(host, port, sc.target_rounds, timeout_s=10)
            if final_round >= sc.target_rounds:
                time.sleep(2)
                break
            time.sleep(1)
        _terminate(proc)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (PermissionError, ProcessLookupError, OSError):
                pass
        except (PermissionError, ProcessLookupError, OSError):
            pass

    for name in ("acfl_trust_scores.json", "acfl_round_reports.json"):
        src = os.path.join(CHECKPOINT_DIR, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(sc_out, name))

    fl_ckpt_dir = _archive_fl_checkpoints(sc_out, final_round)
    checkpoint_path = _archive_server_checkpoint(sc_out, final_round, fl_ckpt_dir)
    metrics = evaluate_scenario(sc, sc_out, fl_checkpoint_dir=fl_ckpt_dir, final_round=final_round)
    metrics.update({
        "scenario": sc.name,
        "final_round": final_round,
        "target_rounds": sc.target_rounds,
        "malicious_clients": sc.malicious_clients,
        "attack_type": sc.attack_type,
        "checkpoint": checkpoint_path,
        "fl_checkpoint_dir": fl_ckpt_dir,
        "log": log_path,
    })
    with open(os.path.join(sc_out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {sc_out}/metrics.json")
    return metrics


def _fl_eval_summary(fl_eval: Dict) -> Dict:
    """Aggregate per-client FL metrics into scenario-level summaries."""
    summary = fl_eval.get("summary") or {}
    if not summary:
        return {}
    accs, f1s, rmses = [], [], []
    for stats in summary.values():
        if "final_accuracy" in stats:
            accs.append(stats["final_accuracy"])
        if "final_f1_score" in stats:
            f1s.append(stats["final_f1_score"])
        if "final_rmse" in stats:
            rmses.append(stats["final_rmse"])
    out = {"clients_evaluated": len(summary)}
    if accs:
        out["mean_final_accuracy"] = sum(accs) / len(accs)
    if f1s:
        out["mean_final_f1"] = sum(f1s) / len(f1s)
    if rmses:
        out["mean_final_rmse"] = sum(rmses) / len(rmses)
    out["num_checkpoints"] = fl_eval.get("num_checkpoints")
    return out


def _build_defense_comparison(scenarios: List[Dict], profile: str = "standard") -> Dict:
    """Compare FedAvg-under-attack vs ACFL-under-attack model quality."""
    by_name = {s.get("scenario"): s for s in scenarios if isinstance(s, dict)}
    if profile == "drone":
        fedavg_key = "drone_classification_fedavg_poisoned"
        acfl_key = "drone_classification_gradient_scaling"
    else:
        fedavg_key = "classification_fedavg_poisoned"
        acfl_key = "classification_gradient_scaling"
    fedavg = by_name.get(fedavg_key, {})
    acfl = by_name.get(acfl_key, {})
    fed_fl = _fl_eval_summary(fedavg.get("fl_eval", {}))
    acfl_fl = _fl_eval_summary(acfl.get("fl_eval", {}))
    return {
        "fedavg_poisoned": {
            "final_round": fedavg.get("final_round"),
            "fl_summary": fed_fl,
        },
        "acfl_gradient_scaling": {
            "final_round": acfl.get("final_round"),
            "trust_auc": acfl.get("acfl_eval", {}).get("trust_score_auc"),
            "fl_summary": acfl_fl,
        },
        "acfl_trust_advantage": (
            acfl.get("acfl_eval", {}).get("trust_score_auc") is not None
            and acfl.get("acfl_eval", {}).get("trust_score_auc", 0.5) > 0.5
        ),
    }


def evaluate_scenario(
    sc: Scenario,
    sc_out: str,
    fl_checkpoint_dir: Optional[str] = None,
    final_round: Optional[int] = None,
) -> Dict:
    metrics = {}
    malicious = set(c.strip() for c in sc.malicious_clients.split(",") if c.strip())

    eval_script = os.path.join(ACFL_ROOT, "scripts", "evaluate_acfl.py")
    if os.path.isfile(eval_script):
        out_json = os.path.join(sc_out, "acfl_evaluation.json")
        cmd = [
            PYTHON, eval_script,
            "--checkpoint-dir", sc_out,
            "--output-dir", sc_out,
            "--warmup-rounds", "2",
        ]
        if malicious:
            cmd.extend(["--malicious-clients", sc.malicious_clients])
        _run(cmd, timeout=60)
        if os.path.isfile(out_json):
            with open(out_json) as f:
                metrics["acfl_eval"] = json.load(f)

    if (
        sc.attack_type == "backdoor"
        and sc.probe_file
        and os.path.isfile(sc.probe_file)
    ):
        try:
            from acfl.backdoor_eval import evaluate_backdoor_trigger_from_dir

            bd_ckpt_dir = sc_out if os.path.isfile(
                os.path.join(sc_out, "server_checkpoint.pt")
            ) else (fl_checkpoint_dir or sc_out)
            metrics["backdoor_eval"] = evaluate_backdoor_trigger_from_dir(
                bd_ckpt_dir,
                sc.probe_file,
                config_module=sc.config,
                fl_ts_dir=FL_TS_DIR,
            )
            bd = metrics["backdoor_eval"]
            if bd.get("backdoor_success_rate") is not None:
                backdoor_path = os.path.join(sc_out, "backdoor_results.json")
                with open(backdoor_path, "w") as f:
                    json.dump(
                        {
                            "backdoor_success_rate": bd["backdoor_success_rate"],
                            "trigger_accuracy": bd.get("backdoor_trigger_accuracy"),
                            "mean_attack_prob": bd.get("backdoor_mean_attack_prob"),
                        },
                        f,
                        indent=2,
                    )
        except Exception as exc:
            metrics["backdoor_eval_error"] = str(exc)

    fl_eval_script = os.path.join(FL_TS_DIR, "evaluate_fl_system.py")
    ckpt_dir = fl_checkpoint_dir or CHECKPOINT_DIR

    if os.path.isdir(sc.data_dir) and os.path.isfile(fl_eval_script) and ckpt_dir and os.path.isdir(ckpt_dir):
        task = "classification" if sc.learning_mode == "classification" else "regression"
        fl_out = os.path.join(sc_out, "fl_eval")
        os.makedirs(fl_out, exist_ok=True)
        fl_cmd = [
            PYTHON, fl_eval_script,
            "--data-dir", sc.data_dir,
            "--task", task,
            "--config", sc.config,
            "--checkpoint-dir", ckpt_dir,
            "--output-dir", fl_out,
        ]
        if sc.quick_fl_eval:
            fl_cmd.append("--skip-plots")
            fl_cmd.append("--no-client-eval")
            if final_round is not None and final_round > 0:
                fl_cmd.extend(["--rounds", str(final_round)])
        if os.path.isfile(os.path.join(sc_out, "backdoor_results.json")):
            fl_cmd.extend([
                "--backdoor-results",
                os.path.join(sc_out, "backdoor_results.json"),
            ])
        # Full 10-agent checkpoint eval routinely exceeds 3 minutes on CPU.
        rc = _run(fl_cmd, cwd=DRIFT_ROOT, timeout=180 if sc.quick_fl_eval else 900)
        report = os.path.join(fl_out, "evaluation_report.json")
        if os.path.isfile(report):
            with open(report) as f:
                metrics["fl_eval"] = json.load(f)
        metrics["fl_eval_rc"] = rc

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Comprehensive ACFL validation")
    ap.add_argument("--quick", action="store_true", help="Run a subset of scenarios")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="Run only named scenarios")
    ap.add_argument("--drone", action="store_true",
                    help="Use drone/edge profile (TinyPacketCNN, 3 clients, 12 rounds, fp16 comms)")
    ap.add_argument("--num-clients", type=int, default=None,
                    help="FL training clients (default: 3 if --drone else 7)")
    ap.add_argument("--min-clients", type=int, default=None,
                    help="Minimum clients before server aggregates (default: 2 if --drone else 3)")
    args = ap.parse_args()

    num_clients = args.num_clients if args.num_clients is not None else (3 if args.drone else 7)
    min_clients = args.min_clients if args.min_clients is not None else (2 if args.drone else 3)

    os.makedirs(OUT_ROOT, exist_ok=True)
    split_policy = (
        "agents never split train/eval; held-out mission or post-attack eval; "
        "multiple missions per agent allowed"
    )
    if args.drone:
        split_policy += "; drone profile: TinyPacketCNN, fp16+gzip FL weights, 3 clients, 12 rounds"

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": "drone" if args.drone else "standard",
        "data": {
            "packets": SITL_PKT,
            "gps_imu": SITL_GI,
            "split_policy": split_policy,
        },
        "pytest": None,
        "probes": None,
        "scenarios": [],
    }

    if not args.skip_tests:
        report["pytest"] = run_pytest()

    if not args.skip_probes:
        report["agent_missions"] = sync_agent_missions()
        report["probes"] = generate_probes()

    if args.drone:
        try:
            rc = _run([PYTHON, os.path.join(ACFL_ROOT, "scripts", "estimate_drone_resources.py"),
                       "--config", "config_classification_drone"], timeout=30)
            report["drone_resource_estimate"] = {"ok": rc == 0}
        except Exception as e:
            report["drone_resource_estimate"] = {"error": str(e)}

    scenarios = _apply_fl_client_scale(
        build_drone_scenarios(quick=args.quick) if args.drone else build_scenarios(quick=args.quick),
        num_clients, min_clients,
    )
    if args.scenarios:
        all_sc = {s.name: s for s in build_scenarios(quick=False)}
        scenarios = [all_sc[n] for n in args.scenarios if n in all_sc]

    classification_ckpt: Optional[str] = None
    for sc in scenarios:
        try:
            result = run_scenario(sc)
            report["scenarios"].append(result)
            if sc.learning_mode == "classification" and result.get("checkpoint"):
                classification_ckpt = result["checkpoint"]
        except Exception as e:
            report["scenarios"].append({"scenario": sc.name, "ok": False, "error": str(e)})
            print(f"  ERROR in {sc.name}: {e}")

    # Layer 3 inference eval using archived classification checkpoint
    l3_out = os.path.join(OUT_ROOT, "layer3_eval.json")
    ckpt = classification_ckpt
    if ckpt and os.path.isfile(ckpt):
        l3_config = "config_classification_drone" if args.drone else "auto"
        probe_path = os.path.join(CHECKPOINT_DIR, "acfl_probes_sitl.pt")
        l3_cmd = [
            PYTHON, os.path.join(ACFL_ROOT, "scripts", "evaluate_layer3_sitl.py"),
            "--data-dir", SITL_PKT,
            "--config", l3_config,
            "--checkpoint", ckpt,
            "--output", l3_out,
        ]
        if os.path.isfile(probe_path):
            l3_cmd.extend(["--probe-file", probe_path])
        rc = _run(l3_cmd, timeout=300)
        if os.path.isfile(l3_out):
            with open(l3_out) as f:
                report["layer3_eval"] = json.load(f)
        report["layer3_eval_rc"] = rc
        report["layer3_checkpoint"] = ckpt
    else:
        report["layer3_eval_skipped"] = "no classification checkpoint archived"

    summary_path = os.path.join(OUT_ROOT, "validation_report.json")
    report["defense_comparison"] = _build_defense_comparison(
        report["scenarios"], profile=report["profile"],
    )
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Comprehensive validation report: {summary_path}")
    print(f"{'='*60}")
    for sc in report["scenarios"]:
        name = sc.get("scenario", "?")
        auc = sc.get("acfl_eval", {}).get("trust_score_auc")
        rounds = sc.get("final_round")
        if rounds is None:
            rounds = sc.get("acfl_eval", {}).get("acfl_rounds")
        print(f"  {name:40s} rounds={rounds}  trust_auc={auc}")
    dc = report.get("defense_comparison", {})
    if dc:
        fa = dc.get("fedavg_poisoned", {}).get("fl_summary", {})
        ac = dc.get("acfl_gradient_scaling", {})
        print(f"\n  Defense comparison (gradient_scaling attack):")
        print(f"    FedAvg  mean_acc={fa.get('mean_final_accuracy')}  mean_f1={fa.get('mean_final_f1')}")
        acfl_fl = ac.get("fl_summary", {})
        print(f"    ACFL    mean_acc={acfl_fl.get('mean_final_accuracy')}  mean_f1={acfl_fl.get('mean_final_f1')}  trust_auc={ac.get('trust_auc')}")
    if report.get("layer3_eval"):
        l3 = report["layer3_eval"]
        print(f"\n  Layer 3 aggregate_auroc={l3.get('aggregate_auroc')}")
    if report.get("pytest"):
        print(f"  pytest: {'PASS' if report['pytest']['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
