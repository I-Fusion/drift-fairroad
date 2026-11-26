#!/usr/bin/env python3
"""
Compute detailed SITL packet-dataset statistics for the journal.

Reads agent/mission CSVs, FL window splits (agent_mission_data), and SITL
collection manifests. Writes JSON + optional LaTeX fragments.

Usage:
    ./acfl_env/bin/python scripts/compute_dataset_statistics.py
    ./acfl_env/bin/python scripts/compute_dataset_statistics.py --latex-out docs/manuscript/dataset_stats.tex
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(ROOT, "fl-time-series")):
    DRIFT_ROOT = ROOT
else:
    DRIFT_ROOT = os.path.join(ROOT, "drift-fairroad")
sys.path.insert(0, os.path.join(DRIFT_ROOT, "fl-time-series"))
sys.path.insert(0, ROOT)

from acfl.layer3_manifest import load_network_attack_types  # noqa: E402
from agent_mission_data import (  # noqa: E402
    build_agent_eval_windows,
    build_agent_train_windows,
    list_agent_ids,
    list_mission_files_for_agent,
)

PACKETS_DIR = os.path.join(DRIFT_ROOT, "data", "sitl_dataset", "fl", "packets")
DEFAULT_JSON_OUT = os.path.join(
    ROOT, "evaluation_results", "dataset_statistics.json"
)


def _import_config():
    import config_classification_drone as cfg

    return cfg


def _load_manifest_entries() -> Dict[str, dict]:
    """Merge drone metadata from all SITL collection manifests."""
    drift = os.path.join(DRIFT_ROOT, "data")
    entries: Dict[str, dict] = {}
    for rel in (
        os.path.join("sitl_dataset", "manifest.json"),
        os.path.join("sitl_validate", "validate_manifest.json"),
        os.path.join("sitl_drone_extra", "drone_extra_manifest.json"),
    ):
        path = os.path.join(drift, rel)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            payload = json.load(f)
        for entry in payload.get("drones", []):
            drone = entry.get("drone")
            if drone:
                entries[drone] = entry
    return entries


def _load_run_meta(run_dir: str) -> Optional[dict]:
    path = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mission_packet_stats(path: str) -> dict:
    df = pd.read_csv(path)
    label_col = "Label"
    n = len(df)
    attack = int((df[label_col] >= 0.5).sum()) if label_col in df.columns else 0
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    duration_s = None
    pps = None
    if ts.notna().sum() >= 2:
        duration_s = float((ts.max() - ts.min()).total_seconds())
        if duration_s > 0:
            pps = n / duration_s

    feat_stats = {}
    for col in ("SrcPort", "DstPort", "Length", "MsgID", "Protocol"):
        if col in df.columns:
            feat_stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=0)),
            }

    return {
        "mission_file": path,
        "mission_basename": os.path.basename(path),
        "packets": n,
        "attack_packets": attack,
        "benign_packets": n - attack,
        "attack_packet_ratio": attack / n if n else 0.0,
        "duration_s": duration_s,
        "packets_per_second": pps,
        "feature_stats": feat_stats,
    }


def _attack_window_stats(run_dir: str) -> dict:
    meta = _load_run_meta(run_dir)
    if not meta:
        return {"attack_windows": 0, "attack_duration_s": None}
    windows = meta.get("attack_windows") or []
    total_dur = 0.0
    for w in windows:
        start = w.get("start")
        end = w.get("end")
        if start is not None and end is not None:
            total_dur += float(end) - float(start)
    return {
        "attack_windows": len(windows),
        "attack_duration_s": total_dur if windows else None,
        "attack_types_in_mission": sorted({w.get("type") for w in windows if w.get("type")}),
    }


def _summarize(values: List[float]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _window_label_stats(y: np.ndarray) -> dict:
    y = np.asarray(y).reshape(-1)
    if len(y) == 0:
        return {"windows": 0, "attack_windows": 0, "attack_window_ratio": 0.0}
    attack = int((y >= 0.5).sum())
    return {
        "windows": int(len(y)),
        "attack_windows": attack,
        "benign_windows": int(len(y) - attack),
        "attack_window_ratio": float(attack / len(y)),
    }


def compute_statistics(
    packets_dir: str = PACKETS_DIR,
    *,
    seed: int = 42,
    include_oversample: bool = True,
) -> dict:
    cfg = _import_config()
    manifest_entries = _load_manifest_entries()
    attack_types = load_network_attack_types(ROOT)

    per_mission: List[dict] = []
    for agent in list_agent_ids(packets_dir):
        for mission_path in list_mission_files_for_agent(packets_dir, agent):
            m = re.search(r"mission_drone_(\d+)", os.path.basename(mission_path))
            drone = f"drone_{m.group(1)}" if m else None
            meta = manifest_entries.get(drone, {}) if drone else {}
            pkt = _mission_packet_stats(mission_path)
            run_dir = meta.get("run_dir")
            aw = _attack_window_stats(run_dir) if run_dir else {}
            per_mission.append({
                "agent": f"agent_{agent:03d}",
                "drone": drone,
                "network_attack": attack_types.get(os.path.basename(mission_path), "unknown"),
                "wind_m_s": meta.get("wind"),
                "mission_waypoints": os.path.basename(meta.get("mission", "")) or None,
                "collection_source": _collection_source(drone),
                **pkt,
                **aw,
            })

    per_mission.sort(key=lambda x: (x.get("drone") or "", x.get("mission_basename") or ""))

    total_packets = sum(m["packets"] for m in per_mission)
    total_attack_packets = sum(m["attack_packets"] for m in per_mission)

    per_agent_fl: List[dict] = []
    train_windows_raw = 0
    train_attack_raw = 0
    train_windows_bal = 0
    train_attack_bal = 0
    eval_windows = 0
    eval_attack = 0
    bal_ratios_with_attack: List[float] = []

    logging.getLogger("data_preprocessing_time_series").setLevel(logging.WARNING)

    for agent in list_agent_ids(packets_dir):
        X_tr, y_tr, _, tr_stats = build_agent_train_windows(
            packets_dir, agent, cfg, oversample=False, seed=seed,
        )
        raw = _window_label_stats(y_tr)
        train_windows_raw += raw["windows"]
        train_attack_raw += raw["attack_windows"]

        if include_oversample:
            X_bal, y_bal, _, bal_stats = build_agent_train_windows(
                packets_dir, agent, cfg, oversample=True, seed=seed,
            )
            bal = _window_label_stats(y_bal)
            oversample_info = next(
                (s.get("oversample") for s in bal_stats if "oversample" in s),
                None,
            )
        else:
            bal = raw
            oversample_info = None

        train_windows_bal += bal["windows"]
        train_attack_bal += bal["attack_windows"]
        if raw["attack_windows"] > 0:
            bal_ratios_with_attack.append(bal["attack_window_ratio"])

        X_ev, y_ev, ev_stats = build_agent_eval_windows(packets_dir, agent, cfg)
        ev = _window_label_stats(y_ev)
        eval_windows += ev["windows"]
        eval_attack += ev["attack_windows"]

        missions = list_mission_files_for_agent(packets_dir, agent)
        heldout = os.path.basename(missions[-1]) if missions else None
        per_agent_fl.append({
            "agent": f"agent_{agent:03d}",
            "missions": len(missions),
            "heldout_eval_mission": heldout,
            "heldout_network_attack": attack_types.get(heldout, "unknown") if heldout else None,
            "train_windows_raw": raw,
            "train_windows_balanced": bal,
            "oversample": oversample_info,
            "eval_windows": ev,
        })

    attack_counts = {}
    wind_counts = {}
    source_counts = {}
    for m in per_mission:
        attack_counts[m["network_attack"]] = attack_counts.get(m["network_attack"], 0) + 1
        w = m.get("wind_m_s")
        if w is not None:
            wind_counts[str(w)] = wind_counts.get(str(w), 0) + 1
        src = m.get("collection_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    collection_args = _first_collection_args(manifest_entries)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "packets_dir": packets_dir,
        "windowing": {
            "window_size": cfg.WINDOW_SIZE,
            "overlap": cfg.OVERLAP,
            "stride": cfg.WINDOW_SIZE - cfg.OVERLAP,
            "temporal_train_split": cfg.TEMPORAL_TRAIN_SPLIT,
            "test_split": cfg.TEST_SPLIT,
            "eval_heldout_missions": cfg.EVAL_HELDOUT_MISSIONS,
            "oversample_attack_windows": cfg.OVERSAMPLE_ATTACK_WINDOWS,
            "target_attack_window_ratio": cfg.TARGET_ATTACK_WINDOW_RATIO,
        },
        "collection": collection_args,
        "topology": {
            "agents": len(list_agent_ids(packets_dir)),
            "missions": len(per_mission),
            "missions_per_agent": len(per_mission) // max(1, len(list_agent_ids(packets_dir))),
            "collection_sources": source_counts,
        },
        "network_attacks": attack_counts,
        "wind_m_s": wind_counts,
        "packet_level": {
            "total_packets": total_packets,
            "attack_packets": total_attack_packets,
            "benign_packets": total_packets - total_attack_packets,
            "attack_packet_ratio": total_attack_packets / total_packets if total_packets else 0.0,
            "packets_per_mission": _summarize([m["packets"] for m in per_mission]),
            "attack_ratio_per_mission": _summarize(
                [m["attack_packet_ratio"] for m in per_mission if m["attack_packets"] > 0]
            ),
            "duration_s_per_mission": _summarize(
                [m["duration_s"] for m in per_mission if m.get("duration_s")]
            ),
            "packets_per_second": _summarize(
                [m["packets_per_second"] for m in per_mission if m.get("packets_per_second")]
            ),
        },
        "window_level": {
            "train_raw": {
                "windows": train_windows_raw,
                "attack_windows": train_attack_raw,
                "attack_window_ratio": train_attack_raw / train_windows_raw if train_windows_raw else 0.0,
            },
            "train_balanced": {
                "windows": train_windows_bal,
                "attack_windows": train_attack_bal,
                "attack_window_ratio": train_attack_bal / train_windows_bal if train_windows_bal else 0.0,
                "per_agent_target_ratio": cfg.TARGET_ATTACK_WINDOW_RATIO,
                "agents_with_attack_train_missions": len(bal_ratios_with_attack),
                "mean_ratio_among_those_agents": (
                    float(np.mean(bal_ratios_with_attack)) if bal_ratios_with_attack else 0.0
                ),
            },
            "eval_heldout": {
                "windows": eval_windows,
                "attack_windows": eval_attack,
                "attack_window_ratio": eval_attack / eval_windows if eval_windows else 0.0,
            },
            "eval_by_network_attack": _eval_by_attack_type(per_agent_fl),
        },
        "per_mission": per_mission,
        "per_agent_fl": per_agent_fl,
        "features": ["Timestamp", "SrcPort", "DstPort", "Length", "MsgID", "Protocol", "Label"],
        "model_features": list(cfg.PACKET_FEATURES),
    }


def _collection_source(drone: Optional[str]) -> str:
    if not drone:
        return "unknown"
    idx = int(drone.split("_")[1])
    if idx < 10:
        return "sitl_dataset"
    if idx < 14:
        return "sitl_validate"
    return "sitl_drone_extra"


def _first_collection_args(manifest_entries: Dict[str, dict]) -> dict:
    drift = os.path.join(DRIFT_ROOT, "data")
    for rel in (
        os.path.join("sitl_dataset", "manifest.json"),
        os.path.join("sitl_validate", "validate_manifest.json"),
        os.path.join("sitl_drone_extra", "drone_extra_manifest.json"),
    ):
        path = os.path.join(drift, rel)
        if os.path.isfile(path):
            with open(path) as f:
                payload = json.load(f)
            args = payload.get("args", {})
            if args:
                return {
                    "speedup": args.get("speedup"),
                    "attack_count_per_mission": args.get("attack_count"),
                    "gps_spoof_duration_s": args.get("gps_duration"),
                    "mission_timeout_s": args.get("mission_timeout"),
                    "winds_m_s": args.get("winds"),
                    "network_attack_types": args.get("attacks"),
                    "area_radius_m": args.get("area_radius"),
                }
    return {}


def _eval_by_attack_type(per_agent_fl: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    for row in per_agent_fl:
        atk = row.get("heldout_network_attack") or "unknown"
        ev = row["eval_windows"]
        out.setdefault(atk, {"agents": 0, "windows": 0, "attack_windows": 0})
        out[atk]["agents"] += 1
        out[atk]["windows"] += ev["windows"]
        out[atk]["attack_windows"] += ev["attack_windows"]
    for atk, v in out.items():
        v["attack_window_ratio"] = (
            v["attack_windows"] / v["windows"] if v["windows"] else 0.0
        )
    return out


def _fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}\\%"


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}").replace("{,}", ",")


def _latex_text(s: str) -> str:
    return s.replace("_", "\\_")


def render_latex(stats: dict) -> str:
    pl = stats["packet_level"]
    wl = stats["window_level"]
    topo = stats["topology"]
    coll = stats["collection"]
    wcfg = stats["windowing"]
    attacks = stats["network_attacks"]
    eval_by = wl["eval_by_network_attack"]

    def atk_count(name: str) -> int:
        return attacks.get(name, 0)

    lines = [
        "% Auto-generated by scripts/compute_dataset_statistics.py — do not edit by hand.",
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{SITL packet dataset characteristics (agent--mission layout).}",
        "\\label{tab:dataset}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}p{0.32\\textwidth}p{0.64\\textwidth}@{}}",
        "\\toprule",
        "Quantity & Value \\\\",
        "\\midrule",
        f"Collection & ArduPilot SITL, {coll.get('speedup', 15)}$\\times$ speedup; "
        f"{atk_count('waypoint')} waypoint, {atk_count('gps')} GNSS, {atk_count('none')} benign missions \\\\",
        f"Topology & {topo['agents']} agents $\\times$ {topo['missions_per_agent']} missions "
        f"({topo['missions']} total); non-IID via cyclic agent assignment \\\\",
        f"Packet rows & {_fmt_int(pl['total_packets'])} total; "
        f"{_fmt_int(pl['attack_packets'])} attack ({_fmt_pct(pl['attack_packet_ratio'])}) \\\\",
        f"Mission size & "
        f"{_fmt_int(int(pl['packets_per_mission']['min']))}--"
        f"{_fmt_int(int(pl['packets_per_mission']['max']))} packets/mission "
        f"(median {_fmt_int(int(pl['packets_per_mission']['median']))}) \\\\",
        f"Flight duration & "
        f"{pl['duration_s_per_mission']['min']:.0f}--{pl['duration_s_per_mission']['max']:.0f}\\,s/mission "
        f"(median {pl['duration_s_per_mission']['median']:.0f}\\,s) \\\\",
        f"Windowing & size {wcfg['window_size']}, stride {wcfg['stride']}; "
        f"temporal {int(100 * (1 - wcfg['test_split']))}/{int(100 * wcfg['test_split'])} train split on train missions \\\\",
        f"FL train windows & {_fmt_int(wl['train_raw']['windows'])} raw "
        f"({_fmt_pct(wl['train_raw']['attack_window_ratio'])} attack); "
        f"{_fmt_int(wl['train_balanced']['windows'])} after oversampling "
        f"({_fmt_pct(wl['train_balanced']['mean_ratio_among_those_agents'])} per agent with attacks) \\\\",
        f"Held-out eval & 1 mission/agent; {_fmt_int(wl['eval_heldout']['windows'])} windows "
        f"({_fmt_pct(wl['eval_heldout']['attack_window_ratio'])} attack) \\\\",
    ]

    for atk in ("waypoint", "gps", "none"):
        if atk in eval_by:
            e = eval_by[atk]
            lines.append(
                f"Eval ({atk}) & {e['agents']} agents, {_fmt_int(e['windows'])} windows, "
                f"{_fmt_pct(e['attack_window_ratio'])} attack \\\\"
            )

    lines.extend([
        f"Features & {', '.join(stats['model_features'])} (binary \\texttt{{Label}}) \\\\",
        f"Injection policy & {coll.get('attack_count_per_mission', 5)} network injections/mission; "
        f"GNSS spoof {coll.get('gps_spoof_duration_s', 25)}\\,s; winds "
        f"\\{{{', '.join(str(int(w)) if float(w).is_integer() else w for w in (coll.get('winds_m_s') or []))}\\}}\\,m/s \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-mission packet statistics (held-out eval missions in bold).}",
        "\\label{tab:dataset-missions}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Agent & Mission & Packets & Attack & Ratio & Duration (s) \\\\",
        "\\midrule",
    ])

    heldout_basenames = {
        row["heldout_eval_mission"] for row in stats["per_agent_fl"] if row.get("heldout_eval_mission")
    }
    for m in stats["per_mission"]:
        base = m["mission_basename"]
        bold_open = "\\textbf{" if base in heldout_basenames else ""
        bold_close = "}" if base in heldout_basenames else ""
        dur = m.get("duration_s")
        dur_s = f"{dur:.0f}" if dur is not None else "---"
        drone_short = (m.get("drone") or "").replace("drone_", "d")
        lines.append(
            f"{_latex_text(m['agent'])} & {bold_open}{drone_short} ({m['network_attack']}){bold_close} & "
            f"{m['packets']} & {m['attack_packets']} & {_fmt_pct(m['attack_packet_ratio'])} & {dur_s} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-agent federated learning window splits. Held-out eval mission shown in parentheses.}",
        "\\label{tab:dataset-agents}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Agent & Train raw & Train bal. & Eval & Eval atk. & Held-out & Type \\\\",
        "\\midrule",
    ])

    for row in stats["per_agent_fl"]:
        tr = row["train_windows_raw"]
        tb = row["train_windows_balanced"]
        ev = row["eval_windows"]
        held = row.get("heldout_eval_mission", "")
        drone_short = ""
        if held:
            m = re.search(r"mission_drone_(\d+)", held)
            if m:
                drone_short = f"d{m.group(1)}"
        atk = row.get("heldout_network_attack", "---")
        lines.append(
            f"{_latex_text(row['agent'])} & {tr['windows']} ({_fmt_pct(tr['attack_window_ratio'])}) & "
            f"{tb['windows']} ({_fmt_pct(tb['attack_window_ratio'])}) & "
            f"{ev['windows']} & {_fmt_pct(ev['attack_window_ratio'])} & "
            f"({drone_short}) & {atk} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Compute SITL dataset statistics")
    ap.add_argument("--packets-dir", default=PACKETS_DIR)
    ap.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    ap.add_argument("--latex-out", default=None, help="Write LaTeX table fragment")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-oversample-stats", action="store_true")
    args = ap.parse_args()

    stats = compute_statistics(
        args.packets_dir,
        seed=args.seed,
        include_oversample=not args.no_oversample_stats,
    )

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {args.json_out}")

    latex_path = args.latex_out or os.path.join(
        ROOT, "docs", "manuscript", "dataset_stats.tex"
    )
    latex = render_latex(stats)
    os.makedirs(os.path.dirname(latex_path), exist_ok=True)
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Wrote {latex_path}")

    pl = stats["packet_level"]
    wl = stats["window_level"]
    print(
        f"Summary: {stats['topology']['missions']} missions, "
        f"{pl['total_packets']} packets ({pl['attack_packet_ratio']:.1%} attack), "
        f"{wl['eval_heldout']['windows']} eval windows "
        f"({wl['eval_heldout']['attack_window_ratio']:.1%} attack)"
    )


if __name__ == "__main__":
    main()
