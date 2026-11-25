#!/usr/bin/env python3
"""Pack ACFL-owned Mission Planner demo assets (no ACM_TRUST).

Per scene × controller (fedavg / acfl):
  runs/<scene>/<controller>/ → scenes/ + mission_planner/<scene>/<controller>/

Usage (ACFL root):
  python scripts/pack_acfl_mp_demo.py
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_roots() -> tuple[Path, Path]:
    """Repo root + simulations/fl-time-series root (flat PR or nested ACFL)."""
    root = Path(__file__).resolve().parents[1]
    if (root / "fl-time-series").is_dir():
        return root, root
    nested = root / "drift-fairroad"
    if nested.is_dir():
        return root, nested
    return root, root

ACFL_ROOT, DRIFT_ROOT = _resolve_roots()
OUT = ACFL_ROOT / "results" / "demo_mission_planner"
RUNS = OUT / "runs"
SCENES = (
    "scene_c_clean",
    "scene_a_gps_spoof",
    "scene_b_waypoint_collude",
    "scene_d_fleet_collapse",
    "scene_e_reference_uav",
)
CONTROLLERS = ("fedavg", "acfl")


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def _cell(summary: dict, key: str, field: str) -> Optional[float]:
    row = summary.get(key) or {}
    if field not in row:
        return None
    return _round(row[field], 3 if "auroc" in field or "prob" in field else 2)


def extract_locked_metrics() -> Dict[str, Any]:
    journal = json.loads(
        (ACFL_ROOT / "evaluation_results/acfl_tuning/drone_journal_benchmark_summary.json").read_text()
    )
    combined = json.loads(
        (ACFL_ROOT / "evaluation_results/combined_layer3/combined_layer3_benchmark_summary.json").read_text()
    )
    return {
        "on_screen": {
            "gradient_scaling": {
                "fedavg_fl_f1": _cell(journal, "gradient_scaling/fedavg", "mean_f1"),
                "acfl_fl_f1": _cell(journal, "gradient_scaling/acfl_full", "mean_f1"),
                "fedavg_layer3_auroc": _cell(
                    combined, "gradient_scaling/fedavg", "mean_layer3_auroc"
                ),
                "acfl_layer3_auroc": _cell(
                    combined, "gradient_scaling/acfl_full", "mean_layer3_auroc"
                ),
            },
            "byzantine": {
                "fedavg_fl_f1": _cell(journal, "byzantine/fedavg", "mean_f1"),
                "acfl_fl_f1": _cell(journal, "byzantine/acfl_full", "mean_f1"),
                "fedavg_layer3_auroc": _cell(
                    combined, "byzantine/fedavg", "mean_layer3_auroc"
                ),
                "acfl_layer3_auroc": _cell(
                    combined, "byzantine/acfl_full", "mean_layer3_auroc"
                ),
            },
            "backdoor": {
                "fedavg_trigger_p": _cell(
                    journal, "backdoor/fedavg", "mean_backdoor_attack_prob"
                ),
                "krum_trigger_p": _cell(
                    journal, "backdoor/krum", "mean_backdoor_attack_prob"
                ),
                "acfl_trigger_p": _cell(
                    journal, "backdoor/acfl_full", "mean_backdoor_attack_prob"
                ),
            },
        },
        "sysid_legend": {
            "1/2/3": "uav_0/1/2 EKF (GLOBAL_POSITION_INT)",
            "11/12/13": "uav_0/1/2 SIM truth (SIM_STATE→GPI)",
        },
        "policy": {
            "separate_logs": "Each scene flown under fedavg and acfl controllers separately",
            "on_detect": "Controller safe_hold → recover",
            "on_miss": "Attack continues until break; EKF icon slides away from SIM",
        },
    }


def copy_run_assets() -> dict:
    scenes_out = OUT / "scenes"
    scenes_out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"scenes": {}, "source": "ACFL SITL runs (fedavg + acfl)"}
    for scene in SCENES:
        manifest["scenes"][scene] = {}
        for ctrl in CONTROLLERS:
            run = RUNS / scene / ctrl
            if not run.is_dir() or not (run / "run_meta.json").is_file():
                print(f"missing run {run}")
                continue
            dest = scenes_out / scene / ctrl
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            meta = json.loads((run / "run_meta.json").read_text())
            files = []
            for v in meta.get("vehicles") or []:
                uid = v["uav_id"]
                if v.get("tlog"):
                    src = run / v["tlog"]
                    shutil.copy2(src, dest / f"{uid}.tlog")
                    files.append(f"{uid}.tlog")
                if v.get("bin"):
                    src = run / v["bin"]
                    shutil.copy2(src, dest / f"{uid}.BIN")
                    files.append(f"{uid}.BIN")
            shutil.copy2(run / "run_meta.json", dest / "run_meta.json")
            manifest["scenes"][scene][ctrl] = {
                "files": files,
                "end_reason": meta.get("end_reason"),
                "recovered": meta.get("recovered"),
                "break_events": meta.get("break_events"),
                "merged_tlog": f"mission_planner/{scene}/{ctrl}/multivehicle_sync.tlog",
            }
    return manifest


def copy_caption_figures(captions: Path) -> list:
    fig_dir = ACFL_ROOT / "docs" / "manuscript" / "figures"
    copied = []
    for name in (
        "fig_main_f1.png",
        "fig_main_f1.pdf",
        "fig_backdoor_dual.png",
        "fig_backdoor_dual.pdf",
    ):
        src = fig_dir / name
        if src.is_file():
            shutil.copy2(src, captions / name)
            copied.append(name)
    return copied


def write_docs(metrics: dict, manifest: dict) -> None:
    gs = metrics["on_screen"]["gradient_scaling"]
    byz = metrics["on_screen"]["byzantine"]
    bd = metrics["on_screen"]["backdoor"]
    (OUT / "README.md").write_text(
        f"""# ACFL Mission Planner video demo pack

Separate **FedAvg** and **ACFL** controller flights per scene. Map shows
**EKF (sysid 1/2/3)** vs **SIM truth (sysid 11/12/13)** so GPS spoof misses
are visible as the EKF icon sliding away from SIM.

## Fly / pack

```bash
python3 scripts/run_acfl_mp_sitl_demo.py --scene all
python3 scripts/pack_acfl_mp_demo.py
```

## Play order in Mission Planner

1. Clean: `mission_planner/scene_c_clean/{{fedavg,acfl}}/multivehicle_sync.tlog`
2. Multi-wave GPS: `.../scene_a_gps_spoof/{{fedavg,acfl}}/...`
3. Waypoint + GPS: `.../scene_b_waypoint_collude/{{fedavg,acfl}}/...`
4. **Fleet collapse** (all-3 spoof): `.../scene_d_fleet_collapse/{{fedavg,acfl}}/...`
5. **Reference UAV** (2-of-3 spoof, clean craft stays locked): `.../scene_e_reference_uav/{{fedavg,acfl}}/...`
6. End card figures in `captions/`

### Map legend

| Sysid | Track |
|------:|-------|
| 1/2/3 | EKF `GLOBAL_POSITION_INT` |
| 11/12/13 | SIM truth (from `SIM_STATE`) |

### Live metrics + waypoints (tlog play)

Each merged tlog includes `NAMED_VALUE_FLOAT` (~1 Hz on EKF sysids):
`EKF_SIM_M`, `MISS_CNT`, `HOLD_CNT`, `DET_CNT`, `FAVG_SCR`, `ACFL_SCR`, `ATK_ON`.
In MP: Status / Quick / HUD User Items / Tuning graph.

Waypoints: load `mission.waypoints` beside each tlog via **Flight Plan → Load WP**
before playing (also embedded as `MISSION_ITEM_INT` in the tlog).

## Locked caption snapshot

- GS: FedAvg FL F1 **{gs['fedavg_fl_f1']}** → ACFL **{gs['acfl_fl_f1']}**; L3 **{gs['fedavg_layer3_auroc']}** → **{gs['acfl_layer3_auroc']}**
- Byzantine: FedAvg F1 **{byz['fedavg_fl_f1']}** / L3 **{byz['fedavg_layer3_auroc']}** → ACFL **{byz['acfl_fl_f1']}** / **{byz['acfl_layer3_auroc']}**
- Backdoor trigger P: FedAvg **{bd['fedavg_trigger_p']}** / Krum **{bd['krum_trigger_p']}** / ACFL **{bd['acfl_trigger_p']}**

See [`DESCRIPTIONS.md`](DESCRIPTIONS.md).
"""
    )
    rows = []
    for sid, ctrls in manifest.get("scenes", {}).items():
        if not isinstance(ctrls, dict):
            continue
        for ctrl, info in ctrls.items():
            rows.append(
                f"| `{sid}` | `{ctrl}` | `{info.get('end_reason')}` | "
                f"{info.get('recovered')} | {info.get('break_events')} |"
            )
    (OUT / "DESCRIPTIONS.md").write_text(
        f"""# ACFL Mission Planner demo — storyboard

Each scenario has **two** tlogs: `fedavg/` (controller=FedAvg) and `acfl/`.

## Map sysids

| Sysid | Meaning |
|------:|---------|
| 1/2/3 | EKF estimated position |
| 11/12/13 | SIM ground truth |

Under GPS spoof, FedAvg miss → EKF (1) slides off SIM (11). ACFL hold → spoof stops, EKF re-converges.

## Cuts

| Cut | Asset | What you should see |
|-----|-------|---------------------|
| 1 | `mission_planner/scene_c_clean/acfl/multivehicle_sync.tlog` | Clean 3-UAV; EKF≈SIM |
| 2a/2b | `.../scene_a_gps_spoof/{{fedavg,acfl}}/...` | Multi-wave GPS; miss→BREAK vs recover |
| 3a/3b | `.../scene_b_waypoint_collude/{{fedavg,acfl}}/...` | WP + GPS; miss vs restore |
| 4a/4b | `.../scene_d_fleet_collapse/{{fedavg,acfl}}/...` | All-3 spoof; fleet collapse vs recover-all |
| 5a/5b | `.../scene_e_reference_uav/{{fedavg,acfl}}/...` | 2-of-3 spoof; clean UAV stays EKF≈SIM |
| 6 | `captions/fig_*.png` | Locked metrics (GS L3 {gs['fedavg_layer3_auroc']}→{gs['acfl_layer3_auroc']}) |

## Last run outcomes

| Scene | Controller | end_reason | recovered | breaks |
|-------|------------|------------|-----------|--------|
{chr(10).join(rows)}

## Mission Planner

1. Open a merged tlog; enable vehicles **1–3 and 11–13**.
2. Play; compare EKF vs SIM icons during `ATTACK:gps`.
3. FedAvg log: look for `ACTIVE:FEDAVG miss` then `BREAK` / `MISSCONSEQ`.
4. ACFL log: look for `ACTIVE:ACFL HOLD` then `RESULT:ACFL recovered`.
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    captions = OUT / "captions"
    captions.mkdir(parents=True, exist_ok=True)

    if not args.skip_merge:
        subprocess.check_call(
            [sys.executable, str(ACFL_ROOT / "scripts" / "merge_mp_multivehicle_tlog.py")],
            cwd=str(ACFL_ROOT),
        )

    if not args.skip_extract:
        extract = ACFL_ROOT / "scripts" / "extract_acfl_mp_tracks.py"
        if extract.is_file():
            try:
                subprocess.check_call([sys.executable, str(extract)], cwd=str(ACFL_ROOT))
            except subprocess.CalledProcessError as exc:
                print(f"extract tracks skipped: {exc}")

    manifest = copy_run_assets()
    metrics = extract_locked_metrics()
    (captions / "locked_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    figs = copy_caption_figures(captions)
    manifest["caption_figures"] = figs
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_docs(metrics, manifest)
    print(f"wrote {OUT}")
    print(json.dumps(manifest["scenes"], indent=2))


if __name__ == "__main__":
    main()
