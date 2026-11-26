#!/usr/bin/env python3
"""Print estimated onboard resource use for a classification FL config."""
import importlib
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
FL_TS = os.path.join(_ROOT, "fl-time-series")
if not os.path.isdir(FL_TS):
    FL_TS = os.path.join(_ROOT, "drift-fairroad", "fl-time-series")
sys.path.insert(0, FL_TS)
sys.path.insert(0, _ROOT)


def _load_config(name: str):
    return importlib.import_module(name)


def _build_model(cfg):
    mod = importlib.import_module(cfg.MODEL_PATH)
    cls = getattr(mod, cfg.MODEL_CLASS)
    mc = dict(cfg.MODEL_CONFIG)
    return cls(**mc)


def estimate(config_name: str = "config_classification_drone") -> dict:
    cfg = _load_config(config_name)
    model = _build_model(cfg)
    n_params = model.count_parameters()

    from fl_comm import estimate_state_dict_bytes

    sd = model.state_dict()
    fp32_b = estimate_state_dict_bytes(sd, fp16=False)
    fp16_b = estimate_state_dict_bytes(sd, fp16=True)

    batch = int(getattr(cfg, "BATCH_SIZE", 32))
    win = int(getattr(cfg, "WINDOW_SIZE", 50))
    feat = int(cfg.INPUT_SIZE)
    # Rough activation memory for one forward pass (bytes)
    act_bytes = batch * win * feat * 4 * 8

    rounds = int(getattr(cfg, "DEFAULT_FL_ROUNDS", 5))
    wpr = int(getattr(cfg, "WINDOWS_PER_ROUND", 10))
    clients = int(getattr(cfg, "NUM_CLIENTS", 3))

    return {
        "config": config_name,
        "model": f"{cfg.MODEL_PATH}.{cfg.MODEL_CLASS}",
        "parameters": n_params,
        "weight_upload_fp32_kb": round(fp32_b / 1024, 1),
        "weight_upload_fp16_kb": round(fp16_b / 1024, 1),
        "batch_size": batch,
        "windows_per_round": wpr,
        "local_epochs": int(getattr(cfg, "LOCAL_EPOCHS", 1)),
        "fl_rounds_default": rounds,
        "fl_participants": clients,
        "est_peak_ram_mb": round((fp32_b * 3 + act_bytes) / (1024 * 1024), 1),
        "checkpoint_keep_last": int(getattr(cfg, "SERVER_CHECKPOINT_KEEP_LAST", 0)),
        "save_client_checkpoints": bool(getattr(cfg, "SAVE_CLIENT_CHECKPOINTS", True)),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Estimate drone FL resource budget")
    ap.add_argument("--config", default="config_classification_drone")
    ap.add_argument("--compare", default="config_classification",
                    help="Baseline config to compare against")
    args = ap.parse_args()

    drone = estimate(args.config)
    base = estimate(args.compare)

    print("=" * 60)
    print("DRONE FL RESOURCE ESTIMATE")
    print("=" * 60)
    for k, v in drone.items():
        print(f"  {k}: {v}")
    print("\nComparison vs baseline:")
    print(f"  params: {base['parameters']:,} -> {drone['parameters']:,} "
          f"({100 * drone['parameters'] / max(base['parameters'], 1):.0f}%)")
    print(f"  upload (fp16): {base['weight_upload_fp16_kb']} KB -> "
          f"{drone['weight_upload_fp16_kb']} KB")
    print(f"  est peak RAM: {base['est_peak_ram_mb']} MB -> {drone['est_peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
