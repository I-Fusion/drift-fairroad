"""
Agent-centric FL data layout: each agent may own many missions.

Splits are applied **per mission** (never by holding out whole agents):
  - Multi-mission agents: last N missions are eval-only (all windows); earlier
    missions contribute train windows (temporal prefix).
  - Single-mission agents: post-attack split (train = pre-attack, eval = attack onward).

Layout (canonical):
    packets/agent_000/mission_drone_000.csv
    packets/agent_000/mission_drone_010.csv
    ...

Legacy (one mission per agent):
    packets/packet_client_000.csv
"""
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from data_preprocessing_time_series import (
    DataPreprocessorTimeSeries,
    oversample_attack_windows,
    split_post_attack_windows,
    split_temporal_windows,
)


def list_agent_ids(packets_dir: str) -> List[int]:
    """Return sorted 0-based agent indices present under packets_dir."""
    ids = set()
    for path in glob.glob(os.path.join(packets_dir, "agent_*")):
        m = re.search(r"agent_(\d+)$", os.path.basename(path.rstrip("/")))
        if m:
            ids.add(int(m.group(1)))
    for path in glob.glob(os.path.join(packets_dir, "packet_client_*.csv")):
        m = re.search(r"packet_client_(\d+)", os.path.basename(path))
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def list_mission_files_for_agent(packets_dir: str, agent_num: int) -> List[str]:
    """All packet mission CSVs for one agent, sorted by mission name."""
    agent_dir = os.path.join(packets_dir, f"agent_{agent_num:03d}")
    missions = sorted(glob.glob(os.path.join(agent_dir, "mission_*.csv")))
    if missions:
        return missions
    legacy = os.path.join(packets_dir, f"packet_client_{agent_num:03d}.csv")
    if os.path.isfile(legacy):
        return [legacy]
    return []


def count_agents(packets_dir: str) -> int:
    ids = list_agent_ids(packets_dir)
    return len(ids)


def _partition_missions(
    missions: List[str],
    config,
) -> Tuple[List[str], List[str], str]:
    """
    Return (train_missions, eval_missions, eval_mode).

    eval_mode is one of: heldout_mission | post_attack | temporal_suffix
    """
    n_held = int(getattr(config, "EVAL_HELDOUT_MISSIONS", 1))
    if len(missions) > n_held > 0:
        return missions[:-n_held], missions[-n_held:], "heldout_mission"
    if len(missions) == 1:
        return missions, missions, "post_attack"
    return missions, [], "temporal_suffix"


def _mission_windows(
    packet_file: str,
    config,
    *,
    partition: str,
    eval_mode: str,
    norm_means: Optional[np.ndarray] = None,
    norm_stds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    """Build train or eval windows for one mission under the agent split policy."""
    pre = DataPreprocessorTimeSeries(
        packet_file=packet_file,
        packet_features=getattr(config, "PACKET_FEATURES", None),
        packet_timestamp_col=getattr(config, "PACKET_TIMESTAMP_COL", "Timestamp"),
        label_col=getattr(config, "LABEL_COL", "Label"),
        use_labels=True,
    )
    test_split = float(getattr(config, "TEST_SPLIT", 0.2))
    use_temporal = bool(getattr(config, "TEMPORAL_TRAIN_SPLIT", False))

    if norm_means is not None and norm_stds is not None:
        pre.set_normalization_params(norm_means, norm_stds)
        norm_frac = None
    elif eval_mode == "heldout_mission" and partition == "eval":
        # Eval mission uses stats from train missions (passed in) or full-mission fit
        norm_frac = None
    elif use_temporal:
        norm_frac = (1.0 - test_split) if partition == "train" else None
    else:
        norm_frac = None

    X, y, num_features = pre.preprocess(
        config.WINDOW_SIZE,
        config.OVERLAP,
        norm_train_fraction=norm_frac,
    )
    y_flat = np.asarray(y).reshape(-1)
    stats = {
        "mission_file": packet_file,
        "windows_total": len(y_flat),
        "eval_mode": eval_mode,
        "partition": partition,
    }

    if eval_mode == "heldout_mission":
        if partition == "train":
            if use_temporal:
                X_out, y_out, _, _, split_stats = split_temporal_windows(
                    X, y_flat, test_split=test_split,
                )
                stats.update(split_stats)
            else:
                X_out, y_out = X, y_flat
                stats["split_mode"] = "full_mission_train"
        else:
            X_out, y_out = X, y_flat
            stats["split_mode"] = "full_mission_eval"
            stats["test_attack_ratio"] = float((y_out >= 0.5).mean())
        return X_out, y_out, num_features, stats

    if eval_mode == "post_attack":
        X_tr, y_tr, X_te, y_te, split_stats = split_post_attack_windows(
            X, y_flat, fallback_test_split=test_split,
        )
        stats.update(split_stats)
        if partition == "train":
            return X_tr, y_tr, num_features, stats
        return X_te, y_te, num_features, stats

    # temporal_suffix (multi-mission, no held-out mission)
    if use_temporal:
        X_tr, y_tr, X_te, y_te, split_stats = split_temporal_windows(
            X, y_flat, test_split=test_split,
        )
        stats.update(split_stats)
        if partition == "train":
            return X_tr, y_tr, num_features, stats
        return X_te, y_te, num_features, stats

    if partition == "train":
        return X, y_flat, num_features, stats
    return np.array([]), np.array([]), num_features, stats


def _fit_agent_norm_from_train_missions(
    train_missions: List[str],
    config,
    eval_mode: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit normalization on train-mission packet rows (train portions only)."""
    if not train_missions:
        return None, None

    chunks = []
    test_split = float(getattr(config, "TEST_SPLIT", 0.2))
    use_temporal = bool(getattr(config, "TEMPORAL_TRAIN_SPLIT", False))
    label_col = getattr(config, "LABEL_COL", "Label")

    for path in train_missions:
        pre = DataPreprocessorTimeSeries(
            packet_file=path,
            packet_features=getattr(config, "PACKET_FEATURES", None),
            packet_timestamp_col=getattr(config, "PACKET_TIMESTAMP_COL", "Timestamp"),
            label_col=label_col,
            use_labels=True,
        )
        merged_df, labels = pre.merge_data()
        feature_cols = merged_df.attrs.get("feature_cols") or [
            c for c in merged_df.columns if c != pre.timestamp_col
        ]
        data = merged_df[feature_cols].values

        if eval_mode == "post_attack" and labels is not None:
            labels_arr = np.asarray(labels).reshape(-1)
            pos = np.flatnonzero(labels_arr >= 0.5)
            if len(pos) > 0:
                # Window k starts at packet k * stride; use first attack packet as cutoff
                fit_rows = max(1, int(pos[0]))
            elif use_temporal:
                fit_rows = max(1, int(len(data) * (1.0 - test_split)))
            else:
                fit_rows = len(data)
            chunks.append(data[:fit_rows])
        elif use_temporal:
            fit_rows = max(1, int(len(data) * (1.0 - test_split)))
            chunks.append(data[:fit_rows])
        else:
            chunks.append(data)

    if not chunks:
        return None, None

    combined = np.vstack(chunks)
    means = np.mean(combined, axis=0)
    stds = np.std(combined, axis=0)
    stds[stds == 0] = 1.0
    return means, stds


def build_agent_train_windows(
    packets_dir: str,
    agent_num: int,
    config,
    *,
    oversample: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int, List[dict]]:
    """Concatenate per-mission **train** windows for one agent."""
    missions = list_mission_files_for_agent(packets_dir, agent_num)
    if not missions:
        raise ValueError(f"agent_{agent_num:03d}: no missions under {packets_dir}")

    train_missions, eval_missions, eval_mode = _partition_missions(missions, config)
    norm_means, norm_stds = _fit_agent_norm_from_train_missions(
        train_missions, config, eval_mode,
    )

    mission_stats: List[dict] = []
    xs, ys = [], []
    num_features = None

    for path in train_missions:
        X, y, nf, st = _mission_windows(
            path,
            config,
            partition="train",
            eval_mode=eval_mode,
            norm_means=norm_means,
            norm_stds=norm_stds,
        )
        st["mission_index"] = len(mission_stats)
        mission_stats.append(st)
        if len(X) == 0:
            continue
        xs.append(X)
        ys.append(np.asarray(y).reshape(-1))
        num_features = nf

    if not xs:
        raise ValueError(f"agent_{agent_num:03d}: no training windows from {packets_dir}")

    X_out = np.concatenate(xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    oversample_stats = None
    if oversample and getattr(config, "OVERSAMPLE_ATTACK_WINDOWS", False):
        X_out, y_out, oversample_stats = oversample_attack_windows(
            X_out,
            y_out,
            target_attack_ratio=getattr(config, "TARGET_ATTACK_WINDOW_RATIO", 0.25),
            seed=seed + agent_num,
        )
    if oversample_stats:
        mission_stats.append({"oversample": oversample_stats})
    mission_stats.append({
        "eval_mode": eval_mode,
        "train_missions": len(train_missions),
        "eval_missions": len(eval_missions),
    })
    return X_out, y_out, int(num_features), mission_stats


def build_agent_eval_windows(
    packets_dir: str,
    agent_num: int,
    config,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Concatenate per-mission **eval** windows for one agent (includes attack labels)."""
    missions = list_mission_files_for_agent(packets_dir, agent_num)
    if not missions:
        return np.array([]), np.array([]), []

    train_missions, eval_missions, eval_mode = _partition_missions(missions, config)
    norm_means, norm_stds = _fit_agent_norm_from_train_missions(
        train_missions, config, eval_mode,
    )

    mission_stats: List[dict] = []
    xs, ys = [], []

    if eval_mode == "heldout_mission":
        eval_paths = eval_missions
    elif eval_mode == "post_attack":
        eval_paths = train_missions
    else:
        eval_paths = train_missions

    for path in eval_paths:
        X, y, _, st = _mission_windows(
            path,
            config,
            partition="eval",
            eval_mode=eval_mode,
            norm_means=norm_means,
            norm_stds=norm_stds,
        )
        st["mission_index"] = len(mission_stats)
        mission_stats.append(st)
        if len(X) == 0:
            continue
        xs.append(X)
        ys.append(np.asarray(y).reshape(-1))

    if not xs:
        return np.array([]), np.array([]), mission_stats
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), mission_stats
