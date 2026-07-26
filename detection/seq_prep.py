"""
================================================================================
Deliverable 3 — Step 1: Sequence Data Preparation
================================================================================
Reshapes the event-level train/val/test CSVs into per-entity ordered windows
suitable for a sequence model (GRU). Two things this handles explicitly:

  - Causal windows: window for event t uses events [t-W+1, ..., t] only
    (never future events) — preserves the no-leakage guarantee from D1/D2.
  - Short-history entities: left-padded with zeros + a mask, so cold-start
    entities (few events) still produce valid, well-formed windows instead
    of being dropped.

Output: .npz files (X, y, mask, entity_ids, labels) for train/val/test.

NOTE: paths below are relative, assuming this script is run from
outputs/detection/ with train.csv/val.csv/test.csv one level up in outputs/.
(Original version had hardcoded /mnt/user-data/outputs/... sandbox paths —
fixed here for local use.)
================================================================================
"""
import numpy as np
import pandas as pd
import os

WINDOW_LEN = 15
OUT_DIR = "."          # write sequence .npz files into the current folder
CSV_DIR = ".."         # train.csv / val.csv / test.csv live one level up
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before", "hour_sin", "hour_cos",
]

def add_hour_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df

def build_windows(df, window_len=WINDOW_LEN):
    """
    For every event, build a causal window of the preceding `window_len` events
    (including itself), left-padded with zeros if the entity has less history.
    Returns:
      X: (n_events, window_len, n_features)
      y: (n_events,) binary is_anomaly label for the LAST event in each window
      y_type: (n_events,) attack-type label for the LAST event in each window
      mask: (n_events, window_len) 1 where real data, 0 where padding
      entity_ids, timestamps for traceability
    """
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    n_features = len(FEATURE_COLS)

    X_list, y_list, ytype_list, mask_list, eid_list, ts_list = [], [], [], [], [], []

    for eid, grp in df.groupby("entity_id", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        feats = grp[FEATURE_COLS].fillna(0).values.astype(np.float32)
        n = len(grp)

        for t in range(n):
            start = max(0, t - window_len + 1)
            window = feats[start:t + 1]
            pad_len = window_len - len(window)
            if pad_len > 0:
                pad = np.zeros((pad_len, n_features), dtype=np.float32)
                window = np.vstack([pad, window])
                m = np.concatenate([np.zeros(pad_len), np.ones(len(feats[start:t + 1]))])
            else:
                m = np.ones(window_len)

            X_list.append(window)
            y_list.append(grp["is_anomaly"].iloc[t])
            ytype_list.append(grp["label"].iloc[t])
            mask_list.append(m)
            eid_list.append(eid)
            ts_list.append(grp["timestamp"].iloc[t])

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    y_type = np.array(ytype_list)
    mask = np.stack(mask_list).astype(np.float32)
    entity_ids = np.array(eid_list)
    timestamps = np.array(ts_list)
    return X, y, y_type, mask, entity_ids, timestamps

def process_split(name):
    df = pd.read_csv(f"{CSV_DIR}/{name}.csv")
    df = add_hour_features(df)
    X, y, y_type, mask, entity_ids, timestamps = build_windows(df)
    np.savez_compressed(
        f"{OUT_DIR}/{name}_sequences.npz",
        X=X, y=y, y_type=y_type, mask=mask, entity_ids=entity_ids,
        timestamps=timestamps.astype(str),
    )
    print(f"{name}: X={X.shape}, y anomaly rate={y.mean():.3%}, "
          f"entities={len(np.unique(entity_ids))}")
    return X, y, y_type, mask, entity_ids

if __name__ == "__main__":
    print(f"Feature columns ({len(FEATURE_COLS)}):", FEATURE_COLS)
    print(f"Window length: {WINDOW_LEN}\n")
    for split in ["train", "val", "test"]:
        process_split(split)
    print(f"\nSaved sequence arrays to {OUT_DIR}/")