"""
Deliverable 3 — Step 5: Sequence Prep for Version B (with long-term features)
Same causal windowing as Version A, but reads *_longterm.csv and includes the
4 new rolling 7d/30d features alongside the original 13 — 17 features total.
"""
import numpy as np
import pandas as pd
import os

WINDOW_LEN = 15
OUT_DIR = "."

FEATURE_COLS = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before", "hour_sin", "hour_cos",
    "rolling_7d_event_count", "rolling_7d_new_resource_count",
    "rolling_30d_new_resource_count", "rolling_7d_offhours_ratio",
]

def add_hour_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df

def build_windows(df, window_len=WINDOW_LEN):
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
    df = pd.read_csv(f"../{name}_longterm.csv")
    df = add_hour_features(df)
    X, y, y_type, mask, entity_ids, timestamps = build_windows(df)
    np.savez_compressed(
        f"{OUT_DIR}/{name}_sequences_v2.npz",
        X=X, y=y, y_type=y_type, mask=mask, entity_ids=entity_ids,
        timestamps=timestamps.astype(str),
    )
    print(f"{name}: X={X.shape}, y anomaly rate={y.mean():.3%}, "
          f"entities={len(np.unique(entity_ids))}")

if __name__ == "__main__":
    print(f"Feature columns ({len(FEATURE_COLS)}):", FEATURE_COLS)
    print(f"Window length: {WINDOW_LEN}\n")
    for split in ["train", "val", "test"]:
        process_split(split)
    print(f"\nSaved v2 sequence arrays to {OUT_DIR}/")