"""
Fix: feature scaling for GRU input.
Diagnosis: implied_travel_speed_kmh, session_duration_zscore, geo_distance_from_prev_km
have extreme outliers (std in the thousands, max in the millions for one feature)
compared to flag features like new_device_flag (0-1). Unscaled, these dominate the
GRU's gate pre-activations, saturating sigmoid/tanh and starving gradients to the
smaller-scale-but-important features. Fixed with RobustScaler (median/IQR — resistant
to the outliers themselves, unlike mean/std scaling) fit on TRAIN only, then clipped
to [-5, 5] to bound any remaining extreme values.

NOTE: OUT_DIR changed from the original hardcoded /mnt/user-data/outputs/detection
sandbox path to "." (current folder) for local use — run this from
outputs/detection/, same place as seq_prep.py's output.
"""
import numpy as np
from sklearn.preprocessing import RobustScaler

OUT_DIR = "."
FEATURE_NAMES = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before", "hour_sin", "hour_cos",
]

def fit_scaler(X_train, mask_train):
    """Fit RobustScaler using only real (non-padded) train timesteps."""
    n_feat = X_train.shape[2]
    flat_X = X_train.reshape(-1, n_feat)
    flat_mask = mask_train.reshape(-1)
    real_rows = flat_X[flat_mask == 1]
    scaler = RobustScaler().fit(real_rows)
    return scaler

def apply_scaler(X, scaler, clip=5.0):
    shape = X.shape
    flat = X.reshape(-1, shape[2])
    scaled = scaler.transform(flat)
    scaled = np.clip(scaled, -clip, clip)
    return scaled.reshape(shape).astype(np.float32)

if __name__ == "__main__":
    print("Loading raw sequences...")
    tr = np.load(f"{OUT_DIR}/train_sequences.npz", allow_pickle=True)
    va = np.load(f"{OUT_DIR}/val_sequences.npz", allow_pickle=True)
    te = np.load(f"{OUT_DIR}/test_sequences.npz", allow_pickle=True)

    print("Fitting RobustScaler on train (real timesteps only)...")
    scaler = fit_scaler(tr["X"], tr["mask"])

    for name, d in [("train", tr), ("val", va), ("test", te)]:
        X_scaled = apply_scaler(d["X"], scaler)
        np.savez_compressed(
            f"{OUT_DIR}/{name}_sequences_scaled.npz",
            X=X_scaled, y=d["y"], y_type=d["y_type"], mask=d["mask"],
            entity_ids=d["entity_ids"], timestamps=d["timestamps"],
        )
        print(f"  {name}: scaled, mean_abs={np.abs(X_scaled).mean():.3f}, "
              f"max_abs={np.abs(X_scaled).max():.3f}")

    import joblib
    joblib.dump(scaler, f"{OUT_DIR}/gru_feature_scaler.joblib")
    print(f"\nSaved scaled sequences + scaler to {OUT_DIR}/")