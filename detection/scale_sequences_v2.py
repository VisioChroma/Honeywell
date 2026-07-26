"""
Scale v2 sequences — same RobustScaler approach as Version A.
Not optional: unscaled features caused Version A's PR-AUC to drop from 0.91 to 0.79.
"""
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

OUT_DIR = "."

def fit_scaler(X_train, mask_train):
    n_feat = X_train.shape[2]
    flat_X = X_train.reshape(-1, n_feat)
    flat_mask = mask_train.reshape(-1)
    real_rows = flat_X[flat_mask == 1]
    return RobustScaler().fit(real_rows)

def apply_scaler(X, scaler, clip=5.0):
    shape = X.shape
    flat = X.reshape(-1, shape[2])
    scaled = scaler.transform(flat)
    scaled = np.clip(scaled, -clip, clip)
    return scaled.reshape(shape).astype(np.float32)

if __name__ == "__main__":
    print("Loading v2 sequences...")
    tr = np.load(f"{OUT_DIR}/train_sequences_v2.npz", allow_pickle=True)
    va = np.load(f"{OUT_DIR}/val_sequences_v2.npz", allow_pickle=True)
    te = np.load(f"{OUT_DIR}/test_sequences_v2.npz", allow_pickle=True)

    print("Fitting RobustScaler on train (real timesteps only)...")
    scaler = fit_scaler(tr["X"], tr["mask"])

    for name, d in [("train", tr), ("val", va), ("test", te)]:
        X_scaled = apply_scaler(d["X"], scaler)
        np.savez_compressed(
            f"{OUT_DIR}/{name}_sequences_v2_scaled.npz",
            X=X_scaled, y=d["y"], y_type=d["y_type"], mask=d["mask"],
            entity_ids=d["entity_ids"], timestamps=d["timestamps"],
        )
        print(f"  {name}: scaled, mean_abs={np.abs(X_scaled).mean():.3f}, "
              f"max_abs={np.abs(X_scaled).max():.3f}")

    joblib.dump(scaler, f"{OUT_DIR}/gru_v2_feature_scaler.joblib")
    print(f"\nSaved scaled v2 sequences + scaler to {OUT_DIR}/")