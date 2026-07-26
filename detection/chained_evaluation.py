"""
Deliverable 4 — Chained Evaluation
Evaluates the trained classifier ONLY on val/test rows the Deliverable 3
ensemble (alpha=0.10) actually flagged as alerts — the real end-to-end
number a deployment would see, as opposed to classify_attack_type.py's
decoupled evaluation (all true anomalies, regardless of detection).

Reuses the exact same alignment check pattern as ensemble_evaluate.py.
"""
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import classification_report, f1_score
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALPHA = 0.10
ALERT_BUDGET_FRAC = 0.01

FEATURE_COLS = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before", "rolling_7d_event_count",
    "rolling_7d_new_resource_count", "rolling_30d_new_resource_count",
    "rolling_7d_offhours_ratio",
]
FULL_FEATURES = FEATURE_COLS + ["entity_type_user", "entity_type_service_account",
                                  "entity_type_edge_device"]

def get_ensemble_alerts(split):
    """Reproduces Deliverable 3's locked ensemble config to get the alert mask."""
    va = np.load(f"{OUT_DIR}/{split}_sequences_scaled.npz", allow_pickle=True)
    X_a, y_a, mask_a, ytype_a = va["X"], va["y"], va["mask"], va["y_type"]
    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    y_pred_a, _ = model_a.forward(X_a, mask_a)

    db = np.load(f"{OUT_DIR}/{split}_sequences_v2_scaled.npz", allow_pickle=True)
    X_b = torch.tensor(db["X"], dtype=torch.float32)
    mask_b = torch.tensor(db["mask"], dtype=torch.float32)
    y_b_np, ytype_b = db["y"], db["y_type"]

    model_b = StatefulGRU(input_size=X_b.shape[2], hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    with torch.no_grad():
        y_pred_b = torch.sigmoid(model_b(X_b, mask_b)).numpy()

    # mandatory alignment check (same as ensemble_evaluate.py)
    if len(y_a) != len(y_b_np) or not np.array_equal(y_a, y_b_np) or not np.array_equal(ytype_a, ytype_b):
        raise SystemExit(f"ALIGNMENT CHECK FAILED on {split} — cannot proceed with chained eval.")

    y_pred = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b
    n_total = len(y_a)
    budget = max(1, int(n_total * ALERT_BUDGET_FRAC))
    threshold = np.quantile(y_pred, 1 - budget / n_total)
    alerts = y_pred >= threshold  # boolean array, length n_total

    return alerts, y_a, ytype_a

def load_and_prep_sorted(split):
    """Same sort order as seq_prep.py/seq_prep_v2.py (entity_id, then timestamp)
    so row i here corresponds to row i in the sequence .npz files."""
    df = pd.read_csv(f"../{split}_longterm.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["entity_type_user"] = (df["entity_type"] == "user").astype(int)
    df["entity_type_service_account"] = (df["entity_type"] == "service_account").astype(int)
    df["entity_type_edge_device"] = (df["entity_type"] == "edge_device").astype(int)
    return df

def evaluate_chained(clf, split):
    alerts, y_seq, ytype_seq = get_ensemble_alerts(split)
    df_sorted = load_and_prep_sorted(split)

    if len(df_sorted) != len(alerts):
        raise SystemExit(
            f"ROW COUNT MISMATCH on {split}: sorted CSV has {len(df_sorted)} rows, "
            f"sequence files have {len(alerts)}. Do not proceed — investigate "
            f"before trusting any chained numbers."
        )
    # extra safety: labels from the sequence files must match the sorted CSV's labels
    if not np.array_equal(df_sorted["label"].values, ytype_seq):
        raise SystemExit(
            f"LABEL ALIGNMENT MISMATCH on {split}: sorted CSV label order doesn't "
            f"match sequence file label order. Stopping — do not trust chained numbers."
        )
    print(f"Row + label alignment confirmed for {split}: {len(df_sorted)} rows.\n")

    flagged_true_anomalies = df_sorted[alerts & (df_sorted["is_anomaly"] == 1)]
    if len(flagged_true_anomalies) == 0:
        print(f"CHAINED eval on {split}: no true anomalies among flagged alerts.")
        return None

    X = flagged_true_anomalies[FULL_FEATURES].fillna(0).values
    y_true = flagged_true_anomalies["label"].values
    y_pred = clf.predict(X)

    print(f"{'='*60}\nCHAINED evaluation on {split} "
          f"({len(flagged_true_anomalies)} true anomalies among {alerts.sum()} flagged alerts)\n{'='*60}")
    print(classification_report(y_true, y_pred, zero_division=0))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Macro F1 (chained): {macro_f1:.4f}")
    return macro_f1

if __name__ == "__main__":
    clf = joblib.load("anomaly_classifier.joblib")
    evaluate_chained(clf, "val")
    evaluate_chained(clf, "test")