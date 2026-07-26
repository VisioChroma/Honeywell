"""
Deliverable 5/6 bridge — reusable explanation function + dashboard JSON export.

Turns the print-loop from explain_alert.py into:
  1. explain_single_alert() — a real, callable function (not a script), usable
     by anything (dashboard, report generator, future API).
  2. A batch export that runs it across the actual top alerts from the
     locked detector (ensemble alpha=0.10), not just one example per class —
     this also addresses the "only 1 spot-check per class" limitation flagged
     earlier, since it now runs across every real alert in the current budget.
  3. Writes dashboard_data.json — the exact structure the HTML dashboard
     will consume directly (embedded, no server/fetch needed).

Includes: risk score, predicted attack type, detection + classification
explanations (with the correlated-feature/timestep caveats already fixed),
cold-start flag (mask history length < 5, per cold_start_eval.py's exact
definition), and a short entity history (last 15 events' key raw features)
for the entity history view.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detection"))

import numpy as np
import pandas as pd
import torch
import joblib
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "../detection"
ALPHA = 0.10
ALERT_BUDGET_FRAC = 0.01
COLD_START_THRESHOLD = 5

DETECTION_FEATURES = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before", "hour_sin", "hour_cos",
]
DETECTION_FEATURES_B = DETECTION_FEATURES + [
    "rolling_7d_event_count", "rolling_7d_new_resource_count",
    "rolling_30d_new_resource_count", "rolling_7d_offhours_ratio",
]
CLASSIFIER_FEATURES = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before", "rolling_7d_event_count",
    "rolling_7d_new_resource_count", "rolling_30d_new_resource_count",
    "rolling_7d_offhours_ratio", "entity_type_user",
    "entity_type_service_account", "entity_type_edge_device",
]
HISTORY_FEATURES = [
    "session_duration", "failed_auth_count_5min", "geo_distance_from_prev_km",
    "resource_diversity_recent10", "new_device_flag",
]

FEATURE_PHRASES = {
    "session_duration": "unusually short/long session duration",
    "resource_access_rank": "accessing a resource rarely touched by this entity",
    "geo_distance_from_prev_km": "large geographic jump since last login",
    "implied_travel_speed_kmh": "impossible travel speed between logins",
    "new_device_flag": "new/unrecognized device fingerprint",
    "session_duration_zscore": "session duration far from this entity's norm",
    "failed_auth_count_5min": "high rate of failed logins in a short window",
    "resource_diversity_recent10": "unusually broad spread of resources accessed",
    "distinct_entities_per_source_5min": "many different accounts from the same source (credential-stuffing pattern)",
    "time_since_last_access_min": "unusual gap since this entity's last activity",
    "resource_seen_before": "accessing a resource never touched before",
    "hour_sin": "unusual time of day", "hour_cos": "unusual time of day",
    "rolling_7d_event_count": "unusual recent activity volume (7-day)",
    "rolling_7d_new_resource_count": "gradually expanding to new resources (7-day)",
    "rolling_30d_new_resource_count": "gradually expanding to new resources (30-day trend)",
    "rolling_7d_offhours_ratio": "elevated off-hours activity (7-day)",
    "entity_type_user": "entity type: user",
    "entity_type_service_account": "entity type: service account",
    "entity_type_edge_device": "entity type: edge device",
}


def get_logit_a(model, x, m):
    _, cache = model.forward(x, m)
    h_final = cache["h"][-1]
    return (h_final @ model.Wo + model.bo).ravel()[0]

def get_logit_b(model, x, m):
    with torch.no_grad():
        return model(x, m).numpy()[0]


def ablate_detection_per_timestep(model_a, model_b, x_a, m_a, x_b, m_b, top_n=3):
    orig_logit_a = get_logit_a(model_a, x_a, m_a)
    x_b_t = torch.tensor(x_b, dtype=torch.float32)
    m_b_t = torch.tensor(m_b, dtype=torch.float32)
    orig_logit_b = get_logit_b(model_b, x_b_t, m_b_t)

    best_per_feature = {}
    real_ts_a = np.where(m_a[0] == 1)[0]
    for i, fname in enumerate(DETECTION_FEATURES):
        best_drop, best_t = -np.inf, None
        for t in real_ts_a:
            x_abl = x_a.copy(); x_abl[0, t, i] = 0
            drop = ALPHA * (orig_logit_a - get_logit_a(model_a, x_abl, m_a))
            if drop > best_drop:
                best_drop, best_t = drop, t
        best_per_feature[fname] = (best_drop, best_t)

    real_ts_b = np.where(m_b[0] == 1)[0]
    for i, fname in enumerate(DETECTION_FEATURES_B):
        best_drop, best_t = -np.inf, None
        for t in real_ts_b:
            x_abl = x_b.copy(); x_abl[:, t, i] = 0
            x_abl_t = torch.tensor(x_abl, dtype=torch.float32)
            drop = (1 - ALPHA) * (orig_logit_b - get_logit_b(model_b, x_abl_t, m_b_t))
            if drop > best_drop:
                best_drop, best_t = drop, t
        if fname in best_per_feature:
            prev_drop, prev_t = best_per_feature[fname]
            best_per_feature[fname] = (prev_drop + best_drop, best_t if best_drop > prev_drop else prev_t)
        else:
            best_per_feature[fname] = (best_drop, best_t)

    supporting = [(f, d, t) for f, (d, t) in best_per_feature.items() if d > 0]
    supporting.sort(key=lambda x: -x[1])
    return supporting[:top_n]


def ablate_classification(clf, x_row, predicted_class, top_n=3):
    orig_proba = clf.predict_proba(x_row)[0]
    class_idx = list(clf.classes_).index(predicted_class)
    orig_score = orig_proba[class_idx]
    eps = 1e-6
    orig_logodds = np.log((orig_score + eps) / (1 - orig_score + eps))
    drops = []
    for i, fname in enumerate(CLASSIFIER_FEATURES):
        x_abl = x_row.copy(); x_abl[0, i] = 0
        abl_p = clf.predict_proba(x_abl)[0][class_idx]
        abl_logodds = np.log((abl_p + eps) / (1 - abl_p + eps))
        drops.append((fname, orig_logodds - abl_logodds))
    supporting = [d for d in drops if d[1] > 0]
    supporting.sort(key=lambda d: -d[1])
    return supporting[:top_n]


def phrases_for(top_features, window_len=15, with_timestep=False):
    out = []
    for item in top_features:
        f = item[0]
        phrase = FEATURE_PHRASES.get(f, f).strip()
        if with_timestep and len(item) == 3 and item[2] is not None:
            steps_ago = window_len - 1 - item[2]
            if steps_ago > 0:
                phrase += f" ({steps_ago} event(s) earlier)"
        out.append(phrase)
    return out


def explain_single_alert(idx, models, data, alert_threshold):
    """
    THE reusable function. Given a row index into the val set and the
    preloaded models/data, returns a fully-populated dict for one alert —
    used by both the batch export below and (later) any live/interactive use.
    """
    model_a, model_b, clf = models["a"], models["b"], models["clf"]
    X_a, mask_a, X_b, mask_b, df = data["X_a"], data["mask_a"], data["X_b"], data["mask_b"], data["df"]

    x_a_s, m_a_s = X_a[idx:idx+1], mask_a[idx:idx+1]
    x_b_s, m_b_s = X_b[idx:idx+1], mask_b[idx:idx+1]

    actual_prob_a = 1 / (1 + np.exp(-get_logit_a(model_a, x_a_s, m_a_s)))
    logit_b_val = get_logit_b(model_b, torch.tensor(x_b_s, dtype=torch.float32),
                               torch.tensor(m_b_s, dtype=torch.float32))
    actual_prob_b = float(torch.sigmoid(torch.tensor(logit_b_val)))
    risk_score = ALPHA * actual_prob_a + (1 - ALPHA) * actual_prob_b

    top_detection = ablate_detection_per_timestep(model_a, model_b, x_a_s, m_a_s, x_b_s, m_b_s)
    detection_reasons = phrases_for(top_detection, with_timestep=True)

    row = df.iloc[[idx]][CLASSIFIER_FEATURES].fillna(0).values
    predicted_class = clf.predict(row)[0]
    class_proba = float(clf.predict_proba(row)[0][list(clf.classes_).index(predicted_class)])
    top_classification = ablate_classification(clf, row, predicted_class)
    classification_reasons = phrases_for(top_classification)

    history_len = int(mask_b[idx].sum())
    cold_start = history_len < COLD_START_THRESHOLD

    entity_id = df.iloc[idx]["entity_id"]
    entity_type = df.iloc[idx]["entity_type"]
    timestamp = str(df.iloc[idx]["timestamp"])

    ent_rows = df[df["entity_id"] == entity_id].sort_values("timestamp")
    ent_rows = ent_rows[ent_rows["timestamp"] <= df.iloc[idx]["timestamp"]].tail(15)
    history = []
    for _, r in ent_rows.iterrows():
        history.append({
            "timestamp": str(r["timestamp"]),
            **{f: float(r[f]) if pd.notna(r[f]) else 0.0 for f in HISTORY_FEATURES}
        })

    return {
        "id": f"alert_{idx}",
        "entity_id": entity_id,
        "entity_type": entity_type,
        "timestamp": timestamp,
        "risk_score": round(float(risk_score), 4),
        "risk_threshold": round(float(alert_threshold), 4),
        "was_flagged": bool(risk_score >= alert_threshold),
        "cold_start": cold_start,
        "history_length": history_len,
        "predicted_attack_type": predicted_class,
        "classification_confidence": round(class_proba, 4),
        "true_label": str(df.iloc[idx]["label"]),  # available since this is val (labeled) data
        "detection_reasons": detection_reasons,
        "classification_reasons": classification_reasons,
        "entity_history": history,
    }


def load_everything():
    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    model_b = StatefulGRU(input_size=17, hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    clf = joblib.load(f"{OUT_DIR}/anomaly_classifier.joblib")

    va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)
    X_a, mask_a, ytype_a = va["X"], va["mask"], va["y_type"]
    vb = np.load(f"{OUT_DIR}/val_sequences_v2_scaled.npz", allow_pickle=True)
    X_b, mask_b = vb["X"], vb["mask"]

    df = pd.read_csv("../val_longterm.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["entity_type_user"] = (df["entity_type"] == "user").astype(int)
    df["entity_type_service_account"] = (df["entity_type"] == "service_account").astype(int)
    df["entity_type_edge_device"] = (df["entity_type"] == "edge_device").astype(int)

    models = {"a": model_a, "b": model_b, "clf": clf}
    data = {"X_a": X_a, "mask_a": mask_a, "X_b": X_b, "mask_b": mask_b, "df": df, "ytype": ytype_a}
    return models, data


if __name__ == "__main__":
    print("Loading models and data...")
    models, data = load_everything()

    print("Computing risk scores for full val set...")
    y_pred_a, _ = models["a"].forward(data["X_a"], data["mask_a"])
    with torch.no_grad():
        y_pred_b = torch.sigmoid(models["b"](
            torch.tensor(data["X_b"], dtype=torch.float32),
            torch.tensor(data["mask_b"], dtype=torch.float32)
        )).numpy()
    all_scores = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b

    n_total = len(all_scores)
    budget = max(1, int(n_total * ALERT_BUDGET_FRAC))
    alert_threshold = np.quantile(all_scores, 1 - budget / n_total)
    alert_indices = np.where(all_scores >= alert_threshold)[0]
    alert_indices = alert_indices[np.argsort(-all_scores[alert_indices])]  # highest risk first

    print(f"Alert budget: {budget}, actual alerts at threshold: {len(alert_indices)}")
    print(f"Generating full explanations for all {len(alert_indices)} alerts "
          f"(this is slower — per-timestep ablation per alert)...\n")

    alerts = []
    for n, idx in enumerate(alert_indices):
        if n % 20 == 0:
            print(f"  {n}/{len(alert_indices)}...")
        alerts.append(explain_single_alert(int(idx), models, data, alert_threshold))

    # cold-start events overall (for the summary cards), independent of budget
    history_lens = data["mask_b"].sum(axis=1)
    cold_start_total = int((history_lens < COLD_START_THRESHOLD).sum())

    label_counts = pd.Series(data["ytype"]).value_counts().to_dict()

    summary = {
        "total_events": int(n_total),
        "alert_budget": int(budget),
        "alerts_flagged": len(alert_indices),
        "alert_threshold": round(float(alert_threshold), 4),
        "cold_start_events_total": cold_start_total,
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "model_config": {
            "detector": "ensemble (10% Version A / 90% Version B)",
            "classifier": "RandomForest",
            "explainability": "logit-space leave-one-out ablation, per-timestep",
        },
    }

    output = {"summary": summary, "alerts": alerts}
    with open("dashboard_data.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved {len(alerts)} full alert explanations to dashboard_data.json")