"""
================================================================================
Deliverable 5 — Explainability Layer, v2 (hardened for all 7 attack types)
================================================================================
Two real issues found in v1's output and fixed here:

1. IMPOSSIBLE_TRAVEL detection explanation didn't mention geo/speed at all,
   even though the classifier explanation correctly did. Root cause: v1
   ablated a feature across the ENTIRE 15-step window at once. If the
   impossible-travel signal is concentrated at ONE timestep in the window,
   whole-window ablation can dilute it relative to features that are
   moderately elevated across many steps. FIX: ablate feature-by-feature,
   TIMESTEP-by-timestep, and take the MAX drop across timesteps per feature
   (not the sum — summing overweights features that are mildly elevated
   everywhere over ones that spike hard once). This also tells you WHICH
   event in the window mattered most, not just which feature.

2. INSIDER_DRIFT (and any other case where the true score is below the
   alert threshold) printed "flagged due to..." even though the score was
   NEGATIVE (i.e., this event would NOT actually be in the alert budget).
   FIX: compute the real alert threshold (same top-1% budget as the locked
   detector) and phrase the explanation honestly depending on whether the
   event actually cleared it.

Also: explicit .strip() on every phrase before joining, to eliminate any
spacing inconsistency in the output regardless of source.
================================================================================
"""
import sys
import os
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
    "hour_sin": "unusual time of day",
    "hour_cos": "unusual time of day",
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
        logits = model(x, m)
    return logits.numpy()[0]


def compute_alert_threshold_probability(model_a, model_b):
    """Real production threshold: top-1% quantile of PROBABILITY-space blend
    (alpha*sigmoid(logit_a) + (1-alpha)*sigmoid(logit_b)), matching
    ensemble_evaluate.py / rank_diagnostic.py exactly. NOT logit-space
    averaging — sigmoid is nonlinear, so avg(sigmoid(a),sigmoid(b)) !=
    sigmoid(avg(a,b)); these can disagree on borderline cases, which is
    exactly the insider_drift scenario this script is meant to get right."""
    va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)
    y_pred_a, _ = model_a.forward(va["X"], va["mask"])

    d = np.load(f"{OUT_DIR}/val_sequences_v2_scaled.npz", allow_pickle=True)
    X_b = torch.tensor(d["X"], dtype=torch.float32)
    mask_b = torch.tensor(d["mask"], dtype=torch.float32)
    with torch.no_grad():
        y_pred_b = torch.sigmoid(model_b(X_b, mask_b)).numpy()

    blended_prob = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b
    n_total = len(blended_prob)
    budget = max(1, int(n_total * ALERT_BUDGET_FRAC))
    threshold = np.quantile(blended_prob, 1 - budget / n_total)
    return threshold


def ablate_detection_per_timestep(model_a, model_b, x_a, m_a, x_b, m_b, top_n=3):
    """
    Per-feature, per-timestep ablation. For each feature, ablate it at each
    REAL (non-padded) timestep individually, and take the MAX drop across
    timesteps — this finds a signal concentrated at one moment in the
    window, which whole-window ablation can dilute.
    Returns: top_n (feature, max_drop, which_timestep) + original combined logit.
    """
    orig_logit_a = get_logit_a(model_a, x_a, m_a)
    x_b_t = torch.tensor(x_b, dtype=torch.float32)
    m_b_t = torch.tensor(m_b, dtype=torch.float32)
    orig_logit_b = get_logit_b(model_b, x_b_t, m_b_t)
    orig_combined = ALPHA * orig_logit_a + (1 - ALPHA) * orig_logit_b

    T = x_a.shape[1]
    real_timesteps = np.where(m_a[0] == 1)[0]  # only ablate real, non-padded steps

    best_per_feature = {}  # fname -> (max_drop, timestep)

    # Version A features (13) — contribute via alpha weight
    for i, fname in enumerate(DETECTION_FEATURES):
        best_drop, best_t = -np.inf, None
        for t in real_timesteps:
            x_ablated = x_a.copy()
            x_ablated[0, t, i] = 0
            ablated_logit_a = get_logit_a(model_a, x_ablated, m_a)
            drop = ALPHA * (orig_logit_a - ablated_logit_a)
            if drop > best_drop:
                best_drop, best_t = drop, t
        best_per_feature[fname] = (best_drop, best_t)

    # Version B features (17, includes long-term) — contribute via (1-alpha) weight
    real_timesteps_b = np.where(m_b[0] == 1)[0]
    for i, fname in enumerate(DETECTION_FEATURES_B):
        best_drop, best_t = -np.inf, None
        for t in real_timesteps_b:
            x_ablated = x_b.copy()
            x_ablated[0, t, i] = 0
            x_ablated_t = torch.tensor(x_ablated, dtype=torch.float32)
            ablated_logit_b = get_logit_b(model_b, x_ablated_t, m_b_t)
            drop = (1 - ALPHA) * (orig_logit_b - ablated_logit_b)
            if drop > best_drop:
                best_drop, best_t = drop, t
        if fname in best_per_feature:
            # combine A and B contributions for shared features (take the
            # timestep from whichever contributed more, sum the drops)
            prev_drop, prev_t = best_per_feature[fname]
            total = prev_drop + best_drop
            best_per_feature[fname] = (total, best_t if best_drop > prev_drop else prev_t)
        else:
            best_per_feature[fname] = (best_drop, best_t)

    supporting = [(f, d, t) for f, (d, t) in best_per_feature.items() if d > 0]
    supporting.sort(key=lambda x: -x[1])
    return supporting[:top_n], orig_combined


def ablate_classification(clf, x_row, predicted_class, top_n=3):
    orig_proba = clf.predict_proba(x_row)[0]
    class_idx = list(clf.classes_).index(predicted_class)
    orig_score = orig_proba[class_idx]
    eps = 1e-6
    orig_logodds = np.log((orig_score + eps) / (1 - orig_score + eps))

    drops = []
    for i, fname in enumerate(CLASSIFIER_FEATURES):
        x_ablated = x_row.copy()
        x_ablated[0, i] = 0
        ablated_proba = clf.predict_proba(x_ablated)[0][class_idx]
        ablated_logodds = np.log((ablated_proba + eps) / (1 - ablated_proba + eps))
        drops.append((fname, orig_logodds - ablated_logodds))

    supporting = [d for d in drops if d[1] > 0]
    supporting.sort(key=lambda d: -d[1])
    return supporting[:top_n]


def build_detection_explanation(top_features, score, threshold, window_len=15):
    was_flagged = score >= threshold
    if not top_features:
        reason = "no single dominant factor (diffuse signal across many features)"
    else:
        phrases = []
        for f, drop, t in top_features:
            phrase = FEATURE_PHRASES.get(f, f).strip()
            steps_ago = window_len - 1 - t if t is not None else None
            if steps_ago is not None and steps_ago > 0:
                phrase += f" ({steps_ago} event(s) earlier in window)"
            phrases.append(phrase)
        reason = " + ".join(phrases)

    if was_flagged:
        return f"FLAGGED — due to {reason}"
    else:
        return f"NOT flagged (score below alert threshold) — nearest contributing factors: {reason}"


def build_classification_explanation(top_features):
    if not top_features:
        return "predicted, but no single dominant factor (diffuse signal)"
    phrases = [FEATURE_PHRASES.get(f, f).strip() for f, drop in top_features]
    return "predicted due to " + " + ".join(phrases)


if __name__ == "__main__":
    print("Loading models...")
    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    model_b = StatefulGRU(input_size=17, hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    clf = joblib.load(f"{OUT_DIR}/anomaly_classifier.joblib")

    print("Computing real alert threshold (top-1% budget, probability space, matches production)...")
    alert_threshold = compute_alert_threshold_probability(model_a, model_b)
    print(f"  alert threshold (probability): {alert_threshold:.4f}\n")

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

    print("Generating example explanations, one per attack type:\n")
    for attack_type in sorted(set(ytype_a)):
        if attack_type == "normal":
            continue
        idx = np.where(ytype_a == attack_type)[0]
        if len(idx) == 0:
            continue
        idx = idx[len(idx) // 2]

        x_a_single = X_a[idx:idx+1]
        m_a_single = mask_a[idx:idx+1]
        x_b_single = X_b[idx:idx+1]
        m_b_single = mask_b[idx:idx+1]

        top_detection, score = ablate_detection_per_timestep(
            model_a, model_b, x_a_single, m_a_single, x_b_single, m_b_single
        )
        # score above is logit-space, used only for ranking features during
        # ablation (legitimate — avoids sigmoid saturation). For the actual
        # flagged/not-flagged decision, use the real production probability.
        actual_prob_a = 1 / (1 + np.exp(-get_logit_a(model_a, x_a_single, m_a_single)))
        logit_b_val = get_logit_b(model_b, torch.tensor(x_b_single, dtype=torch.float32),
                                   torch.tensor(m_b_single, dtype=torch.float32))
        actual_prob_b = float(torch.sigmoid(torch.tensor(logit_b_val)))
        actual_prob = ALPHA * actual_prob_a + (1 - ALPHA) * actual_prob_b

        detection_explanation = build_detection_explanation(top_detection, actual_prob, alert_threshold)

        row = df.iloc[[idx]][CLASSIFIER_FEATURES].fillna(0).values
        predicted_class = clf.predict(row)[0]
        top_classification = ablate_classification(clf, row, predicted_class)
        classification_explanation = build_classification_explanation(top_classification)

        print(f"--- True label: {attack_type} | entity: {df.iloc[idx]['entity_id']} ---")
        print(f"  Detection score: {actual_prob:.4f} (probability)  |  threshold: {alert_threshold:.4f}"
              f"  [ranking used logit-space, shown separately: {score:.3f}]")
        print(f"  Detection explanation: {detection_explanation}")
        print(f"  Predicted attack type: {predicted_class}")
        print(f"  Classification explanation: {classification_explanation}")
        print()