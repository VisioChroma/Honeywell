"""
Deliverable 3 — Concept drift handling (EWMA-gated baseline)

HONEST SCOPING NOTE (read before trusting the numbers below):
Your dataset does not have a separately-labeled "legitimate behavior drift
over weeks" scenario with its own ground truth. What it DOES have is
new_device_flag=1 events labeled "normal" — i.e. a real device change that
is NOT an attack. This is exactly the kind of event the brief's concept-
drift requirement is worried about ("new devices... should not be
permanently flagged"). This script uses that as a proxy test — it is a
genuine, real signal already in your schema, but it is one specific form
of drift, not a full simulation of "behavior slowly evolving over weeks."
Report this scoping honestly rather than implying full concept-drift
coverage.

MECHANISM (per your original Option 3 architecture, step 4):
For each entity, maintain an EWMA baseline of its own raw anomaly scores,
updated only using LOW-RISK events (raw_score below the global alert
threshold) — so an actual attack (high score) never gets absorbed into
what the baseline considers "normal" for that entity. Events processed in
chronological, per-entity order (this is already the natural row order in
the sequence files, since seq_prep_v2.py groups by entity_id then sorts by
timestamp before windowing).

    baseline[entity] = 0.95 * baseline[entity] + 0.05 * raw_score   (only if raw_score < global_threshold)
    adjusted_score   = max(raw_score - baseline[entity], 0)

Then alerts are drawn from adjusted_score instead of raw_score, using the
same total budget size (so this is a re-ranking, not a budget change).

WHAT WE MEASURE:
1. False positive rate on new_device_flag=1 / normal events: raw vs adjusted
   (the concept-drift claim: should go DOWN without hurting attack recall)
2. Overall attack recall: raw vs adjusted (must NOT meaningfully drop —
   otherwise we've just made the detector worse, not "handled drift")
"""
import numpy as np
import torch
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALERT_BUDGET_FRAC = 0.01
ALPHA = 0.10
EWMA_DECAY = 0.95   # weight on old baseline (0.95 -> slow, stable adaptation)


def load_a():
    va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)
    return va["X"], va["y"], va["mask"], va["y_type"]


def load_b_scaled():
    d = np.load(f"{OUT_DIR}/val_sequences_v2_scaled.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    return X, y, mask, d["y_type"]


def load_b_raw_and_ids():
    """Unscaled v2 sequences, needed for: (a) true new_device_flag value,
    (b) entity_ids to group the EWMA state correctly. Row order is assumed
    identical to the scaled version (same script produces both from the
    same source rows, before vs after RobustScaler)."""
    d = np.load(f"{OUT_DIR}/val_sequences_v2.npz", allow_pickle=True)
    keys = d.files
    if "entity_ids" not in keys:
        raise SystemExit(
            "val_sequences_v2.npz has no 'entity_ids' field — check "
            "seq_prep_v2.py's savez call and confirm it saves entity_ids "
            "(seq_prep.py's non-v2 version does). Cannot run the "
            "per-entity EWMA baseline without knowing which events belong "
            "to which entity."
        )
    return d["X"], d["entity_ids"]


# index of new_device_flag within the 17-feature v2 vector — this MUST match
# your actual seq_prep_v2.py FEATURE_COLS order. The original 13 features
# (from seq_prep.py) had new_device_flag at index 4; v2 should preserve that
# ordering and simply append 4 long-term columns after it. Verify against
# your seq_prep_v2.py FEATURE_COLS list before trusting this index.
NEW_DEVICE_FLAG_IDX = 4


if __name__ == "__main__":
    X_a, y_a, mask_a, ytype_a = load_a()
    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    y_pred_a, _ = model_a.forward(X_a, mask_a)

    X_b, y_b, mask_b, ytype_b = load_b_scaled()
    model_b = StatefulGRU(input_size=X_b.shape[2], hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    with torch.no_grad():
        logits_b = model_b(X_b, mask_b)
        y_pred_b = torch.sigmoid(logits_b).numpy()

    y = y_a
    y_type = ytype_a
    raw_score = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b

    X_b_raw, entity_ids = load_b_raw_and_ids()
    new_device_flag = X_b_raw[:, -1, NEW_DEVICE_FLAG_IDX]  # current event's own flag value

    n_total = len(y)
    budget = max(1, int(n_total * ALERT_BUDGET_FRAC))
    global_threshold = np.quantile(raw_score, 1 - budget / n_total)

    # --- EWMA-gated baseline, processed in existing (entity-grouped,
    # chronological) row order ---
    baseline = {}
    adjusted_score = np.zeros_like(raw_score)
    for i in range(n_total):
        eid = entity_ids[i]
        s = raw_score[i]
        if eid not in baseline:
            baseline[eid] = s  # initialize on first sighting
        adjusted_score[i] = max(s - baseline[eid], 0.0)
        if s < global_threshold:  # gate: only low-risk events update baseline
            baseline[eid] = EWMA_DECAY * baseline[eid] + (1 - EWMA_DECAY) * s

    adj_threshold = np.quantile(adjusted_score, 1 - budget / n_total)
    alerts_raw = raw_score >= global_threshold
    alerts_adj = adjusted_score >= adj_threshold

    print(f"Global budget: {budget} alerts (top {ALERT_BUDGET_FRAC:.0%})\n")

    # --- 1. false positives on the benign new-device-flag proxy ---
    drift_candidates = (y == 0) & (new_device_flag == 1)
    n_drift = drift_candidates.sum()
    fp_raw = int((alerts_raw & drift_candidates).sum())
    fp_adj = int((alerts_adj & drift_candidates).sum())
    print("--- Concept-drift proxy: benign new_device_flag=1 events (normal, not attacks) ---")
    print(f"  total such events in val: {n_drift}")
    print(f"  falsely alerted, RAW score:      {fp_raw} ({fp_raw/max(n_drift,1):.2%} of them)")
    print(f"  falsely alerted, ADJUSTED score: {fp_adj} ({fp_adj/max(n_drift,1):.2%} of them)")
    print()

    # --- 2. overall attack recall: raw vs adjusted (must not regress) ---
    n_anom = int(y.sum())
    recall_raw = (y[alerts_raw] == 1).sum() / max(n_anom, 1)
    recall_adj = (y[alerts_adj] == 1).sum() / max(n_anom, 1)
    print("--- Overall attack recall: raw vs EWMA-adjusted (must not drop) ---")
    print(f"  RAW recall:      {recall_raw:.1%}")
    print(f"  ADJUSTED recall: {recall_adj:.1%}")
    print()

    print("--- Recall by attack type: raw vs adjusted ---")
    for t in sorted(np.unique(y_type)):
        if t == "normal":
            continue
        m = y_type == t
        total = m.sum()
        c_raw = (alerts_raw[m] & (y[m] == 1)).sum()
        c_adj = (alerts_adj[m] & (y[m] == 1)).sum()
        print(f"  {t}: raw {c_raw}/{total} ({c_raw/max(total,1):.1%})  "
              f"| adjusted {c_adj}/{total} ({c_adj/max(total,1):.1%})")