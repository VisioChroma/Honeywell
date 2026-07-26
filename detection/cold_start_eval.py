"""
Deliverable 3 — Cold-start validation
"Handles cold-start entities" was previously an architectural claim
(left-padding + mask means short-history entities produce valid windows) —
this script turns it into a measured number.

DEFINITION: an event is "cold-start" if mask.sum() < 5 for its window — i.e.
the entity had fewer than 5 real prior events at that point. This is exact,
not approximate: seq_prep_v2.py left-pads (padding at the START of the
window), so mask.sum() IS the entity's true history length at that event,
capped at the window length.

METHODOLOGY: use the SAME global alert threshold the full system would use
in production (top-1% of ALL val events) — not a separately-tuned threshold
for the cold-start subset. This answers the real question: "when our system
raises its normal budget of alerts, how well does it cover new entities
specifically?" — not an idealized best-case for that subset alone.

Uses the locked final config: ensemble at alpha=0.10 (10% Version A / 90%
Version B), per the alpha sweep in optimize_ensemble_alpha.py.
"""
import numpy as np
import torch
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALERT_BUDGET_FRAC = 0.01
ALPHA = 0.10                # locked ensemble weight (Version A share)
COLD_START_THRESHOLD = 5    # fewer than this many real prior events = cold-start


def load_a():
    va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)
    return va["X"], va["y"], va["mask"], va["y_type"]


def load_b():
    d = np.load(f"{OUT_DIR}/val_sequences_v2_scaled.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    return X, y, mask, d["y_type"]


def report(alerts, y, y_type, subset_mask, label):
    m = subset_mask
    n_subset = m.sum()
    n_anom_subset = int(y[m].sum())
    caught = int((alerts[m] & (y[m] == 1)).sum())
    n_alerts_in_subset = int(alerts[m].sum())

    print(f"--- {label} ---")
    print(f"  events in subset: {n_subset}  (true anomalies: {n_anom_subset})")
    print(f"  alerts landing in this subset: {n_alerts_in_subset}")
    if n_anom_subset > 0:
        print(f"  recall: {caught}/{n_anom_subset} = {caught/n_anom_subset:.1%}")
    else:
        print("  recall: n/a (no anomalies in this subset)")
    if n_alerts_in_subset > 0:
        print(f"  precision (within subset's alerts): {caught/n_alerts_in_subset:.1%}")
    print()

    print("  recall by attack type within this subset:")
    for t in sorted(np.unique(y_type)):
        if t == "normal":
            continue
        tm = m & (y_type == t)
        total = tm.sum()
        if total == 0:
            continue
        c = int((alerts[tm] & (y[tm] == 1)).sum())
        print(f"    {t}: {c}/{total} caught ({c/total:.1%})")
    print()


if __name__ == "__main__":
    X_a, y_a, mask_a, ytype_a = load_a()
    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    y_pred_a, _ = model_a.forward(X_a, mask_a)

    X_b, y_b, mask_b, ytype_b = load_b()
    model_b = StatefulGRU(input_size=X_b.shape[2], hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    with torch.no_grad():
        logits_b = model_b(X_b, mask_b)
        y_pred_b = torch.sigmoid(logits_b).numpy()

    y = y_a  # alignment already confirmed in prior scripts (same val set)
    y_type = ytype_a
    mask_b_np = mask_b.numpy()

    # history length = number of real (non-padded) timesteps in the window
    history_len = mask_b_np.sum(axis=1)
    cold_start = history_len < COLD_START_THRESHOLD
    warm = ~cold_start

    print(f"Cold-start definition: history_len < {COLD_START_THRESHOLD} real prior events")
    print(f"Cold-start events: {cold_start.sum()} ({cold_start.mean():.1%} of val)")
    print(f"Warm events: {warm.sum()} ({warm.mean():.1%} of val)\n")

    blended = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b

    n_total = len(y)
    budget = max(1, int(n_total * ALERT_BUDGET_FRAC))
    threshold = np.quantile(blended, 1 - budget / n_total)
    alerts = blended >= threshold

    print(f"Global alert budget: {budget} alerts (top {ALERT_BUDGET_FRAC:.0%} of ALL val events)\n")
    print("=" * 60)
    report(alerts, y, y_type, cold_start, "COLD-START subset")
    print("=" * 60)
    report(alerts, y, y_type, warm, "WARM (established history) subset")
    print("=" * 60)
    report(alerts, y, y_type, np.ones_like(cold_start, dtype=bool), "FULL val set (reference)")