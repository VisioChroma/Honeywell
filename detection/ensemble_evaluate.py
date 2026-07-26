"""
Deliverable 3 — Ensemble evaluation: Version A (fixed-window NumPy GRU)
combined with Version B (stateful PyTorch GRU + long-term features).

WHY AN ENSEMBLE:
A and B are good at different things for structural reasons — A is simple
and stable and wins on the easy/well-separated classes (brute_force), B has
long-term memory and wins on the slow-building ones (low_and_slow_exfil,
impossible_travel). Under a single shared alert budget, forcing one model's
loss-weighting to cover all 7 classes creates a real trade-off (see B's
brute_force regression). Combining both models' scores lets each one "own"
its strength instead of compromising.

CRITICAL PRE-CHECK — ALIGNMENT:
A and B were trained on sequences built by two different prep scripts
(seq_prep.py -> val_sequences_scaled.npz, 14 features vs.
seq_prep_v2.py -> val_sequences_v2_scaled.npz, 17 features). Nothing
guarantees row i means the same event in both files unless the two prep
scripts iterate entities/windows in the exact same order. We verify this
directly (by comparing y and y_type arrays element-by-element) before
combining anything. If they don't match, this script will tell you and
stop — it will NOT silently average misaligned scores.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALERT_BUDGET_FRAC = 0.01


def load_a():
    va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)
    return va["X"], va["y"], va["mask"], va["y_type"]


def load_b():
    d = np.load(f"{OUT_DIR}/val_sequences_v2_scaled.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    return X, y, mask, d["y_type"]


def check_alignment(y_a, ytype_a, y_b, ytype_b):
    if len(y_a) != len(y_b):
        raise SystemExit(
            f"ALIGNMENT CHECK FAILED: different row counts "
            f"(A={len(y_a)}, B={len(y_b)}). These val sets are not the same "
            f"events in the same order — do not ensemble them as-is."
        )
    y_match = np.array_equal(y_a, y_b)
    type_match = np.array_equal(ytype_a, ytype_b)
    if not (y_match and type_match):
        n_y_mismatch = int((y_a != y_b).sum())
        n_type_mismatch = int((ytype_a != ytype_b).sum())
        raise SystemExit(
            "ALIGNMENT CHECK FAILED: val_sequences_scaled.npz and "
            "val_sequences_v2_scaled.npz do not line up row-for-row "
            f"(label mismatches: {n_y_mismatch}, type mismatches: {n_type_mismatch}). "
            "Combining scores from these two files would silently produce "
            "meaningless numbers. You'll need to rebuild one of the sequence "
            "files so both prep scripts emit rows in identical entity/window "
            "order before an ensemble is valid."
        )
    print(f"Alignment check passed: {len(y_a)} rows, labels and types match exactly.\n")


def eval_scores(y_pred, y, y_type, label):
    n_total = len(y)
    n_anomalies = int(y.sum())
    budget = max(1, int(n_total * ALERT_BUDGET_FRAC))
    threshold = np.quantile(y_pred, 1 - budget / n_total)
    alerts = y_pred >= threshold

    pr_auc = average_precision_score(y, y_pred)
    recall = (y[alerts] == 1).sum() / max(y.sum(), 1)
    precision = (y[alerts] == 1).sum() / max(alerts.sum(), 1)
    theoretical_max = min(1.0, budget / max(n_anomalies, 1))

    print(f"--- {label} ---")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Alert budget (top 1%): {budget} alerts")
    print(f"Theoretical max recall: {theoretical_max:.1%}")
    print(f"Achieved recall: {recall:.1%} ({recall/theoretical_max:.1%} of ceiling)")
    print(f"Achieved precision: {precision:.1%}")
    per_class = {}
    for t in sorted(np.unique(y_type)):
        if t == "normal":
            continue
        m = y_type == t
        caught, total = alerts[m].sum(), m.sum()
        r = caught / max(total, 1)
        per_class[t] = r
        print(f"  {t}: {caught}/{total} caught ({r:.1%})")
    print()
    return {"pr_auc": pr_auc, "recall": recall, "precision": precision, "per_class": per_class}


if __name__ == "__main__":
    # --- load both models' predictions on their respective val sets ---
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
    y_b_np = y_b.numpy()

    # --- mandatory alignment check before combining anything ---
    check_alignment(y_a, ytype_a, y_b_np, ytype_b)

    y = y_a  # identical to y_b_np after the check above
    y_type = ytype_a

    # --- individual baselines, same methodology, for the comparison table ---
    metrics_a = eval_scores(y_pred_a, y, y_type, "Version A alone")
    metrics_b = eval_scores(y_pred_b, y, y_type, "Version B alone")

    # --- ensemble: max and mean, both worth checking ---
    # Scores are both sigmoid outputs in [0,1] from models trained on the
    # same label definition, so combining them directly is reasonable
    # (no rescaling needed) — but max vs mean make different assumptions:
    # max lets either model "win" on a sample: mean requires both to agree.
    y_pred_max = np.maximum(y_pred_a, y_pred_b)
    y_pred_mean = (y_pred_a + y_pred_b) / 2.0

    metrics_max = eval_scores(y_pred_max, y, y_type, "ENSEMBLE (max of A, B)")
    metrics_mean = eval_scores(y_pred_mean, y, y_type, "ENSEMBLE (mean of A, B)")

    print("=" * 70)
    print("SUMMARY — recall by attack type, all four side by side")
    print("=" * 70)
    all_types = sorted(metrics_a["per_class"].keys())
    header = f"{'attack type':<22}{'A':>8}{'B':>8}{'max':>8}{'mean':>8}"
    print(header)
    for t in all_types:
        row = (
            f"{t:<22}"
            f"{metrics_a['per_class'][t]*100:>7.1f}%"
            f"{metrics_b['per_class'][t]*100:>7.1f}%"
            f"{metrics_max['per_class'][t]*100:>7.1f}%"
            f"{metrics_mean['per_class'][t]*100:>7.1f}%"
        )
        print(row)
    print()
    print(
        f"{'PR-AUC':<22}"
        f"{metrics_a['pr_auc']:>8.4f}"
        f"{metrics_b['pr_auc']:>8.4f}"
        f"{metrics_max['pr_auc']:>8.4f}"
        f"{metrics_mean['pr_auc']:>8.4f}"
    )
    print(
        f"{'precision@budget':<22}"
        f"{metrics_a['precision']*100:>7.1f}%"
        f"{metrics_b['precision']*100:>7.1f}%"
        f"{metrics_max['precision']*100:>7.1f}%"
        f"{metrics_mean['precision']*100:>7.1f}%"
    )