"""
Deliverable 3 — Optimal ensemble weighting
Instead of guessing between max/mean, grid-search the blend weight alpha in
    score = alpha * A_score + (1 - alpha) * B_score
across [0, 1] in 0.05 steps, and pick the alpha that maximizes a chosen
composite objective at the SAME fixed alert budget (215 alerts / top 1%) used
everywhere else in this project, so the result stays comparable and honest.

Two composite objectives are reported (pick whichever matches how you want
to frame "maximum" for the judges):
  - macro_recall: unweighted average recall across all 7 attack types
    (treats every attack type as equally important, regardless of how many
    val examples it has)
  - micro_recall: overall recall across all anomalies pooled together
    (same as the 'Achieved recall' number in your other scripts — dominated
    by whichever classes have the most examples, e.g. brute_force)

Uses the exact same alignment check as ensemble_evaluate.py before combining
anything — this is not optional, see that file's docstring for why.
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
    if len(y_a) != len(y_b) or not np.array_equal(y_a, y_b) or not np.array_equal(ytype_a, ytype_b):
        raise SystemExit(
            "ALIGNMENT CHECK FAILED: val_sequences_scaled.npz and "
            "val_sequences_v2_scaled.npz do not line up row-for-row. "
            "Do not proceed with an ensemble until this is fixed."
        )
    print(f"Alignment check passed: {len(y_a)} rows.\n")


def per_class_recall_at_budget(y_pred, y, y_type, budget_frac=ALERT_BUDGET_FRAC):
    n_total = len(y)
    budget = max(1, int(n_total * budget_frac))
    threshold = np.quantile(y_pred, 1 - budget / n_total)
    alerts = y_pred >= threshold
    precision = (y[alerts] == 1).sum() / max(alerts.sum(), 1)

    per_class = {}
    for t in sorted(np.unique(y_type)):
        if t == "normal":
            continue
        m = y_type == t
        total = m.sum()
        caught = alerts[m].sum()
        per_class[t] = caught / max(total, 1)
    return per_class, precision


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
    y_b_np = y_b.numpy()

    check_alignment(y_a, ytype_a, y_b_np, ytype_b)
    y, y_type = y_a, ytype_a

    print(f"{'alpha':>6}{'macro_recall':>14}{'micro_recall':>14}{'precision':>12}{'pr_auc':>10}")
    results = []
    for alpha in np.arange(0.0, 1.0001, 0.05):
        blended = alpha * y_pred_a + (1 - alpha) * y_pred_b
        per_class, precision = per_class_recall_at_budget(blended, y, y_type)
        macro_recall = np.mean(list(per_class.values()))
        micro_recall = (blended >= np.quantile(blended, 1 - (int(len(y)*ALERT_BUDGET_FRAC))/len(y)))
        micro_recall = (y[micro_recall] == 1).sum() / max(y.sum(), 1)
        pr_auc = average_precision_score(y, blended)
        results.append((alpha, macro_recall, micro_recall, precision, pr_auc, per_class))
        print(f"{alpha:>6.2f}{macro_recall*100:>13.1f}%{micro_recall*100:>13.1f}%{precision*100:>11.1f}%{pr_auc:>10.4f}")

    best_macro = max(results, key=lambda r: r[1])
    best_micro = max(results, key=lambda r: r[2])

    print()
    print(f"Best alpha for MACRO recall (equal weight per attack type): alpha={best_macro[0]:.2f}, "
          f"macro_recall={best_macro[1]*100:.1f}%, pr_auc={best_macro[4]:.4f}")
    print("  per-class recall at this alpha:")
    for t, r in best_macro[5].items():
        print(f"    {t}: {r*100:.1f}%")

    print()
    print(f"Best alpha for MICRO recall (overall, size-weighted): alpha={best_micro[0]:.2f}, "
          f"micro_recall={best_micro[2]*100:.1f}%, pr_auc={best_micro[4]:.4f}")
    print("  per-class recall at this alpha:")
    for t, r in best_micro[5].items():
        print(f"    {t}: {r*100:.1f}%")