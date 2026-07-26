"""
Deliverable 3 — FINAL test-set evaluation of the locked ensemble config.
Alpha=0.10, raw scores, no EWMA. Run ONCE. This produces the numbers that
go in the report.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALPHA = 0.10
ALERT_BUDGET_FRAC = 0.01

def load_a(split):
    d = np.load(f"{OUT_DIR}/{split}_sequences_scaled.npz", allow_pickle=True)
    return d["X"], d["y"], d["mask"], d["y_type"]

def load_b(split):
    d = np.load(f"{OUT_DIR}/{split}_sequences_v2_scaled.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    return X, y, mask, d["y_type"]

def check_alignment(y_a, ytype_a, y_b, ytype_b):
    if len(y_a) != len(y_b) or not np.array_equal(y_a, y_b) or not np.array_equal(ytype_a, ytype_b):
        raise SystemExit("ALIGNMENT CHECK FAILED on TEST set — stopping.")
    print(f"Alignment check passed: {len(y_a)} rows.\n")

if __name__ == "__main__":
    X_a, y_a, mask_a, ytype_a = load_a("test")
    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    y_pred_a, _ = model_a.forward(X_a, mask_a)

    X_b, y_b, mask_b, ytype_b = load_b("test")
    model_b = StatefulGRU(input_size=X_b.shape[2], hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    with torch.no_grad():
        y_pred_b = torch.sigmoid(model_b(X_b, mask_b)).numpy()
    y_b_np = y_b.numpy()

    check_alignment(y_a, ytype_a, y_b_np, ytype_b)
    y, y_type = y_a, ytype_a

    y_pred = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b

    n_total = len(y)
    n_anomalies = int(y.sum())
    budget = max(1, int(n_total * ALERT_BUDGET_FRAC))
    threshold = np.quantile(y_pred, 1 - budget / n_total)
    alerts = y_pred >= threshold

    pr_auc = average_precision_score(y, y_pred)
    recall = (y[alerts] == 1).sum() / max(y.sum(), 1)
    precision = (y[alerts] == 1).sum() / max(alerts.sum(), 1)
    theoretical_max = min(1.0, budget / max(n_anomalies, 1))

    print("=" * 60)
    print("FINAL TEST-SET RESULT — Ensemble alpha=0.10 (locked config)")
    print("=" * 60)
    print(f"Test events: {n_total}, true anomalies: {n_anomalies} ({n_anomalies/n_total:.2%})")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Alert budget (top 1%): {budget} alerts")
    print(f"Theoretical max recall: {theoretical_max:.1%}")
    print(f"Achieved recall: {recall:.1%} ({recall/theoretical_max:.1%} of ceiling)")
    print(f"Achieved precision: {precision:.1%}")
    print("\nRecall by attack type:")
    for t in sorted(np.unique(y_type)):
        if t == "normal":
            continue
        m = y_type == t
        caught, total = alerts[m].sum(), m.sum()
        print(f"  {t}: {caught}/{total} caught ({caught/max(total,1):.1%})")