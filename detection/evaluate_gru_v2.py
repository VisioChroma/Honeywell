"""
Deliverable 3 — Step 7: Evaluate Version B (PyTorch GRU + long-term features)
Same methodology as Version A's evaluation — val set only, top-1% alert budget,
per-attack-type recall breakdown — so the two are directly comparable.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."

def load_split(name):
    d = np.load(f"{OUT_DIR}/{name}_sequences_v2_scaled.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    return X, y, mask, d["y_type"]

if __name__ == "__main__":
    X_val, y_val, mask_val, ytype_val = load_split("val")

    model = StatefulGRU(input_size=X_val.shape[2], hidden_size=32)
    model.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(X_val, mask_val)
        y_pred = torch.sigmoid(logits).numpy()

    y_val_np = y_val.numpy()
    pr_auc = average_precision_score(y_val_np, y_pred)
    print(f"Val PR-AUC: {pr_auc:.4f}")

    n_total = len(y_val_np)
    budget = max(1, int(n_total * 0.01))
    threshold = np.quantile(y_pred, 1 - budget / n_total)
    alerts = y_pred >= threshold

    recall = (y_val_np[alerts] == 1).sum() / max(y_val_np.sum(), 1)
    precision = (y_val_np[alerts] == 1).sum() / max(alerts.sum(), 1)
    n_anomalies = int(y_val_np.sum())
    theoretical_max = min(1.0, budget / max(n_anomalies, 1))

    print(f"Alert budget (top 1%): {budget} alerts")
    print(f"Theoretical max recall: {theoretical_max:.1%}")
    print(f"Achieved recall: {recall:.1%} ({recall/theoretical_max:.1%} of ceiling)")
    print(f"Achieved precision: {precision:.1%}")
    print()
    
    print("Recall by attack type at top-1% budget:")
    for t in sorted(np.unique(ytype_val)):
        if t == "normal":
            continue
        m = ytype_val == t
        caught, total = alerts[m].sum(), m.sum()
        print(f"  {t}: {caught}/{total} caught ({caught/max(total,1):.1%})")