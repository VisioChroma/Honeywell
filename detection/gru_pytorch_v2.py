"""
Deliverable 3 — Step 6: Stateful GRU (PyTorch, Version B)
Same masked, per-timestep hidden-state design as the NumPy Version A model
(padded timesteps carry the hidden state forward unchanged, not corrupted by
zero-input), but using nn.GRUCell + autograd instead of hand-written BPTT.
Trained on the 17-feature set including long-term (7d/30d) memory.

v3 of the training loop (two fixes layered together, in order of what we
learned from the previous two runs):

1. CLASS-WEIGHTED LOSS
   Plain BCE is dominated by whichever patterns are easiest to fit
   (brute_force, credential_stuffing, device_spoofing — big obvious signals).
   The hard, subtle classes (lateral_movement, low_and_slow_exfil,
   insider_drift) get almost no gradient signal once the easy classes are
   nailed, no matter how long/short we train. Upweighting their samples in
   the loss forces the model to keep improving on them specifically.

2. COMPOSITE CHECKPOINT METRIC
   Checkpointing on raw val_loss picked an overfit epoch (run 1).
   Checkpointing on overall val PR-AUC picked an epoch that was best in
   aggregate but *worse* on the hard classes than the simpler baseline
   (run 2) — because PR-AUC is dominated by the same easy/common classes.
   So: track recall-at-1%-budget for the hard classes specifically, each
   epoch, and checkpoint on the mean of those. This directly optimizes the
   thing we actually care about, instead of a proxy for it.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import json

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = "."
HIDDEN_SIZE = 32
LR = 0.005
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 8

# Classes we're explicitly trying to win on. Tune these weights if needed —
# start at 3x for the hard classes, 1x for everything else (normal + the
# easy attack types already sit near their ceiling).
HARD_CLASSES = ["lateral_movement", "low_and_slow_exfil", "insider_drift"]
CLASS_WEIGHTS = {
    "normal": 1.0,
    "brute_force": 1.0,
    "credential_stuffing": 1.0,
    "device_spoofing": 1.0,
    "impossible_travel": 1.5,
    "lateral_movement": 3.0,
    "low_and_slow_exfil": 3.0,
    "insider_drift": 3.0,
}

ALERT_BUDGET_FRAC = 0.01  # top-1%, matching evaluate_gru_v2.py


class StatefulGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, X, mask):
        # X: (batch, T, input_size), mask: (batch, T) — 1=real, 0=padding
        batch, T, _ = X.shape
        h = torch.zeros(batch, self.hidden_size, device=X.device)
        for t in range(T):
            x_t = X[:, t, :]
            m_t = mask[:, t:t + 1]
            h_new = self.cell(x_t, h)
            h = m_t * h_new + (1 - m_t) * h  # padded steps: keep h unchanged
        logits = self.out(h).squeeze(-1)
        return logits


def load_split(name):
    d = np.load(f"{OUT_DIR}/{name}_sequences_v2_scaled.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    y_type = d["y_type"]
    return X, y, mask, y_type


def weights_for(y_type_array):
    """Map an array of string labels -> float32 tensor of per-sample weights."""
    default = 1.0
    w = np.array([CLASS_WEIGHTS.get(t, default) for t in y_type_array], dtype=np.float32)
    return torch.tensor(w, dtype=torch.float32)


def hard_class_recall_at_budget(y_val_np, y_pred, ytype_val, budget_frac=ALERT_BUDGET_FRAC):
    """
    Compute recall-at-top-k%-budget for each hard class, using a single
    global threshold (same alert budget the real system would operate
    under) — then return the per-class dict and the mean across hard
    classes (the number we checkpoint on).
    """
    n_total = len(y_val_np)
    budget = max(1, int(n_total * budget_frac))
    threshold = np.quantile(y_pred, 1 - budget / n_total)
    alerts = y_pred >= threshold

    per_class = {}
    for t in HARD_CLASSES:
        m = (ytype_val == t)
        total = m.sum()
        if total == 0:
            continue
        caught = alerts[m].sum()
        per_class[t] = caught / total

    if not per_class:
        return per_class, 0.0
    mean_recall = float(np.mean(list(per_class.values())))
    return per_class, mean_recall


def train():
    X_train, y_train, mask_train, ytype_train = load_split("train")
    X_val, y_val, mask_val, ytype_val = load_split("val")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    model = StatefulGRU(input_size=X_train.shape[2], hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pos_idx = torch.where(y_train == 1)[0].numpy()
    neg_idx = torch.where(y_train == 0)[0].numpy()
    print(f"  train: {len(pos_idx)} anomalies, {len(neg_idx)} normal")

    # Precompute per-sample loss weights for the full train set once;
    # we'll index into this per-batch (including the oversampled positives).
    sample_weights = weights_for(ytype_train)

    n = X_train.shape[0]
    history = []
    best_score = -float("inf")   # tracks mean hard-class recall@budget
    best_state = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(epoch)
        order = rng.permutation(n)
        n_pos_per_batch = max(1, BATCH_SIZE // 8)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n, BATCH_SIZE):
            batch_idx = order[start:start + BATCH_SIZE]
            if len(pos_idx) > 0:
                extra_pos = rng.choice(pos_idx, size=n_pos_per_batch, replace=True)
                batch_idx = np.concatenate([batch_idx, extra_pos])

            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            mb = mask_train[batch_idx]
            wb = sample_weights[batch_idx]

            optimizer.zero_grad()
            logits = model(Xb, mb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, mask_val)
            val_loss = nn.functional.binary_cross_entropy_with_logits(val_logits, y_val).item()
            val_pred = torch.sigmoid(val_logits).numpy()
            val_pr_auc = average_precision_score(y_val.numpy(), val_pred)

        per_class_recall, composite_score = hard_class_recall_at_budget(
            y_val.numpy(), val_pred, ytype_val
        )

        avg_train_loss = epoch_loss / n_batches
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
            "hard_class_recall": per_class_recall,
            "composite_score": composite_score,
        })

        if composite_score > best_score:
            best_score = composite_score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = "  <- best so far"
        else:
            epochs_no_improve += 1
            marker = ""

        recall_str = " ".join(f"{k}={v:.2f}" for k, v in per_class_recall.items())
        print(
            f"  epoch {epoch+1}/{EPOCHS}  train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_pr_auc={val_pr_auc:.4f}  "
            f"hard_recall_mean={composite_score:.4f} [{recall_str}]{marker}"
        )
        if epochs_no_improve >= PATIENCE:
            print(f"  early stopping (no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT_DIR}/gru_v2_model.pt")
    print(f"  saved BEST model (composite hard-class recall={best_score:.4f})")

    with open(f"{OUT_DIR}/gru_v2_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved model + history to {OUT_DIR}/")
    return model


if __name__ == "__main__":
    train()