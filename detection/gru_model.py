"""
================================================================================
Deliverable 3 — Step 2: Sequence-Aware GRU Detection Model (baseline version)
================================================================================
A GRU implemented directly in NumPy (vectorized across the batch dimension,
looped only over the 15 timesteps — this is what deep learning frameworks do
internally, just written explicitly here). No external DL framework dependency
means no install risk and a fully transparent, explainable model for the report.

Architecture: single-layer GRU -> dense -> sigmoid, trained with Adam + BPTT,
on the causal windows built in seq_prep.py.

This is the SIMPLE / fixed-window baseline detector (per the plan: build this
first, then the stateful+EWMA version, then compare both on val/test).

NOTE: OUT_DIR changed from the original hardcoded /mnt/user-data/outputs/detection
sandbox path to "." (current folder) for local use.
================================================================================
"""
import numpy as np
import json
import os

SEED = 42
np.random.seed(SEED)

OUT_DIR = "."
HIDDEN_SIZE = 24
LR = 0.01
BATCH_SIZE = 256
EPOCHS = 12
CLIP_NORM = 5.0

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def tanh(x):
    return np.tanh(x)

class NumpyGRU:
    """Single-layer GRU + dense sigmoid head, batch-vectorized, trained via BPTT."""

    def __init__(self, input_size, hidden_size, seed=SEED):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale_in = np.sqrt(1.0 / input_size)
        scale_hid = np.sqrt(1.0 / hidden_size)

        # Gate weights: z (update), r (reset), h (candidate)
        self.Wz = rng.uniform(-scale_in, scale_in, (input_size, hidden_size))
        self.Uz = rng.uniform(-scale_hid, scale_hid, (hidden_size, hidden_size))
        self.bz = np.zeros(hidden_size)

        self.Wr = rng.uniform(-scale_in, scale_in, (input_size, hidden_size))
        self.Ur = rng.uniform(-scale_hid, scale_hid, (hidden_size, hidden_size))
        self.br = np.zeros(hidden_size)

        self.Wh = rng.uniform(-scale_in, scale_in, (input_size, hidden_size))
        self.Uh = rng.uniform(-scale_hid, scale_hid, (hidden_size, hidden_size))
        self.bh = np.zeros(hidden_size)

        self.Wo = rng.uniform(-scale_hid, scale_hid, (hidden_size, 1))
        self.bo = np.zeros(1)

        self.params = ["Wz", "Uz", "bz", "Wr", "Ur", "br", "Wh", "Uh", "bh", "Wo", "bo"]
        # Adam optimizer state
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def forward(self, X, mask):
        """
        X: (batch, T, input_size), mask: (batch, T) with 1=real, 0=padding
        Returns: y_pred (batch,), and cache needed for backward pass.
        """
        batch, T, _ = X.shape
        H = self.hidden_size
        h = np.zeros((batch, H))
        cache = {"z": [], "r": [], "hcand": [], "h": [h.copy()], "x": []}

        for t in range(T):
            x_t = X[:, t, :]
            m_t = mask[:, t:t+1]  # (batch,1)

            z = sigmoid(x_t @ self.Wz + h @ self.Uz + self.bz)
            r = sigmoid(x_t @ self.Wr + h @ self.Ur + self.br)
            hcand = tanh(x_t @ self.Wh + (r * h) @ self.Uh + self.bh)
            h_new = (1 - z) * h + z * hcand

            # padded timesteps: carry forward previous h unchanged (no update)
            h = m_t * h_new + (1 - m_t) * h

            cache["z"].append(z); cache["r"].append(r)
            cache["hcand"].append(hcand); cache["x"].append(x_t)
            cache["h"].append(h.copy())

        logits = h @ self.Wo + self.bo
        y_pred = sigmoid(logits).ravel()
        cache["y_pred"] = y_pred
        cache["mask"] = mask
        return y_pred, cache

    def backward(self, cache, y_true):
        batch = y_true.shape[0]
        H = self.hidden_size
        T = len(cache["z"])

        grads = {p: np.zeros_like(getattr(self, p)) for p in self.params}

        # Loss: binary cross-entropy. dL/dlogits = (y_pred - y_true)/batch
        y_pred = cache["y_pred"]
        dlogits = (y_pred - y_true).reshape(-1, 1) / batch  # (batch,1)
        h_final = cache["h"][-1]

        grads["Wo"] = h_final.T @ dlogits
        grads["bo"] = dlogits.sum(axis=0)
        dh_next = dlogits @ self.Wo.T  # (batch, H)

        for t in reversed(range(T)):
            h_prev = cache["h"][t]
            z, r, hcand, x_t = cache["z"][t], cache["r"][t], cache["hcand"][t], cache["x"][t]
            m_t = cache["mask"][:, t:t+1]

            dh = dh_next * m_t  # padded steps don't propagate gradient

            dz = dh * (hcand - h_prev)
            dhcand = dh * z
            dh_prev_from_update = dh * (1 - z)

            dhcand_pre = dhcand * (1 - hcand ** 2)  # tanh'
            grads["Wh"] += x_t.T @ dhcand_pre
            grads["Uh"] += (r * h_prev).T @ dhcand_pre
            grads["bh"] += dhcand_pre.sum(axis=0)

            dr_h = dhcand_pre @ self.Uh.T
            dr = dr_h * h_prev
            dh_prev_from_r = dr_h * r

            dz_pre = dz * z * (1 - z)  # sigmoid'
            grads["Wz"] += x_t.T @ dz_pre
            grads["Uz"] += h_prev.T @ dz_pre
            grads["bz"] += dz_pre.sum(axis=0)
            dh_prev_from_z = dz_pre @ self.Uz.T

            dr_pre = dr * r * (1 - r)
            grads["Wr"] += x_t.T @ dr_pre
            grads["Ur"] += h_prev.T @ dr_pre
            grads["br"] += dr_pre.sum(axis=0)
            dh_prev_from_rgate = dr_pre @ self.Ur.T

            dh_next = (dh_prev_from_update + dh_prev_from_r +
                       dh_prev_from_z + dh_prev_from_rgate)
            # unmasked (padded) timesteps: gradient just passes straight through
            dh_next = m_t * dh_next + (1 - m_t) * dh

        return grads

    def clip_grads(self, grads):
        total_norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        if total_norm > CLIP_NORM:
            scale = CLIP_NORM / (total_norm + 1e-8)
            for k in grads:
                grads[k] *= scale
        return grads

    def adam_step(self, grads, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def save(self, path):
        state = {p: getattr(self, p) for p in self.params}
        state["hidden_size"] = self.hidden_size
        state["input_size"] = self.input_size
        np.savez(path, **state)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(int(d["input_size"]), int(d["hidden_size"]))
        for p in model.params:
            setattr(model, p, d[p])
        return model


def bce_loss(y_pred, y_true, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_gru(X, y, mask, X_val, y_val, mask_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    n = X.shape[0]
    model = NumpyGRU(input_size=X.shape[2], hidden_size=HIDDEN_SIZE)

    # class weighting: attacks are ~2% of data, upweight positive class in loss
    # by oversampling anomalies each epoch so the GRU doesn't just predict "normal"
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    print(f"  train: {len(pos_idx)} anomalies, {len(neg_idx)} normal")

    history = []
    for epoch in range(epochs):
        # balanced-ish batches: oversample positives so each batch has a
        # meaningful fraction of attacks (helps a lot under 2% imbalance)
        rng = np.random.RandomState(epoch)
        n_pos_per_batch = max(1, batch_size // 8)
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            if len(pos_idx) > 0:
                extra_pos = rng.choice(pos_idx, size=n_pos_per_batch, replace=True)
                batch_idx = np.concatenate([batch_idx, extra_pos])

            Xb, yb, mb = X[batch_idx], y[batch_idx], mask[batch_idx]
            y_pred, cache = model.forward(Xb, mb)
            loss = bce_loss(y_pred, yb)
            grads = model.backward(cache, yb)
            grads = model.clip_grads(grads)
            model.adam_step(grads)

            epoch_loss += loss
            n_batches += 1

        # validation loss (no oversampling — real distribution)
        val_pred, _ = model.forward(X_val, mask_val)
        val_loss = bce_loss(val_pred, y_val)
        avg_train_loss = epoch_loss / n_batches
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})
        print(f"  epoch {epoch+1}/{epochs}  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

    return model, history


if __name__ == "__main__":
    print("Loading sequence data...")
    tr = np.load(f"{OUT_DIR}/train_sequences_scaled.npz", allow_pickle=True)
    va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)

    X_train, y_train, mask_train = tr["X"], tr["y"], tr["mask"]
    X_val, y_val, mask_val = va["X"], va["y"], va["mask"]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print("\nTraining fixed-window GRU (baseline sequence model)...")
    model, history = train_gru(X_train, y_train, mask_train, X_val, y_val, mask_val)

    model.save(f"{OUT_DIR}/gru_baseline_model.npz")
    with open(f"{OUT_DIR}/gru_baseline_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved model + training history to {OUT_DIR}/")