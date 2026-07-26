"""
Quick diagnostic before picking a third gate threshold — print the actual
score distribution so we choose a percentile that means something, instead
of guessing again. v1 (top-1% gate) was too loose, v2 (median gate) was too
strict because the median is 0 in a heavily skewed score distribution.
"""
import numpy as np
import torch
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALPHA = 0.10

va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)
model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
y_pred_a, _ = model_a.forward(va["X"], va["mask"])

d = np.load(f"{OUT_DIR}/val_sequences_v2_scaled.npz", allow_pickle=True)
X_b = torch.tensor(d["X"], dtype=torch.float32)
mask_b = torch.tensor(d["mask"], dtype=torch.float32)
model_b = StatefulGRU(input_size=X_b.shape[2], hidden_size=32)
model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
model_b.eval()
with torch.no_grad():
    y_pred_b = torch.sigmoid(model_b(X_b, mask_b)).numpy()

raw_score = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b
y = va["y"]

print("Score distribution (all events):")
for p in [10, 25, 50, 60, 70, 75, 80, 85, 90, 92, 95, 97, 98, 99, 99.5]:
    val = np.percentile(raw_score, p)
    print(f"  p{p}: {val:.5f}")

print(f"\nFraction of events with score == 0.0: {(raw_score == 0).mean():.1%}")
print(f"Fraction of events with score < 0.001: {(raw_score < 0.001).mean():.1%}")
print(f"Fraction of events with score < 0.01: {(raw_score < 0.01).mean():.1%}")

print(f"\nScore stats for TRUE ANOMALIES only (y==1):")
anom_scores = raw_score[y == 1]
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: {np.percentile(anom_scores, p):.5f}")
print(f"  min: {anom_scores.min():.5f}, max: {anom_scores.max():.5f}")

print(f"\nScore stats for NORMAL events only (y==0):")
norm_scores = raw_score[y == 0]
for p in [50, 90, 95, 99, 99.9]:
    print(f"  p{p}: {np.percentile(norm_scores, p):.5f}")