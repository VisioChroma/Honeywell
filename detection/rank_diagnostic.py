"""
Diagnostic before investing more effort in insider_drift / low_and_slow_exfil:
recall-at-budget is a hard cutoff (caught or not) — it can't tell us whether
these classes are "completely unranked" (model has no signal at all) or
"ranked meaningfully above normal, just not high enough to clear the top-1%
budget" (model has partial signal, just not enough to win the competition
for scarce alert slots). These need very different fixes.
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
y_type = d["y_type"]
model_b = StatefulGRU(input_size=X_b.shape[2], hidden_size=32)
model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
model_b.eval()
with torch.no_grad():
    y_pred_b = torch.sigmoid(model_b(X_b, mask_b)).numpy()

raw_score = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b
percentile_rank = raw_score.argsort().argsort() / (len(raw_score) - 1) * 100

n_total = len(raw_score)
budget = max(1, int(n_total * 0.01))
budget_percentile_cutoff = 100 * (1 - budget / n_total)
print(f"Alert budget cutoff sits at percentile: {budget_percentile_cutoff:.2f}\n")

print(f"{'attack type':<22}{'n':>6}{'median pctile':>16}{'p25 pctile':>13}{'max pctile':>13}{'>cutoff':>10}")
for t in sorted(np.unique(y_type)):
    m = y_type == t
    n = m.sum()
    if n == 0:
        continue
    ranks = percentile_rank[m]
    above_cutoff = (ranks >= budget_percentile_cutoff).sum()
    print(f"{t:<22}{n:>6}{np.median(ranks):>15.1f}%{np.percentile(ranks,25):>12.1f}%"
          f"{ranks.max():>12.1f}%{above_cutoff:>10}")

print("\nInterpretation guide:")
print("  - median pctile near 99-100: model ranks this class near the very top -> strong signal")
print("  - median pctile mid-range (e.g. 70-90) but max near 100: PARTIAL signal, some events")
print("    rank highly but most don't -> worth more feature work, not a lost cause")
print("  - median pctile near 50 (same as random/normal): NO signal currently -> feature")
print("    engineering needed, more loss-weighting alone won't fix this")