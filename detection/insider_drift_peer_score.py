"""
Deliverable 3 — Insider drift: peer-group contextual layer (post-processing)

DESIGN (per the reasoning above, not a retrain):
The core detector (ensemble, alpha=0.10) is UNCHANGED and stays the system
of record for the primary recall/precision/budget numbers. This script adds
a SEPARATE, secondary signal on top, using two things that already exist:

  1. The detector's own raw score (recomputed here, not retrained)
  2. entity_type, already in your schema (user / service_account / edge_device)

PEER-GROUP SCORE:
For each event, compare the entity's own long-term resource-footprint growth
(rolling_30d_new_resource_count — chosen because the brief's own definition
of insider_drift is "slowly expanding privilege or resource footprint") to
the AVERAGE for that entity_type, computed from TRAIN data only (never val/
test) using only events labeled "normal" — so the peer baseline itself
isn't contaminated by attacks.

    peer_score = entity's rolling_30d_new_resource_count / peer_group_average

SECONDARY FLAG (not part of the primary alert budget):
An event is flagged "Potential Insider Drift" if BOTH:
  - the core detector's raw score is elevated (top 10% — suspicious, but not
    necessarily in the top-1% alert budget)
  - peer_score exceeds a threshold (default 1.5x the peer-type average)

This is evaluated and reported SEPARATELY from the main budget-based recall/
precision — it is explicitly not competing for the same 215 alert slots.
"""
import numpy as np
import pandas as pd
import torch
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALPHA = 0.10
ELEVATED_PERCENTILE = 90     # "worth a second look" bar for the core detector
PEER_SCORE_THRESHOLD = 1.5   # entity doing 1.5x+ their peer-type average
PEER_FEATURE = "rolling_30d_new_resource_count"


def load_a():
    va = np.load(f"{OUT_DIR}/val_sequences_scaled.npz", allow_pickle=True)
    return va["X"], va["y"], va["mask"], va["y_type"]


def load_b_scaled():
    d = np.load(f"{OUT_DIR}/val_sequences_v2_scaled.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    return X, mask


if __name__ == "__main__":
    # --- recompute the existing, unchanged core detector score ---
    X_a, y, mask_a, y_type = load_a()
    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    y_pred_a, _ = model_a.forward(X_a, mask_a)

    X_b, mask_b = load_b_scaled()
    model_b = StatefulGRU(input_size=X_b.shape[2], hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    with torch.no_grad():
        y_pred_b = torch.sigmoid(model_b(X_b, mask_b)).numpy()

    raw_score = ALPHA * y_pred_a + (1 - ALPHA) * y_pred_b
    elevated_threshold = np.percentile(raw_score, ELEVATED_PERCENTILE)
    elevated = raw_score >= elevated_threshold

    # --- build peer-type baseline from TRAIN, normal events only ---
    train_df = pd.read_csv(f"../train_longterm.csv")
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    normal_train = train_df[train_df["label"] == "normal"]
    peer_avg = normal_train.groupby("entity_type")[PEER_FEATURE].mean().to_dict()
    print("Peer-group averages (train, normal events only):")
    for et, avg in peer_avg.items():
        print(f"  {et}: {avg:.2f}")
    print()

    # --- apply to val: need per-row entity_type + PEER_FEATURE, same row
    # order as the sequence files (both built by grouping entity_id then
    # sorting timestamp — same order guaranteed by seq_prep_v2.py) ---
    val_df = pd.read_csv(f"../val_longterm.csv")
    val_df["timestamp"] = pd.to_datetime(val_df["timestamp"])
    val_df = val_df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    if len(val_df) != len(raw_score):
        raise SystemExit(
            f"ROW COUNT MISMATCH: val_longterm.csv has {len(val_df)} rows but "
            f"the sequence arrays have {len(raw_score)}. This script assumes "
            f"identical row order between the two (both sorted by entity_id "
            f"then timestamp) — do not trust the peer scores below until "
            f"this is resolved."
        )

    entity_types = val_df["entity_type"].values
    peer_feature_values = val_df[PEER_FEATURE].values
    default_avg = normal_train[PEER_FEATURE].mean()  # fallback for unseen entity_types
    peer_group_avg = np.array([peer_avg.get(et, default_avg) for et in entity_types])
    peer_score = peer_feature_values / np.maximum(peer_group_avg, 1e-6)

    secondary_flag = elevated & (peer_score >= PEER_SCORE_THRESHOLD)

    print(f"Elevated (top {100-ELEVATED_PERCENTILE}% of core detector score): {elevated.sum()} events")
    print(f"Secondary 'Potential Insider Drift' flag: {secondary_flag.sum()} events\n")

    print("--- Secondary flag performance (separate from primary alert budget) ---")
    for t in sorted(np.unique(y_type)):
        m = y_type == t
        n = m.sum()
        flagged = secondary_flag[m].sum()
        print(f"  {t}: {flagged}/{n} flagged ({flagged/max(n,1):.1%})")

    print("\nNote: this is a SECONDARY signal, evaluated independently of the")
    print("primary top-1% alert budget. It does not change, compete with, or")
    print("replace the core detector's recall/precision numbers reported")
    print("elsewhere — it demonstrates a UEBA-style contextual layer for")
    print("ambiguous edge cases, per the brief's own framing of insider_drift.")