"""
================================================================================
Deliverable 3/6 — Streaming Replay Simulation (real-time feasibility proof)
================================================================================
Replays test.csv events in TRUE GLOBAL CHRONOLOGICAL ORDER (interleaved across
all entities, exactly as a live system would receive them — NOT the
entity-grouped order used for training/batch evaluation), scoring one event
at a time through the full locked pipeline (Ensemble A+B, alpha=0.10, then
the Deliverable 4 classifier for flagged alerts), measuring real per-event
latency.

WHAT THIS PROVES: per-event scoring is fast enough for real-time use, and
streaming (one-at-a-time) scores exactly match batch scores (no hidden
future-peeking or batch-only behavior).

WHAT THIS DOES NOT PROVE (explicit, documented limitation — not hidden):
the 7d/30d rolling features and the 15-event window were precomputed in
batch (pandas .rolling() over full history) by seq_prep.py/longterm_features.py,
not computed incrementally as each event "arrives". A production system
would need an online feature engine (per-entity ring buffer for the window,
incremental rolling-window counters for 7d/30d stats) — that is real,
unbuilt engineering, called out here as documented future work rather than
silently assumed solved.
================================================================================
"""
import time
import numpy as np
import torch
import joblib
import pandas as pd
from gru_model import NumpyGRU
from gru_pytorch_v2 import StatefulGRU

OUT_DIR = "."
ALPHA = 0.10
ALERT_BUDGET_FRAC = 0.01


def load_all():
    va = np.load(f"{OUT_DIR}/test_sequences_scaled.npz", allow_pickle=True)
    X_a, y_a, mask_a, ytype_a = va["X"], va["y"], va["mask"], va["y_type"]
    ts_a = va["timestamps"]

    vb = np.load(f"{OUT_DIR}/test_sequences_v2_scaled.npz", allow_pickle=True)
    X_b_np, y_b, mask_b_np, ytype_b = vb["X"], vb["y"], vb["mask"], vb["y_type"]
    ts_b = vb["timestamps"]

    if len(y_a) != len(y_b) or not np.array_equal(y_a, y_b) or not np.array_equal(ytype_a, ytype_b):
        raise SystemExit("ALIGNMENT CHECK FAILED — cannot proceed with streaming replay.")
    if not np.array_equal(ts_a, ts_b):
        raise SystemExit("TIMESTAMP MISMATCH between A and B sequence files — cannot proceed.")

    print(f"Alignment check passed: {len(y_a)} rows.\n")
    return X_a, X_b_np, mask_a, mask_b_np, y_a, ytype_a, ts_a


def main():
    X_a, X_b_np, mask_a, mask_b_np, y, y_type, timestamps = load_all()
    n = len(y)

    model_a = NumpyGRU.load(f"{OUT_DIR}/gru_baseline_model.npz")
    model_b = StatefulGRU(input_size=X_b_np.shape[2], hidden_size=32)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/gru_v2_model.pt"))
    model_b.eval()
    clf = joblib.load("anomaly_classifier.joblib")

    FEATURE_COLS = [
        "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
        "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
        "failed_auth_count_5min", "resource_diversity_recent10",
        "distinct_entities_per_source_5min", "time_since_last_access_min",
        "resource_seen_before", "rolling_7d_event_count",
        "rolling_7d_new_resource_count", "rolling_30d_new_resource_count",
        "rolling_7d_offhours_ratio",
    ]
    FULL_FEATURES = FEATURE_COLS + ["entity_type_user", "entity_type_service_account",
                                      "entity_type_edge_device"]

    df = pd.read_csv("../test_longterm.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")  # CHANGED: handle mixed micro-second/no-microsecond timestamp formats
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["entity_type_user"] = (df["entity_type"] == "user").astype(int)
    df["entity_type_service_account"] = (df["entity_type"] == "service_account").astype(int)
    df["entity_type_edge_device"] = (df["entity_type"] == "edge_device").astype(int)
    if len(df) != n or not np.array_equal(df["label"].values, y_type):
        raise SystemExit("CSV alignment check FAILED — cannot map classifier features to sequence rows.")

    # --- batch scores (ground truth to compare streaming scores against) ---
    y_pred_a_batch, _ = model_a.forward(X_a, mask_a)
    X_b_torch = torch.tensor(X_b_np, dtype=torch.float32)
    mask_b_torch = torch.tensor(mask_b_np, dtype=torch.float32)
    with torch.no_grad():
        y_pred_b_batch = torch.sigmoid(model_b(X_b_torch, mask_b_torch)).numpy()
    batch_ensemble = ALPHA * y_pred_a_batch + (1 - ALPHA) * y_pred_b_batch

    # --- global chronological order (TRUE streaming order, not entity-grouped) ---
    global_order = np.argsort(pd.to_datetime(timestamps.astype(str), format="mixed"))  # CHANGED: was pd.to_datetime(timestamps.astype(str))
    print(f"Replaying {n} events in true global chronological order...\n")

    detection_latencies_ms = []
    classification_latencies_ms = []
    streaming_scores = np.zeros(n)
    n_alerts_fired = 0

    # alert threshold: same top-1% budget as everywhere else, computed on batch
    budget = max(1, int(n * ALERT_BUDGET_FRAC))
    threshold = np.quantile(batch_ensemble, 1 - budget / n)

    for idx in global_order:
        t0 = time.perf_counter()

        x_a_single = X_a[idx:idx+1]
        m_a_single = mask_a[idx:idx+1]
        score_a, _ = model_a.forward(x_a_single, m_a_single)

        x_b_single = torch.tensor(X_b_np[idx:idx+1], dtype=torch.float32)
        m_b_single = torch.tensor(mask_b_np[idx:idx+1], dtype=torch.float32)
        with torch.no_grad():
            score_b = torch.sigmoid(model_b(x_b_single, m_b_single)).numpy()

        score = ALPHA * score_a[0] + (1 - ALPHA) * score_b[0]
        streaming_scores[idx] = score

        t1 = time.perf_counter()
        detection_latencies_ms.append((t1 - t0) * 1000)

        if score >= threshold:
            n_alerts_fired += 1
            t2 = time.perf_counter()
            row_features = df.iloc[[idx]][FULL_FEATURES].fillna(0).values
            _ = clf.predict(row_features)
            t3 = time.perf_counter()
            classification_latencies_ms.append((t3 - t2) * 1000)

    # --- correctness check: streaming scores must match batch scores ---
    scores_match = np.allclose(streaming_scores, batch_ensemble, atol=1e-5)
    max_diff = np.max(np.abs(streaming_scores - batch_ensemble))

    det_lat = np.array(detection_latencies_ms)
    print("=" * 60)
    print("STREAMING REPLAY RESULTS")
    print("=" * 60)
    print(f"Events replayed: {n}")
    print(f"Streaming scores match batch scores: {scores_match} (max diff: {max_diff:.2e})")
    print()
    print("Detection latency per event (ensemble A+B):")
    print(f"  mean:   {det_lat.mean():.3f} ms")
    print(f"  median: {np.median(det_lat):.3f} ms")
    print(f"  p95:    {np.percentile(det_lat, 95):.3f} ms")
    print(f"  p99:    {np.percentile(det_lat, 99):.3f} ms")
    print(f"  max:    {det_lat.max():.3f} ms")
    print(f"  implied throughput: {1000/det_lat.mean():.0f} events/sec (single-threaded, sequential)")
    print()
    if classification_latencies_ms:
        cls_lat = np.array(classification_latencies_ms)
        print(f"Classification latency (only for {n_alerts_fired} flagged alerts):")
        print(f"  mean:   {cls_lat.mean():.3f} ms")
        print(f"  p95:    {np.percentile(cls_lat, 95):.3f} ms")
        print()
        print(f"Full pipeline latency (detection + classification) for a flagged event:")
        print(f"  mean: {det_lat.mean() + cls_lat.mean():.3f} ms")
    print()
    print("LIMITATION (documented, not hidden): rolling 7d/30d features and the")
    print("15-event window were precomputed in batch, not incrementally. A live")
    print("system needs an online feature engine (per-entity ring buffer + rolling")
    print("counters) — real remaining engineering, not covered by this replay.")


if __name__ == "__main__":
    main()