"""
One-off diagnostic: why doesn't impossible_travel's DETECTION explanation
mention geo/speed, when the CLASSIFICATION explanation (same event) does?
Print the raw, unscaled values across the window to see whether geo/speed
is genuinely extreme for this event or not, before accepting either
"real feature-weighting difference" or "remaining bug" as the answer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detection"))
import numpy as np
import pandas as pd

OUT_DIR = "../detection"

va = np.load(f"{OUT_DIR}/val_sequences_v2.npz", allow_pickle=True)  # UNSCALED
X_raw = va["X"]
ytype = va["y_type"]
mask = va["mask"]

FEATURE_COLS = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before", "hour_sin", "hour_cos",
    "rolling_7d_event_count", "rolling_7d_new_resource_count",
    "rolling_30d_new_resource_count", "rolling_7d_offhours_ratio",
]

idx_list = np.where(ytype == "impossible_travel")[0]
idx = idx_list[len(idx_list) // 2]  # same example used in explain_alert.py

print(f"Event index: {idx}, true label: impossible_travel\n")
print("Raw (unscaled) values across the 15-step window (only real/non-padded steps shown):")
print(f"{'step':>5}{'geo_distance_km':>18}{'travel_speed_kmh':>18}{'session_duration':>18}"
      f"{'resource_seen_before':>22}{'time_since_last_min':>22}")

for t in range(X_raw.shape[1]):
    if mask[idx, t] == 0:
        continue
    row = X_raw[idx, t]
    geo = row[FEATURE_COLS.index("geo_distance_from_prev_km")]
    speed = row[FEATURE_COLS.index("implied_travel_speed_kmh")]
    sess = row[FEATURE_COLS.index("session_duration")]
    seen_before = row[FEATURE_COLS.index("resource_seen_before")]
    gap = row[FEATURE_COLS.index("time_since_last_access_min")]
    steps_ago = X_raw.shape[1] - 1 - t
    print(f"{steps_ago:>5}{geo:>18.2f}{speed:>18.2f}{sess:>18.2f}{seen_before:>22.0f}{gap:>22.2f}")

print("\n('step' = events ago from the current/flagged event, 0 = current)")
print("\nFor reference, typical/normal ranges (approx, from earlier RobustScaler diagnostics):")
print("  geo_distance_from_prev_km: near 0 for normal same-location logins")
print("  implied_travel_speed_kmh: near 0 normally; >900 km/h is physically implausible (flight speed)")