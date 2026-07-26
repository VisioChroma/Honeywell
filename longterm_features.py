"""
================================================================================
Deliverable 3 — Step 4b: Long-Term Memory Features, v2 (adds TREND features)
================================================================================
The original 4 features (rolling_7d_event_count, rolling_7d_new_resource_count,
rolling_30d_new_resource_count, rolling_7d_offhours_ratio) capture LEVEL —
how much activity happened. What's missing for low_and_slow_exfil specifically
is TREND — whether this week looks different from the entity's own recent
past. "Gradual buildup" is a slope, not a magnitude; an entity that's always
had high resource-diversity looks identical to one that's slowly ramping up,
under level features alone.

Diagnostic evidence this is worth doing (rank_diagnostic.py, this session):
low_and_slow_exfil's median percentile rank is 98.9%, and the alert budget
cutoff is 99.0% — a razor-thin margin. This is the signature of "real signal,
narrowly missing the ranking competition," which is exactly what trend
features are suited to fix (per the industry UEBA research: "Instead of
'How many files today?' they ask 'Is file access increasing over time?'").

TWO NEW FEATURES ADDED (kept to 2, deliberately — small, defensible addition
rather than a large speculative feature dump given limited time remaining):

1. growth_rate_new_resources_7d
   = (this week's new-resource touches - prior week's) / (prior week's + 1)
   Positive and large -> resource diversity is accelerating, not just elevated.

2. offhours_ratio_trend_7d
   = this week's off-hours ratio - prior week's off-hours ratio
   Positive -> entity is shifting toward off-hours access relative to their
   OWN recent baseline, not just "sometimes works late" (which level alone
   can't distinguish from a genuine drift toward off-hours behavior).

Both computed the same way as the existing 4: per-entity, time-indexed
pandas rolling windows, no leakage (each row only looks backward from its
own timestamp).
================================================================================
"""
import numpy as np
import pandas as pd

def add_longterm_features(df):
    df = df.sort_values(["entity_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    all_rows = []
    for eid, grp in df.groupby("entity_id", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        ts = grp["timestamp"]
        hour = ts.dt.hour
        is_offhours = ((hour < 6) | (hour > 22)).astype(int)

        seen_resources = set()
        new_resource_flags = []
        for res in grp["resource_accessed"]:
            new_resource_flags.append(int(res not in seen_resources))
            seen_resources.add(res)
        new_resource_flags = pd.Series(new_resource_flags, index=grp.index)

        s = pd.Series(1, index=ts)
        rolling_7d_count = s.rolling("7D").count().values
        rolling_14d_count = s.rolling("14D").count().values
        rolling_prior7d_count = rolling_14d_count - rolling_7d_count  # days -14 to -7

        new_res_series = pd.Series(new_resource_flags.values, index=ts)
        rolling_7d_new = new_res_series.rolling("7D").sum().values
        rolling_14d_new = new_res_series.rolling("14D").sum().values
        rolling_prior7d_new = rolling_14d_new - rolling_7d_new
        rolling_30d_new = new_res_series.rolling("30D").sum().values

        offhours_series = pd.Series(is_offhours.values, index=ts)
        rolling_7d_offhours_sum = offhours_series.rolling("7D").sum().values
        rolling_14d_offhours_sum = offhours_series.rolling("14D").sum().values
        rolling_prior7d_offhours_sum = rolling_14d_offhours_sum - rolling_7d_offhours_sum

        rolling_7d_offhours_ratio = rolling_7d_offhours_sum / np.maximum(rolling_7d_count, 1)
        rolling_prior7d_offhours_ratio = (
            rolling_prior7d_offhours_sum / np.maximum(rolling_prior7d_count, 1)
        )

        # --- existing 4 features (unchanged) ---
        grp["rolling_7d_event_count"] = rolling_7d_count
        grp["rolling_7d_new_resource_count"] = rolling_7d_new
        grp["rolling_30d_new_resource_count"] = rolling_30d_new
        grp["rolling_7d_offhours_ratio"] = rolling_7d_offhours_ratio

        # --- NEW: trend features ---
        growth_rate_new_resources_7d = (
            (rolling_7d_new - rolling_prior7d_new) / np.maximum(rolling_prior7d_new + 1, 1)
        )
        offhours_ratio_trend_7d = rolling_7d_offhours_ratio - rolling_prior7d_offhours_ratio

        grp["growth_rate_new_resources_7d"] = growth_rate_new_resources_7d
        grp["offhours_ratio_trend_7d"] = offhours_ratio_trend_7d

        all_rows.append(grp)

    return pd.concat(all_rows, ignore_index=True)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        df = pd.read_csv(f"{split}.csv")
        df_with_lt = add_longterm_features(df)
        out_path = f"{split}_longterm.csv"
        df_with_lt.to_csv(out_path, index=False)
        print(f"  saved {out_path}, shape={df_with_lt.shape}")

    print("\nDone. Columns added: rolling_7d_event_count, "
          "rolling_7d_new_resource_count, rolling_30d_new_resource_count, "
          "rolling_7d_offhours_ratio, growth_rate_new_resources_7d, "
          "offhours_ratio_trend_7d")