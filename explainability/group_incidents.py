"""
Incident grouping for the analyst dashboard.

Problem: the raw alert feed contains many near-duplicate alerts fired
in rapid succession for the same entity + same attack type (e.g. a single
brute-force burst can trigger 20-60 individual alert records within a
few minutes). Shown raw, these bury every other incident in the queue.

Rule: consecutive alerts are collapsed into ONE incident when they share
the same entity_id AND the same predicted_attack_type AND are no more
than GROUP_WINDOW_MINUTES apart from the previous alert in that same
running group (chained window, not "all within N minutes of the first
alert" -- this correctly merges long bursts like the 60-event usr_00102
case where the burst itself spans more than one window if you anchor
only to the first event).

Representative score for a group = MAX risk_score within the group
(the peak severity is what would have first caught an analyst's eye;
it's also simpler to justify than an average).

Output: dashboard_data_grouped.json
  - summary: unchanged, passed through
  - incidents: one row per grouped incident, sorted by risk_score desc
      - alert_ids: list of original alert ids folded into this incident
      - event_count: how many raw alerts were folded in
      - first_seen / last_seen: time range of the burst
      - representative fields (risk_score, predicted_attack_type, etc.)
        taken from the alert with the MAX risk_score in the group
      - all_true_labels: set of distinct true_label values in the group
        (almost always one value, but kept as a list for safety/audit)
  - raw_alerts: the original, ungrouped alert list, untouched
      (kept so ground-truth analysis / the report can still work off
      every individual event if needed)

Usage:
    python3 group_incidents.py dashboard_data.json dashboard_data_grouped.json
"""

import json
import sys
from datetime import datetime, timedelta
from collections import defaultdict

GROUP_WINDOW_MINUTES = 5


def parse_ts(ts_str):
    """Parse the timestamp formats seen in the data (with/without microseconds)."""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {ts_str!r}")


def group_alerts(alerts, window_minutes=GROUP_WINDOW_MINUTES):
    """
    Chained-window grouping.

    Alerts are first sorted by (entity_id, predicted_attack_type, timestamp).
    Within each (entity_id, predicted_attack_type) bucket, we walk alerts in
    time order and start a new group whenever the gap since the PREVIOUS
    alert in the current group exceeds `window_minutes`. This lets a long
    burst (e.g. 60 events over 3 minutes with sub-second gaps) stay as one
    group, while two genuinely separate incidents hours apart for the same
    entity/type correctly become two groups.
    """
    window = timedelta(minutes=window_minutes)

    buckets = defaultdict(list)
    for alert in alerts:
        key = (alert["entity_id"], alert["predicted_attack_type"])
        buckets[key].append(alert)

    groups = []
    for key, bucket_alerts in buckets.items():
        bucket_alerts.sort(key=lambda a: parse_ts(a["timestamp"]))

        current_group = [bucket_alerts[0]]
        for alert in bucket_alerts[1:]:
            prev_ts = parse_ts(current_group[-1]["timestamp"])
            this_ts = parse_ts(alert["timestamp"])
            if this_ts - prev_ts <= window:
                current_group.append(alert)
            else:
                groups.append(current_group)
                current_group = [alert]
        groups.append(current_group)

    return groups


def build_incident(group):
    """Collapse one group of alerts into a single incident record."""
    representative = max(group, key=lambda a: a["risk_score"])

    timestamps = [parse_ts(a["timestamp"]) for a in group]
    first_seen = min(timestamps)
    last_seen = max(timestamps)

    true_labels = sorted({a.get("true_label") for a in group if "true_label" in a})
    entity_types = sorted({a.get("entity_type") for a in group if "entity_type" in a})

    incident = {
        "incident_id": f"incident_{representative['id']}",
        "entity_id": representative["entity_id"],
        "entity_type": entity_types[0] if len(entity_types) == 1 else entity_types,
        "predicted_attack_type": representative["predicted_attack_type"],
        "risk_score": representative["risk_score"],
        "risk_threshold": representative.get("risk_threshold"),
        "classification_confidence": representative.get("classification_confidence"),
        "cold_start": representative.get("cold_start"),
        "history_length": representative.get("history_length"),
        "detection_reasons": representative.get("detection_reasons", []),
        "classification_reasons": representative.get("classification_reasons", []),
        "entity_history": representative.get("entity_history", []),
        "event_count": len(group),
        "first_seen": first_seen.isoformat(sep=" "),
        "last_seen": last_seen.isoformat(sep=" "),
        "representative_alert_id": representative["id"],
        "alert_ids": [a["id"] for a in group],
        "all_true_labels": true_labels,
    }
    return incident


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 group_incidents.py <input.json> <output.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, "r") as f:
        data = json.load(f)

    raw_alerts = data["alerts"]
    groups = group_alerts(raw_alerts)
    incidents = [build_incident(g) for g in groups]
    incidents.sort(key=lambda inc: inc["risk_score"], reverse=True)

    output = {
        "summary": data.get("summary", {}),
        "grouping_meta": {
            "rule": (
                f"Consecutive alerts with the same entity_id and "
                f"predicted_attack_type are merged into one incident if "
                f"no more than {GROUP_WINDOW_MINUTES} minutes separate "
                f"consecutive alerts within the burst (chained window)."
            ),
            "representative_score": "max risk_score within the group",
            "raw_alert_count": len(raw_alerts),
            "grouped_incident_count": len(incidents),
            "reduction_pct": round(
                100 * (1 - len(incidents) / len(raw_alerts)), 1
            ) if raw_alerts else 0,
        },
        "incidents": incidents,
        "raw_alerts": raw_alerts,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Raw alerts:      {len(raw_alerts)}")
    print(f"Grouped incidents: {len(incidents)}")
    print(f"Reduction:       {output['grouping_meta']['reduction_pct']}%")
    print(f"Written to:      {output_path}")


if __name__ == "__main__":
    main()