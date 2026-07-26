
import numpy as np
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_USERS = 400
N_SERVICE_ACCOUNTS = 60
N_EDGE_DEVICES = 140

SIM_DAYS = 42                     # ~6 weeks
SIM_START = datetime(2026, 6, 1)
COLD_START_FRACTION = 0.08
DRIFT_FRACTION = 0.10
ATTACK_INJECTION_RATE = 0.02      # target ~2%, within the 0.5-3% band

RESOURCE_POOL = [f"/resource/{i}" for i in range(1, 61)]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
GEO_REGIONS = [
    ("us-east", 39.0, -77.0), ("us-west", 37.4, -122.1),
    ("eu-west", 51.5, -0.1), ("ap-south", 19.1, 72.9),
    ("ap-southeast", 1.35, 103.8), ("sa-east", -23.5, -46.6),
]

OUT_DIR = "/mnt/user-data/outputs"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def region_ip(_region_name):
    return fake.ipv4_public()

def make_entity_id(entity_type, idx):
    prefix = {"user": "usr", "service_account": "svc", "edge_device": "dev"}[entity_type]
    return f"{prefix}_{idx:05d}"

def _new_row(entity_id, entity_type, ts, geo_name, geo_lat, geo_lon, resource,
             auth_method, auth_success, duration, cmd_seq, device_fp, label, source_ip=None):
    return {
        "entity_id": entity_id, "entity_type": entity_type, "timestamp": ts,
        "geo_region": geo_name, "geo_lat": geo_lat, "geo_lon": geo_lon,
        "source_ip": source_ip if source_ip is not None else region_ip(geo_name),
        "resource_accessed": resource,
        "auth_method": auth_method, "auth_success": auth_success,
        "session_duration": round(duration, 2), "command_sequence": cmd_seq,
        "device_fingerprint": device_fp, "label": label,
    }


def build_entities():
    entities = []
    idx = 0

    def base_profile(entity_type):
        home_region = random.choice(GEO_REGIONS)
        n_habitual_resources = {
            "user": random.randint(4, 12),
            "service_account": random.randint(1, 3),
            "edge_device": random.randint(1, 2),
        }[entity_type]
        habitual_resources = random.sample(RESOURCE_POOL, n_habitual_resources)

        if entity_type == "user":
            active_hour_center = random.choice([9, 10, 13, 14])
            active_hour_spread = 3.5
            sessions_per_day = max(np.random.gamma(shape=3, scale=1.3), 0.3)
            session_duration_mean = random.uniform(4, 45)
            auth_pref = random.choices(AUTH_METHODS, weights=[0.5, 0.3, 0.1, 0.1])[0]
        elif entity_type == "service_account":
            active_hour_center = random.choice(range(24))
            active_hour_spread = 0.6
            sessions_per_day = np.random.uniform(5, 30)
            session_duration_mean = random.uniform(0.2, 3)
            auth_pref = random.choices(AUTH_METHODS, weights=[0.05, 0.55, 0.35, 0.05])[0]
        else:  # edge_device
            active_hour_center = random.choice(range(24))
            active_hour_spread = 1.0
            sessions_per_day = np.random.uniform(2, 12)
            session_duration_mean = random.uniform(0.1, 1.5)
            auth_pref = random.choices(AUTH_METHODS, weights=[0.1, 0.2, 0.6, 0.1])[0]

        return {
            "home_region": home_region,
            "habitual_resources": habitual_resources,
            "active_hour_center": active_hour_center,
            "active_hour_spread": active_hour_spread,
            "sessions_per_day": sessions_per_day,
            "session_duration_mean": session_duration_mean,
            "auth_pref": auth_pref,
            "device_fp": f"{fake.mac_address()}|{random.choice(['linux','windows','ios','android','firmware-2.3'])}",
            "home_ip": fake.ipv4_public(),
        }

    for _ in range(N_USERS):
        eid = make_entity_id("user", idx); idx += 1
        entities.append({"entity_id": eid, "entity_type": "user", **base_profile("user")})
    for _ in range(N_SERVICE_ACCOUNTS):
        eid = make_entity_id("service_account", idx); idx += 1
        entities.append({"entity_id": eid, "entity_type": "service_account", **base_profile("service_account")})
    for _ in range(N_EDGE_DEVICES):
        eid = make_entity_id("edge_device", idx); idx += 1
        entities.append({"entity_id": eid, "entity_type": "edge_device", **base_profile("edge_device")})

    entities_df = pd.DataFrame(entities)
    cold_start_ids = set(entities_df.sample(frac=COLD_START_FRACTION, random_state=SEED)["entity_id"])
    remaining = entities_df[~entities_df["entity_id"].isin(cold_start_ids)]
    drift_ids = set(remaining.sample(frac=DRIFT_FRACTION, random_state=SEED)["entity_id"])
    entities_df["is_cold_start"] = entities_df["entity_id"].isin(cold_start_ids)
    entities_df["has_benign_drift"] = entities_df["entity_id"].isin(drift_ids)
    return entities_df

# ============================================================================
# STEP 2 — Normal, time-ordered event generation (+ benign noise/drift/cold-start)
# ============================================================================
def sample_hour(center, spread):
    return np.random.normal(center, spread) % 24

def generate_normal_events(entities_df, sim_days=SIM_DAYS):
    all_events = []
    for _, ent in entities_df.iterrows():
        eid, etype = ent["entity_id"], ent["entity_type"]
        habitual = list(ent["habitual_resources"])
        region_name, lat, lon = ent["home_region"]

        if ent["is_cold_start"]:
            n_events = random.randint(0, 2)
            days_active = [sim_days - random.randint(0, 2) for _ in range(n_events)]
        else:
            days_active = list(range(sim_days))

        for day in days_active:
            n_sessions_today = 1 if ent["is_cold_start"] else np.random.poisson(ent["sessions_per_day"])

            drifted = ent["has_benign_drift"] and day > sim_days * 0.6
            resource_choices = habitual.copy()
            hour_center = ent["active_hour_center"]
            if drifted:
                extra = random.sample(
                    [r for r in RESOURCE_POOL if r not in habitual],
                    k=min(3, len(RESOURCE_POOL) - len(habitual))
                )
                resource_choices = habitual + extra
                hour_center = (hour_center + random.uniform(-2, 2)) % 24

            for _ in range(n_sessions_today):
                hour = sample_hour(hour_center, ent["active_hour_spread"])
                ts = SIM_START + timedelta(days=day, hours=hour)

                # benign geo jitter (e.g. VPN) — only plausible for human users;
                # service accounts / edge devices are physically stationary
                use_home_geo = (etype != "user") or (random.random() > 0.03)
                if use_home_geo:
                    geo_lat, geo_lon, geo_name = lat, lon, region_name
                    src_ip = ent["home_ip"]
                else:
                    geo_name, geo_lat, geo_lon = random.choice(GEO_REGIONS)
                    src_ip = fake.ipv4_public()  # VPN jitter also changes apparent source IP

                resource = random.choice(resource_choices)
                duration = max(0.1, np.random.normal(ent["session_duration_mean"],
                                                       ent["session_duration_mean"] * 0.35))
                auth_success = random.random() > 0.02  # occasional benign typo failure

                all_events.append({
                    "entity_id": eid, "entity_type": etype, "timestamp": ts,
                    "geo_region": geo_name, "geo_lat": geo_lat, "geo_lon": geo_lon,
                    "source_ip": src_ip, "resource_accessed": resource,
                    "auth_method": ent["auth_pref"], "auth_success": auth_success,
                    "session_duration": round(duration, 2),
                    "command_sequence": f"seq_{random.randint(1,9999)}",
                    "device_fingerprint": ent["device_fp"], "label": "normal",
                })

    return pd.DataFrame(all_events).sort_values("timestamp").reset_index(drop=True)

# ============================================================================
# STEP 3 — Attack pattern injectors (7 categories, multi-field correlated)
# ============================================================================
def inject_brute_force(entities_df, n_incidents):
    rows = []
    for _, ent in entities_df.sample(n=n_incidents, random_state=SEED, replace=True).iterrows():
        day = random.randint(1, SIM_DAYS - 1)
        start_ts = SIM_START + timedelta(days=day, hours=random.uniform(0, 24))
        geo_name, geo_lat, geo_lon = random.choice(GEO_REGIONS)
        attacker_ip = fake.ipv4_public()  # one fixed source for the whole burst
        for i in range(random.randint(15, 60)):
            ts = start_ts + timedelta(seconds=i * random.uniform(1, 4))
            rows.append(_new_row(ent["entity_id"], ent["entity_type"], ts, geo_name, geo_lat, geo_lon,
                                  random.choice(RESOURCE_POOL), "password", False, 0.05,
                                  f"seq_bf_{i}", "unknown|unknown", "brute_force", source_ip=attacker_ip))
    return rows

def inject_impossible_travel(entities_df, n_incidents):
    rows = []
    pool = entities_df[entities_df["entity_type"] == "user"]
    for _, ent in pool.sample(n=min(n_incidents, len(pool)), random_state=SEED).iterrows():
        day = random.randint(1, SIM_DAYS - 1)
        home_region, home_lat, home_lon = ent["home_region"]
        far_region, far_lat, far_lon = random.choice([g for g in GEO_REGIONS if g[0] != home_region])
        base_ts = SIM_START + timedelta(days=day, hours=random.uniform(6, 20))
        rows.append(_new_row(ent["entity_id"], "user", base_ts, home_region, home_lat, home_lon,
                              random.choice(ent["habitual_resources"]), ent["auth_pref"], True,
                              5.0, "seq_it_1", ent["device_fp"], "normal", source_ip=ent["home_ip"]))
        gap_minutes = random.uniform(5, 25)
        rows.append(_new_row(ent["entity_id"], "user", base_ts + timedelta(minutes=gap_minutes),
                              far_region, far_lat, far_lon, random.choice(RESOURCE_POOL),
                              ent["auth_pref"], True, 3.0, "seq_it_2", ent["device_fp"], "impossible_travel"))
    return rows

def inject_credential_stuffing(entities_df, n_incidents):
    rows = []
    for _ in range(n_incidents):
        day = random.randint(1, SIM_DAYS - 1)
        start_ts = SIM_START + timedelta(days=day, hours=random.uniform(0, 24))
        geo_name, geo_lat, geo_lon = random.choice(GEO_REGIONS)
        attacker_ip = fake.ipv4_public()  # one fixed source hitting many distinct entities
        n_targets = random.randint(20, 80)
        targets = entities_df.sample(n=min(n_targets, len(entities_df)))
        for i, (_, ent) in enumerate(targets.iterrows()):
            ts = start_ts + timedelta(seconds=i * random.uniform(0.5, 2))
            success = random.random() < 0.03
            rows.append(_new_row(ent["entity_id"], ent["entity_type"], ts, geo_name, geo_lat, geo_lon,
                                  random.choice(RESOURCE_POOL), "password", success, 0.1,
                                  f"seq_cs_{i}", "unknown|unknown", "credential_stuffing", source_ip=attacker_ip))
    return rows

def inject_lateral_movement(entities_df, n_incidents):
    rows = []
    for _, ent in entities_df.sample(n=n_incidents, random_state=SEED, replace=True).iterrows():
        day = random.randint(1, SIM_DAYS - 1)
        start_ts = SIM_START + timedelta(days=day, hours=random.uniform(0, 24))
        never_touched = [r for r in RESOURCE_POOL if r not in ent["habitual_resources"]]
        chosen = random.sample(never_touched, min(random.randint(6, 15), len(never_touched)))
        home_region, home_lat, home_lon = ent["home_region"]
        for i, res in enumerate(chosen):
            ts = start_ts + timedelta(minutes=i * random.uniform(1, 6))
            rows.append(_new_row(ent["entity_id"], ent["entity_type"], ts, home_region, home_lat, home_lon,
                                  res, ent["auth_pref"], True, 2.0, f"seq_lm_{i}",
                                  ent["device_fp"], "lateral_movement", source_ip=ent["home_ip"]))
    return rows

def inject_device_spoofing(entities_df, n_incidents):
    rows = []
    pool = entities_df[entities_df["entity_type"].isin(["edge_device", "user"])]
    for _, ent in pool.sample(n=n_incidents, random_state=SEED, replace=True).iterrows():
        day = random.randint(1, SIM_DAYS - 1)
        ts = SIM_START + timedelta(days=day, hours=random.uniform(0, 24))
        spoofed_fp = f"{fake.mac_address()}|{random.choice(['legacy-fw-1.0','unknown-os'])}"
        home_region, home_lat, home_lon = ent["home_region"]
        rows.append(_new_row(ent["entity_id"], ent["entity_type"], ts, home_region, home_lat, home_lon,
                              random.choice(ent["habitual_resources"]), ent["auth_pref"], True,
                              2.0, "seq_ds_1", spoofed_fp, "device_spoofing"))
    return rows

def inject_low_and_slow_exfil(entities_df, n_incidents):
    rows = []
    for _, ent in entities_df.sample(n=n_incidents, random_state=SEED, replace=True).iterrows():
        start_day = random.randint(1, SIM_DAYS - 10)
        home_region, home_lat, home_lon = ent["home_region"]
        never_touched = [r for r in RESOURCE_POOL if r not in ent["habitual_resources"]]
        for d in range(random.randint(7, 10)):
            if random.random() < 0.7:
                ts = SIM_START + timedelta(days=start_day + d, hours=random.uniform(1, 5))
                res = random.choice(never_touched) if never_touched else random.choice(RESOURCE_POOL)
                rows.append(_new_row(ent["entity_id"], ent["entity_type"], ts, home_region, home_lat, home_lon,
                                      res, ent["auth_pref"], True, random.uniform(1, 4),
                                      f"seq_lse_{d}", ent["device_fp"], "low_and_slow_exfil", source_ip=ent["home_ip"]))
    return rows

def inject_insider_drift(entities_df, n_incidents):
    """Edge case: ambiguous, legitimate-looking scope expansion — used for FP tuning."""
    rows = []
    pool = entities_df[entities_df["entity_type"] == "user"]
    for _, ent in pool.sample(n=min(n_incidents, len(pool)), random_state=SEED).iterrows():
        start_day = random.randint(5, SIM_DAYS - 10)
        home_region, home_lat, home_lon = ent["home_region"]
        adjacent = [r for r in RESOURCE_POOL if r not in ent["habitual_resources"]]
        for d in range(random.randint(5, 9)):
            ts = SIM_START + timedelta(days=start_day + d, hours=ent["active_hour_center"])
            res = random.choice(adjacent) if adjacent else random.choice(RESOURCE_POOL)
            rows.append(_new_row(ent["entity_id"], "user", ts, home_region, home_lat, home_lon,
                                  res, ent["auth_pref"], True, ent["session_duration_mean"],
                                  f"seq_id_{d}", ent["device_fp"], "insider_drift", source_ip=ent["home_ip"]))
    return rows

def inject_all_attacks(entities_df, n_normal_events, target_rate=ATTACK_INJECTION_RATE):
    target_total_attack_rows = int(n_normal_events * target_rate / (1 - target_rate))
    weights = {
        "brute_force": 0.20, "impossible_travel": 0.12, "credential_stuffing": 0.18,
        "lateral_movement": 0.18, "device_spoofing": 0.08,
        "low_and_slow_exfil": 0.14, "insider_drift": 0.10,
    }
    avg_rows_per_incident = {
        "brute_force": 35, "impossible_travel": 2, "credential_stuffing": 45,
        "lateral_movement": 10, "device_spoofing": 1,
        "low_and_slow_exfil": 6, "insider_drift": 6,
    }
    counts = {k: max(1, int(target_total_attack_rows * w / avg_rows_per_incident[k]))
              for k, w in weights.items()}

    rows = []
    rows += inject_brute_force(entities_df, counts["brute_force"])
    rows += inject_impossible_travel(entities_df, counts["impossible_travel"])
    rows += inject_credential_stuffing(entities_df, counts["credential_stuffing"])
    rows += inject_lateral_movement(entities_df, counts["lateral_movement"])
    rows += inject_device_spoofing(entities_df, counts["device_spoofing"])
    rows += inject_low_and_slow_exfil(entities_df, counts["low_and_slow_exfil"])
    rows += inject_insider_drift(entities_df, counts["insider_drift"])
    print("Incident counts per attack type:", counts)
    return pd.DataFrame(rows)


def merge_events(normal_df, attack_df):
    full = pd.concat([normal_df, attack_df], ignore_index=True)
    full = full.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    full["is_anomaly"] = (full["label"] != "normal").astype(int)
    full["_row_id"] = range(len(full))
    return full

def add_source_ip_features(df):

    df = df.sort_values(["source_ip", "timestamp"]).copy()

    def rolling_distinct_entities(grp):
        grp = grp.set_index("timestamp")
        # rolling 5-min window of distinct entity_ids seen up to (and including) each row
        counts = []
        entity_series = grp["entity_id"]
        window = "300s"
        for ts in grp.index:
            recent = entity_series.loc[ts - pd.Timedelta(window):ts]
            counts.append(recent.nunique())
        grp["distinct_entities_per_source_5min"] = counts
        return grp.reset_index()

    result_rows = []
    for ip, grp in df.groupby("source_ip", sort=False):
        if grp["entity_id"].nunique() == 1:
            # single-entity IP (the common case for stable home IPs) — trivially always 1
            grp = grp.copy()
            grp["distinct_entities_per_source_5min"] = 1
            result_rows.append(grp)
        elif len(grp) == 1:
            grp = grp.copy()
            grp["distinct_entities_per_source_5min"] = 1
            result_rows.append(grp)
        else:
            result_rows.append(rolling_distinct_entities(grp.sort_values("timestamp")))

    return pd.concat(result_rows, ignore_index=True)

def engineer_features(df):
    """All features computed causally (only prior history at each row) — no leakage."""
    df = df.sort_values(["entity_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    feature_rows = []
    for eid, grp in df.groupby("entity_id", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        resource_history, seen_devices, durations, recent_fail_times = {}, set(), [], []
        last_ts, last_geo, recent_resources_window = None, None, []

        feats = {k: [] for k in [
            "prior_event_count", "time_since_last_access_min", "resource_seen_before",
            "resource_access_rank", "geo_distance_from_prev_km", "implied_travel_speed_kmh",
            "new_device_flag", "session_duration_zscore", "failed_auth_count_5min",
            "resource_diversity_recent10"]}

        for i, row in grp.iterrows():
            ts = row["timestamp"]
            feats["prior_event_count"].append(i)

            gap_min = (ts - last_ts).total_seconds() / 60 if last_ts is not None else np.nan
            feats["time_since_last_access_min"].append(gap_min)

            res = row["resource_accessed"]
            seen_before = res in resource_history
            feats["resource_seen_before"].append(int(seen_before))
            freq = resource_history.get(res, 0)
            total_seen = sum(resource_history.values()) + 1e-9
            feats["resource_access_rank"].append(freq / total_seen)
            resource_history[res] = resource_history.get(res, 0) + 1

            if last_geo is not None and gap_min and gap_min > 0.5:
                dist = haversine_km(last_geo[0], last_geo[1], row["geo_lat"], row["geo_lon"])
                speed = dist / (gap_min / 60)
            else:
                dist, speed = 0.0, 0.0
            feats["geo_distance_from_prev_km"].append(dist)
            feats["implied_travel_speed_kmh"].append(speed)
            last_geo = (row["geo_lat"], row["geo_lon"])

            fp = row["device_fingerprint"]
            feats["new_device_flag"].append(int(fp not in seen_devices and len(seen_devices) > 0))
            seen_devices.add(fp)

            if len(durations) >= 3:
                mu, sigma = np.mean(durations), np.std(durations) + 1e-6
                z = (row["session_duration"] - mu) / sigma
            else:
                z = 0.0
            feats["session_duration_zscore"].append(z)
            durations.append(row["session_duration"])

            if not row["auth_success"]:
                recent_fail_times.append(ts)
            recent_fail_times = [t for t in recent_fail_times if (ts - t).total_seconds() <= 300]
            feats["failed_auth_count_5min"].append(len(recent_fail_times))

            recent_resources_window.append(res)
            recent_resources_window = recent_resources_window[-10:]
            feats["resource_diversity_recent10"].append(len(set(recent_resources_window)))

            last_ts = ts

        for k, v in feats.items():
            grp[k] = v
        feature_rows.append(grp)

    result = pd.concat(feature_rows, ignore_index=True)
    result["time_since_last_access_min"] = result["time_since_last_access_min"].fillna(-1)

    # add source-IP-level aggregate (catches credential_stuffing's cross-entity signature)
    # merged on the unique _row_id assigned in merge_events() to avoid any risk of
    # duplicate-key collisions from rapid-fire attack rows sharing entity/timestamp/ip
    ip_feat = add_source_ip_features(df)[["_row_id", "distinct_entities_per_source_5min"]]
    result = result.merge(ip_feat, on="_row_id", how="left")
    result["distinct_entities_per_source_5min"] = result["distinct_entities_per_source_5min"].fillna(1)
    result = result.drop(columns=["_row_id"])
    return result

def time_based_split(df, train_frac=0.7, val_frac=0.15):
    df = df.sort_values("timestamp")
    min_day, max_day = df["timestamp"].min(), df["timestamp"].max()
    total_span = (max_day - min_day).total_seconds()
    train_cutoff = min_day + pd.Timedelta(seconds=total_span * train_frac)
    val_cutoff = min_day + pd.Timedelta(seconds=total_span * (train_frac + val_frac))
    train = df[df["timestamp"] <= train_cutoff]
    val = df[(df["timestamp"] > train_cutoff) & (df["timestamp"] <= val_cutoff)]
    test = df[df["timestamp"] > val_cutoff]
    return train, val, test


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Step 1/4: building entity population...")
    entities_df = build_entities()
    print(f"  {len(entities_df)} entities "
          f"({(entities_df.entity_type=='user').sum()} users, "
          f"{(entities_df.entity_type=='service_account').sum()} service accounts, "
          f"{(entities_df.entity_type=='edge_device').sum()} edge devices)")

    print("Step 2/4: generating normal behavior events...")
    normal_df = generate_normal_events(entities_df)
    print(f"  {len(normal_df)} normal events generated")

    print("Step 3/4: injecting attack patterns...")
    attack_df = inject_all_attacks(entities_df, len(normal_df))
    print(f"  {len(attack_df)} attack events generated")

    full = merge_events(normal_df, attack_df)
    print(f"Merged dataset: {len(full)} rows, anomaly rate = {full['is_anomaly'].mean():.3%}")

    print("Step 4/4: engineering features + time-based split...")
    featured = engineer_features(full)
    train, val, test = time_based_split(featured)
    print(f"  Train: {len(train)} (anomaly rate {train['is_anomaly'].mean():.3%})")
    print(f"  Val:   {len(val)} (anomaly rate {val['is_anomaly'].mean():.3%})")
    print(f"  Test:  {len(test)} (anomaly rate {test['is_anomaly'].mean():.3%})")

    featured.to_csv(f"{OUT_DIR}/full_dataset_with_features.csv", index=False)
    train.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test.csv", index=False)
    print(f"\nSaved outputs to {OUT_DIR}/")

if __name__ == "__main__":
    main()
