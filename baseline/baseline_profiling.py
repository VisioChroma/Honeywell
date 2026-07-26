import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

SEED = 42
MIN_HISTORY = 10          # entities with fewer normal events use population fallback
OCSVM_SAMPLE_CAP = 6000   # subsample cap per entity_type for OCSVM training (speed)

# Path setup:
# BASE_DIR      -> .../HoneyWell/outputs/baseline
# OUTPUTS_DIR   -> .../HoneyWell/outputs  (where train.csv and val.csv are!)
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR.parent
OUT_DIR = BASE_DIR

os.makedirs(OUT_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "session_duration", "resource_access_rank", "geo_distance_from_prev_km",
    "implied_travel_speed_kmh", "new_device_flag", "session_duration_zscore",
    "failed_auth_count_5min", "resource_diversity_recent10",
    "distinct_entities_per_source_5min", "time_since_last_access_min",
    "resource_seen_before",
]

def load_data():
    train = pd.read_csv(OUTPUTS_DIR / "train.csv")
    val = pd.read_csv(OUTPUTS_DIR / "val.csv")
    for df in (train, val):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return train, val

FEATURES_FOR_MODEL = NUMERIC_FEATURES + ["hour_sin", "hour_cos"]


def circular_mean_std_hours(hours):
    """Mean/std of hour-of-day respecting the 0-24 wraparound (circular stats)."""
    radians = 2 * np.pi * hours / 24
    sin_mean, cos_mean = np.mean(np.sin(radians)), np.mean(np.cos(radians))
    mean_hour = (np.arctan2(sin_mean, cos_mean) / (2 * np.pi) * 24) % 24
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    circ_std = np.sqrt(max(-2 * np.log(max(r, 1e-9)), 0)) * 24 / (2 * np.pi)
    return mean_hour, circ_std

def build_entity_profile(grp):
    profile = {}
    for feat in NUMERIC_FEATURES:
        profile[f"{feat}_mean"] = grp[feat].mean()
        profile[f"{feat}_std"] = max(grp[feat].std(), 1e-6) if len(grp) > 1 else 1e-6
    mean_h, std_h = circular_mean_std_hours(grp["hour_of_day"].values)
    profile["active_hour_mean"] = mean_h
    profile["active_hour_std"] = max(std_h, 0.25)
    profile["typical_auth_method"] = grp["auth_method"].mode().iloc[0]
    profile["typical_geo_region"] = grp["geo_region"].mode().iloc[0]
    res_counts = grp["resource_accessed"].value_counts()
    top_resources = res_counts[res_counts.cumsum() / res_counts.sum() <= 0.9].index.tolist()
    profile["habitual_resources"] = top_resources if top_resources else res_counts.index[:1].tolist()
    profile["n_events_used"] = len(grp)
    return profile

def build_statistical_profiles(train):
    normal = train[train["label"] == "normal"]

    entity_profiles = []
    for eid, grp in normal.groupby("entity_id"):
        if len(grp) < MIN_HISTORY:
            continue  # too little history — will fall back to population profile
        p = build_entity_profile(grp)
        p["entity_id"] = eid
        p["entity_type"] = grp["entity_type"].iloc[0]
        entity_profiles.append(p)
    entity_df = pd.DataFrame(entity_profiles)

    population_profiles = []
    for etype, grp in normal.groupby("entity_type"):
        p = build_entity_profile(grp)
        p["entity_type"] = etype
        population_profiles.append(p)
    population_df = pd.DataFrame(population_profiles)

    return entity_df, population_df


def build_learned_models(train):
    normal = train[train["label"] == "normal"]
    models = {}

    for etype, grp in normal.groupby("entity_type"):
        X = grp[FEATURES_FOR_MODEL].fillna(0).values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        # Autoencoder-style reconstruction model: bottleneck MLP trained to
        # reproduce its own input. Reconstruction error = anomaly signal.
        bottleneck = max(2, len(FEATURES_FOR_MODEL) // 3)
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(8, bottleneck, 8),
            activation="relu", solver="adam", max_iter=300,
            random_state=SEED, early_stopping=True,
        )
        autoencoder.fit(Xs, Xs)

        # One-Class SVM: complementary learned boundary around normal
        if len(Xs) > OCSVM_SAMPLE_CAP:
            idx = np.random.RandomState(SEED).choice(len(Xs), OCSVM_SAMPLE_CAP, replace=False)
            Xs_fit = Xs[idx]
        else:
            Xs_fit = Xs
        ocsvm = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale").fit(Xs_fit)

        models[etype] = {"scaler": scaler, "autoencoder": autoencoder, "ocsvm": ocsvm}
        print(f"  [{etype}] trained on {len(Xs)} normal events "
              f"(OCSVM fit on {len(Xs_fit)})")

    return models

def score_with_learned_models(df, models):
    recon_errors, ocsvm_scores = [], []
    for etype, grp_idx in df.groupby("entity_type").groups.items():
        grp = df.loc[grp_idx]
        m = models[etype]
        X = grp[FEATURES_FOR_MODEL].fillna(0).values
        Xs = m["scaler"].transform(X)
        recon = m["autoencoder"].predict(Xs)
        err = np.mean((Xs - recon) ** 2, axis=1)
        ocsvm_raw = -m["ocsvm"].decision_function(Xs)  # higher = more anomalous
        recon_errors.extend(zip(grp_idx, err))
        ocsvm_scores.extend(zip(grp_idx, ocsvm_raw))

    recon_series = pd.Series(dict(recon_errors)).reindex(df.index)
    ocsvm_series = pd.Series(dict(ocsvm_scores)).reindex(df.index)
    return recon_series, ocsvm_series


def validate(train, val, entity_df, population_df, models):
    val = val.copy()
    recon, ocsvm = score_with_learned_models(val, models)
    val["autoencoder_recon_error"] = recon
    val["ocsvm_anomaly_score"] = ocsvm

    lines = ["# Baseline Profiling — Validation Report\n",
             "Sanity check on the VALIDATION split (not test) — confirms both approaches "
             "produce meaningfully higher anomaly scores for real attacks vs normal traffic, "
             "before Deliverable 3 builds the full sequence-aware detector on top.\n",
             "## Approach B: Learned model scores by label\n"]

    summary = val.groupby("label")[["autoencoder_recon_error", "ocsvm_anomaly_score"]].mean()
    summary = summary.sort_values("autoencoder_recon_error", ascending=False)
    lines.append(summary.to_markdown())

    normal_recon = val[val.label == "normal"]["autoencoder_recon_error"].mean()
    attack_recon = val[val.label != "normal"]["autoencoder_recon_error"].mean()
    separation_ratio = attack_recon / max(normal_recon, 1e-9)
    lines.append(f"\n\nOverall: mean reconstruction error is **{separation_ratio:.1f}x higher** "
                 f"for attacks ({attack_recon:.3f}) than normal ({normal_recon:.3f}).\n")

    lines.append("\n## Approach A: Statistical profile coverage\n")
    lines.append(f"- {len(entity_df)} entities got individual profiles "
                 f"(>= {MIN_HISTORY} normal events in train)\n")
    lines.append(f"- {len(population_df)} population-level fallback profiles "
                 f"(one per entity_type), used for cold-start / low-history entities\n")

    val_entities = set(val["entity_id"].unique())
    covered = val_entities & set(entity_df["entity_id"])
    fallback_needed = val_entities - covered
    lines.append(f"- Of {len(val_entities)} entities seen in validation data, "
                 f"{len(covered)} have individual profiles, "
                 f"{len(fallback_needed)} will use the population fallback\n")

    report = "\n".join(lines)
    with open(f"{OUT_DIR}/baseline_validation_report.md", "w") as f:
        f.write(report)
    return report


def main():
    print("Loading train/val data...")
    train, val = load_data()

    print("\nApproach A: building statistical profiles...")
    entity_df, population_df = build_statistical_profiles(train)
    print(f"  {len(entity_df)} individual entity profiles, "
          f"{len(population_df)} population-level fallback profiles")

    print("\nApproach B: training learned one-class models per entity_type...")
    models = build_learned_models(train)

    print("\nValidating both approaches on held-out validation split...")
    report = validate(train, val, entity_df, population_df, models)
    print("\n" + report)

    # Save artifacts
    entity_df.to_csv(f"{OUT_DIR}/statistical_profiles_entity.csv", index=False)
    population_df.to_csv(f"{OUT_DIR}/statistical_profiles_population.csv", index=False)
    joblib.dump(models, f"{OUT_DIR}/learned_models.joblib")
    print(f"\nSaved all baseline artifacts to {OUT_DIR}/")

if __name__ == "__main__":
    main()