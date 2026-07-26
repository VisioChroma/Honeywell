# AI-Powered Behavioural Anomaly Detection

Sequence-aware detection of compromised-credential activity, insider drift, and device/network-level attacks across users, service accounts, and edge devices — built for Honeywell Hackathon (Problem Statement 4, Industrial Cybersecurity).

## Overview

Traditional signature-based security fails against novel or slow, low-and-slow intrusions. This project learns what "normal" behaviour looks like per entity (user / service account / edge device) and flags deviations, using a synthetic dataset (real intrusion logs are scarce/privacy-restricted), a dual-GRU ensemble detector, an attack-type classifier, an explainability layer, and an analyst-facing dashboard.

**Result:** PR-AUC 0.9508, 98.5% precision at a top-1% alert budget, ~2ms detection latency (CPU-only), streaming-validated against batch scores (max diff 7.15e-7).

## Repository structure

```
outputs/
├── ASSUMPTIONS.md              # Behavioural assumptions + attack taxonomy
├── synthetic_data_generator.py # Synthetic access-log generator
├── longterm_features.py        # 7d/30d rolling feature engineering
├── train.csv / val.csv / test.csv          # Time-based split (70/15/15)
├── baseline/
│   ├── baseline_profiling.py           # Statistical profile + autoencoder + OCSVM
│   └── baseline_validation_report.md   # Baseline separation results
├── detection/
│   ├── gru_model.py             # Version A — hand-implemented GRU (NumPy)
│   ├── gru_pytorch_v2.py        # Version B — stateful GRU (PyTorch, long-memory)
│   ├── seq_prep.py / seq_prep_v2.py   # Causal windowing
│   ├── optimize_ensemble_alpha.py     # Alpha grid search
│   ├── classify_attack_type.py        # RandomForest attack classifier
│   ├── streaming_replay.py            # Real-time feasibility validation
│   ├── rank_diagnostic.py             # Signal-strength diagnostic per class
│   └── gru_baseline_evaluation.md     # Baseline GRU results
├── explainability/
│   ├── explain_alert.py         # Per-feature, per-timestep ablation
│   ├── export_dashboard_data.py # Batch alert export → dashboard_data.json
│   ├── group_incidents.py       # Alert-burst → incident grouping
│   └── dashboard.html           # Analyst-facing dashboard (embedded data)
└── deliverable3_final_report.md # Full detection model report
```

## How to run

```bash
pip install -r requirements.txt

# 1. Generate synthetic data
python outputs/synthetic_data_generator.py

# 2. Baseline profiling
python outputs/baseline/baseline_profiling.py

# 3. Train detection ensemble
python outputs/detection/gru_model.py
python outputs/detection/gru_pytorch_v2.py
python outputs/detection/optimize_ensemble_alpha.py

# 4. Classify flagged alerts
python outputs/detection/classify_attack_type.py

# 5. Generate explanations + dashboard data
python outputs/explainability/export_dashboard_data.py
python outputs/explainability/group_incidents.py dashboard_data.json dashboard_data_grouped.json

# 6. Open the dashboard
open outputs/explainability/dashboard.html
```

## Key design decisions

- **Ensemble over single model:** fixed-window GRU (fast, short-term) + stateful GRU (long-memory, 7d/30d features), combined via `alpha=0.10`, only after both were independently tested.
- **Time-based split**, never random row split, to prevent leakage on sequential data.
- **Explainability via per-timestep ablation**, not whole-window ablation — catches signals concentrated at a single event in the window.
- **Alert threshold in probability space**, not logit averaging, since sigmoid is nonlinear.
- **Incident grouping** collapses alert bursts (e.g. 60 brute-force events in minutes) into one queue entry using a chained time window.

## Known limitations

- Rare attack types (`low_and_slow_exfil`, `insider_drift`) remain hard to catch within a tight alert budget — `insider_drift` is an intentionally ambiguous edge case by design.
- Long-term rolling features are currently computed in batch; a production system needs an incremental online feature engine.
- Concept-drift adaptation (EWMA) is built and tested but not validated against real benign drift, since the synthetic data has zero such events.

Full details: see `ASSUMPTIONS.md` and `deliverable3_final_report.md`.

## Author

Addanki Sai Krishna Siddu — Student ID 22011P0523
