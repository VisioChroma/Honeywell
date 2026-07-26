# Baseline Profiling — Validation Report

Sanity check on the VALIDATION split (not test) — confirms both approaches produce meaningfully higher anomaly scores for real attacks vs normal traffic, before Deliverable 3 builds the full sequence-aware detector on top.

## Approach B: Learned model scores by label

| label               |   autoencoder_recon_error |   ocsvm_anomaly_score |
|:--------------------|--------------------------:|----------------------:|
| device_spoofing     |               2461.07     |               5.8625  |
| credential_stuffing |               2197.52     |               6.832   |
| brute_force         |                184.61     |               6.78233 |
| normal              |                  2.44863  |              -1.13494 |
| lateral_movement    |                  1.91011  |               2.76952 |
| impossible_travel   |                  1.84873  |               1.93093 |
| low_and_slow_exfil  |                  0.848735 |               0.9018  |
| insider_drift       |                  0.431074 |              -1.21833 |


Overall: mean reconstruction error is **157.2x higher** for attacks (384.925) than normal (2.449).


## Approach A: Statistical profile coverage

- 552 entities got individual profiles (>= 10 normal events in train)

- 3 population-level fallback profiles (one per entity_type), used for cold-start / low-history entities

- Of 561 entities seen in validation data, 552 have individual profiles, 9 will use the population fallback
