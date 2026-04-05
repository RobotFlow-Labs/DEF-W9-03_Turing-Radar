# Turing-Radar — Asset Manifest

## Paper
- Title: The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving
- ArXiv: 2602.03856 (submitted January 23, 2026)
- DOI: https://doi.org/10.48550/arXiv.2602.03856
- Authors: Edward Gunn, Adam Hosford, Robert Jones, Leo Zeitler, Ian Groves, Victoria Nockles

## Status
- READY (dataset + challenge repository publicly available)

## Reference Repositories
| Repository | Purpose | URL | Local Path | Status |
|---|---|---|---|---|
| Turing Deinterleaving Challenge | Official dataset loader, baseline deinterleaver, evaluation utilities | https://github.com/alan-turing-institute/turing-deinterleaving-challenge | repositories/turing-deinterleaving-challenge | DONE |

## Datasets
| Dataset | Scope | Source | Path (CUDA server target) | Local Path | Status |
|---|---|---|---|---|---|
| TSRD (train/validation/test) | Interleaved PDW pulse trains in scan+stare modes | https://huggingface.co/datasets/alan-turing-institute/turing-deinterleaving-challenge | /mnt/forge-data/datasets/turing_deinterleaving_challenge | data/ (download on demand) | READY |

## Dataset Scale (paper v1)
| Receiver | Pulse trains | Total pulses | Max emitters |
|---|---:|---:|---:|
| Stare | 3000 | 2,876,184,895 | 77 |
| Scan | 3000 | 90,138,651 | 90 |

## Dataset Scale (challenge repo update, Feb 2026)
| Receiver | Pulse trains | Total pulses | Max emitters |
|---|---:|---:|---:|
| Stare | 3000 | 3.86B | 85 |
| Scan | 3000 | 282.8M | 90 |

## PDW Feature Contract
1. `time_of_arrival` (microseconds)
2. `centre_frequency` (MHz)
3. `pulse_width` (microseconds)
4. `angle_of_arrival` (degrees)
5. `amplitude` (dB)

## Baseline / Expected Metrics
| Model | Receiver | V-measure | ARI | AMI | Homogeneity | Completeness | MCC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HDBSCAN identity baseline | Stare | 0.538 | 0.270 | 0.496 | 0.638 | 0.504 | 0.057 | 0.010 |
| HDBSCAN identity baseline | Scan | 0.187 | 0.017 | 0.146 | 0.409 | 0.127 | 0.071 | 0.037 |

## Hyperparameters from Source Material
The dataset paper is model-agnostic and does not prescribe a canonical training recipe.
For reproducible baseline benchmarking in this module, we lock:
- Clustering: HDBSCAN
- `cluster_selection_epsilon`: 0.0 (identity baseline in challenge notebook)
- Evaluation metric of record: median V-measure

## Constraints / Notes
- Labels are pulse-train local only (label IDs are not globally consistent across files).
- Test windows have unknown emitter count; formulation is clustering, not closed-set classification.
- This module should remain challenge-compatible first; CUDA optimization is a later pass.
