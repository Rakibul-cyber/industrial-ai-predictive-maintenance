# Model comparison

| Model | Version | CV RMSE (mean ± std) | CV MAE (mean ± std) | Test RMSE | Test MAE | NASA score | Test RMSE (truth clipped) |
|---|---|---|---|---|---|---|---|
| baseline | `ridge-20261108T150138Z` | 19.152 ± 1.182 | 15.112 ± 1.156 | 19.144 | 15.347 | 590.6 | 17.845 |
| primary (selected) | `hgb-20261108T150441Z` | 16.559 ± 0.695 | 11.843 ± 0.676 | 16.610 | 12.567 | 441.5 | 15.333 |

CV: 5-fold GroupKFold by engine on the training set (capped labels). Test: official FD001 test set, last observed cycle of each of the 100 engines, unclipped truth unless stated. Predictions clamped to [0, RUL_CAP] (ADR-027).

Selected primary because its 5-fold GroupKFold CV RMSE (16.559) is lower than baseline's (19.152) (ADR-024). Official test metrics are reported but do not drive the choice. The official test ranking agrees with the CV ranking.
