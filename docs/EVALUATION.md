# Evaluation

These figures are the saved metrics in `ml/models/`. They describe the holdout used at training time. The app rounds money to the nearest €10,000 on screen. Provenance for the tables is in [DATA.md](../DATA.md).

## Transfer value

Each finished season is labelled with the Transfermarkt quote nearest 30 June of that season’s end year, looking from 400 days before that date to 45 days after it. Seasons with no quote in that window are left unlisted and are not trained on. The in-progress season is not trained on.

The comparison uses a later-season holdout. Models fit on 2021/22–2024/25 (1,823 rows) and are scored on 2025/26 (468 rows). The winner is refit on all 2,291 finished rows and saved. The numbers below are the holdout, not the error of that refit on its own training rows.

Choosing the model family on this same holdout makes the winning score a little optimistic. The note stored with the metrics says that.

| Model | Holdout MAE | Holdout RMSE | Holdout R² |
| --- | ---: | ---: | ---: |
| HistGradientBoosting | €10.83M | €16.56M | 0.483 |
| RandomForest | €11.45M | €17.24M | 0.439 |
| LinearRegression | €11.49M | €17.36M | 0.431 |
| Ridge | €11.49M | €17.38M | 0.430 |
| Ridge on a log target | €12.52M | €19.19M | 0.305 |

HistGradientBoosting is the saved model. Error grows with the quote:

| Quote band | Rows | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Under €10M | 102 | €5.52M | €7.48M |
| €10–30M | 216 | €7.10M | €9.49M |
| €30–60M | 105 | €12.22M | €14.84M |
| Over €60M | 45 | €37.54M | €42.16M |

Features are age, appearances, goals, assists, minutes, goals per 90, assists per 90, and position. The model does not see the quote, the club, or the player’s name. Estimates are clipped at zero.

**Limits.** A €10.8M typical error is wide for a squad player and very wide for the players quoted over €60M. The quote is a public valuation, not a completed transfer fee. Name matching leaves some players unlisted. An FPL price is never used as a stand-in fee.

## Match odds

Features are sorted by date. The model trains on the earlier 3,344 rows and is scored on the later 836, from 6 April 2024 through 24 May 2026. The feature table starts on 8 August 2015. Form, goals, shots, and head-to-head rates use only matches before the fixture being scored.

| Metric | Holdout |
| --- | ---: |
| Accuracy | 46.2% |
| Log loss | 1.045 |
| Multiclass Brier | 0.629 |
| Rows | 3,344 train / 836 test |

On that holdout, 351 of 836 matches were home wins (42.0%). Predicting home every time would land there. The model’s 46.2% is a small step above that. Draws are the weak class: 45 of 211 draws were called correctly.

Confusion matrix, rows are the result and columns are the prediction, in order away / draw / home:

|  | Pred away | Pred draw | Pred home |
| --- | ---: | ---: | ---: |
| Away | 146 | 55 | 73 |
| Draw | 81 | 45 | 85 |
| Home | 92 | 64 | 195 |

**Limits.** Three-way football results are noisy, and this model only uses recent team rates. It is not a betting system. Team names have to match the names in the results file. The published scores are this saved holdout. Retrain after new results if you want the metrics to include them.

## Scouting

Neighbour search is not a classifier, so there is no accuracy score. The index is 273 current-squad players with at least 900 minutes in 2025/26, out of 667 players on the current squad list. Distance uses eight per-90 rates inside one position: goals, assists, expected goals, expected assists, goals conceded, saves, clean sheets, and defensive contribution. Age, appearances, and raw totals are not in that distance.

The map is k-means with 8 clusters on those rates plus position. Silhouette is 0.233. The first two principal components show 31.0% and 21.5% of the scaled variance. Cluster names are descriptive labels, and a group is only named for a position when that position is at least 60% of the group.

**Limits.** Similarity is “similar 2025/26 per-90 profile”, not “similar transfer value” or “similar role in every season”. Players under 900 minutes stay searchable and get no recommendation. New signings with no 2025/26 row keep their current partial stats on the card and are left out of the index. The low silhouette means the clusters overlap; use the neighbour list, not the cluster name, when comparing two players.
