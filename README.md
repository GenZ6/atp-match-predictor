# ATP Tennis Match Predictor

Predicting the outcome of professional (ATP) men's singles matches from
pre-match information, built from scratch in Python.

## Result

| Predictor            | Accuracy | Notes                          |
|----------------------|----------|--------------------------------|
| Rank baseline        | 64.4%    | higher-ranked player wins      |
| Elo baseline         | 64.6%    | higher blended-Elo player wins |
| **Logistic model**   | **65.4%**| log-loss 0.616, well calibrated|
| Betting market       | 68.5%    | bookmaker favourite (ceiling)  |

Evaluated on 9,345 held-out matches (2023–2026), trained on 2005–2022.

## Data

- Source: [describe the Kaggle dataset], ATP tour-level singles, 2000–July 2026, with bookmaker odds
- ~68,000 matches after cleaning
- [one line on scope: singles only, team events like Davis Cup excluded]

## Cleaning decisions
- [placeholder ranks of -1 dropped]
- [impossible odds (≤1.0) marked missing, not dropped — why]
- [player-name variants merged; note the ones deliberately kept separate]

## Method
- [Elo from scratch: general, surface-specific, blended (40% surface)]
- [features: elo_diff, rank_diff (log), gap_diff, form_diff]
- [no-leakage discipline: features recorded before ratings update]
- [time-based split, never random — why]

## What I found
- [Elo beats rank; blended beats general]
- [recent form added ~nothing once Elo included — and why that makes sense]
- [layoff signal is real: accuracy drops from 68% to ~65% after a 2-week gap]
- [model is well calibrated — reference the calibration table]

## Limitations
- [no injury/withdrawal information — the market's edge]
- [dataset excludes incomplete matches upstream]
- [gradient boosting deferred — environment constraint]

## Running it
    python clean_kaggle.py   # raw -> clean_tennis.csv
    python elo.py            # adds Elo features -> tennis_elo.csv
    python model.py          # trains and evaluates
