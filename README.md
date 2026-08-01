# ATP Tennis Match Predictor

A machine-learning model that predicts the outcome of professional men's
singles (ATP) tennis matches, built from scratch in Python.

**Live app:** [your-streamlit-url-here]

## Results

Evaluated on 9,398 held-out matches (2023–2026), trained on 2005–2022:

| Predictor          | Accuracy | Notes                            |
|--------------------|----------|----------------------------------|
| Rank baseline      | 64.3%    | higher-ranked player wins        |
| Elo baseline       | 64.5%    | higher blended-Elo player wins   |
| **Logistic model** | **65.3%**| log-loss 0.616, well calibrated  |
| Betting market     | 68.5%    | bookmaker favourite (ceiling)    |

The model beats both the ranking and Elo baselines, and closes roughly a
third of the gap to the betting market — which has access to information
(injuries, conditions, late news) this model does not.

## The problem

[Your words: 2-3 sentences. Why tennis prediction is interesting, and why
~65% is a strong result rather than a weak one. You understand this well —
the market caps around 69%, 80% would signal leakage. Say it in your voice.]

## Data

- Source: [ATP Tennis 2000–2026 Daily Update](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull), Kaggle (CC0 / public domain)
- ~68,000 ATP tour-level singles matches, 2000–July 2026, with bookmaker odds
- Refreshable via the Kaggle API (see "Updating" below)

## Cleaning decisions

[Your words for each — you made these calls and know why:]
- Placeholder ranks (−1) dropped
- Impossible odds (≤1.0) marked missing rather than dropped — [why: they're
  still valid matches for training, just not for market comparison]
- Player-name variants merged (e.g. three spellings of Del Potro), but
  genuinely different players with similar names kept separate (e.g. the two
  Kuznetsovs) — [a sentence on this judgment call]
- Scope: tour-level singles only; team events (Davis Cup etc.) excluded upstream

## Method

**Elo ratings, built from scratch:**
- General Elo (K=32, base 1500), updated chronologically match by match
- Surface-specific Elo (Hard/Clay/Grass), blended 40% surface / 60% general
- [One line on the no-leakage discipline: features recorded *before* each
  rating update — why this matters]

**Features** (all as differences between the two players):
`elo_diff`, `rank_diff` (log-transformed), `gap_diff` (days since last match),
`form_diff` (win rate over last 10)

**Model:** logistic regression, time-based split (train 2005–2022, test
2023–2026 — never random, to prevent temporal leakage).

## What I found

[This section is the most valuable — your genuine findings, in your words:]
- Elo beats raw ranking; blended surface Elo beats general Elo
- Recent form added almost nothing once Elo was included — [why that makes
  sense: Elo already absorbs winning streaks]
- Layoff matters: accuracy drops from ~68% (match-fresh) to ~65% after a
  two-week gap
- The model is well calibrated — predicted probabilities match actual
  outcomes within ~2 points across all buckets [reference the calibration table]

## Limitations

- No injury, withdrawal, or conditions data — the market's edge
- Predictions are a snapshot of ratings at the last data refresh
- Retirements/walkovers excluded upstream by the data provider

## Running it

    pip install -r requirements.txt
    python clean_kaggle.py   # raw -> clean_tennis.csv
    python elo.py            # Elo features -> tennis_elo.csv, elo_state.json
    python model.py          # train, evaluate -> model.pkl
    streamlit run app.py     # launch the web app

## Updating

Data refreshes via the Kaggle API:

    kaggle datasets download -d dissfya/atp-tennis-2000-2023daily-pull -p data/ --unzip
    python clean_kaggle.py && python elo.py && python model.py

Re-running the pipeline regenerates all features and the model with current data.

## Future work

[Your words: gradient boosting (interactions), tournament-context features,
WTA data, automated daily retraining.]
