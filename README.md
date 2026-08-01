# ATP Tennis Match Predictor

A machine-learning model that predicts the outcome of professional men's
singles (ATP) tennis matches, built from scratch in Python.

**Live app:** https://atp-match-predictor-jatfpbhkrc5fmheyycwyqk.streamlit.app/

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

Tennis is genuinely unpredictable — upsets are common, and the skill gap
between tour-level players is often small, so the better player loses
frequently. Betting markets, which price in far more information than this
model has access to, top out around 68–69% accuracy. Reaching 65% puts the
model close to that practical ceiling, which is why it's a strong result
rather than a weak one. A model claiming 80%+ would almost certainly be
leaking future information rather than genuinely predicting.

## Data

- Source: [ATP Tennis 2000–2026 Daily Update](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull), Kaggle (CC0 / public domain)
- ~68,000 ATP tour-level singles matches, 2000–July 2026, with bookmaker odds
- Refreshable via the Kaggle API (see "Updating" below)

## Cleaning decisions

- Placeholder ranks (−1) dropped
- Impossible odds (≤1.0) marked missing rather than dropped — a match with
  missing odds still has a valid winner, rank, and surface, so it's useful for
  training; it just can't be used when comparing against the betting market.
- Player-name variants merged (e.g. three spellings of Del Potro), but
  genuinely different players with similar names kept separate. Merging
  required judgment: "Del Potro J.M." and "Del Potro J. M." are clearly the
  same player, but "Kuznetsov Al." and "Kuznetsov An." are two different people
  (Alex and Andrey). I used a hand-checked dictionary rather than an automatic
  rule, since automation would have wrongly merged distinct players.
- Scope: tour-level singles only; team events (Davis Cup etc.) excluded upstream.

## Method

**Elo ratings, built from scratch:**
- General Elo (K=32, base 1500), updated chronologically match by match
- Surface-specific Elo (Hard/Clay/Grass), blended 40% surface / 60% general
- Each match's features are recorded *before* the ratings are updated with that
  match's result — otherwise the model would be told the outcome it's supposed
  to predict.

**Features** (all as differences between the two players):
`elo_diff`, `rank_diff` (log-transformed), `gap_diff` (days since last match),
`form_diff` (win rate over last 10).

**Model:** logistic regression, time-based split (train 2005–2022, test
2023–2026 — never random, to prevent temporal leakage).

## What I found

- Elo beats raw ranking; blended surface Elo beats general Elo.
- Recent form added almost nothing once Elo was included. This makes sense:
  Elo already rises and falls based on match results, so a player's recent wins
  and losses are baked into their rating. A separate form feature is largely
  redundant with what Elo already captures.
- Layoff matters: accuracy drops from ~68% (match-fresh) to ~65% after a
  two-week gap.
- The model is well calibrated — predicted probabilities match actual outcomes
  within ~2 points across all buckets.

## Limitations

- No injury, withdrawal, or conditions data — the market's edge.
- Predictions are a snapshot of ratings at the last data refresh.
- Retirements and walkovers are excluded upstream by the data provider.

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

- Gradient boosting (XGBoost/LightGBM) to capture feature interactions the
  linear model can't
- Tournament-context features (round, best-of-5, indoor/outdoor)
- Extending to WTA (women's) matches
- Automated daily retraining via a scheduled pipeline