import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv('data/tennis_elo.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'].dt.year >= 2005].copy()

# Target: did Player_1 win?
df['y'] = (df['Winner'] == df['Player_1']).astype(int)

# Features as differences - only the gap between the two players matters.
# Rank is log-transformed because rank spacing is uneven: #1 vs #5 is a far
# bigger real gap than #95 vs #99, even though both are "4 apart".
df['elo_diff']  = df['Blend_1'] - df['Blend_2']
df['rank_diff'] = np.log(df['Rank_2']) - np.log(df['Rank_1'])
df['gap_diff']  = df['Gap_1'] - df['Gap_2']
df['form_diff'] = df['Form_1'] - df['Form_2']

FEATURES = ['elo_diff', 'rank_diff', 'gap_diff', 'form_diff']

# Time-based split - never random, or the model sees the future
train = df[df['Date'].dt.year <= 2022]
test  = df[df['Date'].dt.year >= 2023].copy()

# Scale so coefficients are comparable. Fit on train only.
scaler = StandardScaler().fit(train[FEATURES])
Xtr = scaler.transform(train[FEATURES])
Xte = scaler.transform(test[FEATURES])

model = LogisticRegression(max_iter=1000).fit(Xtr, train['y'])
pred  = model.predict(Xte)
proba = model.predict_proba(Xte)[:, 1]

print(f"Train: {len(train)} matches ({train['Date'].dt.year.min()}-{train['Date'].dt.year.max()})")
print(f"Test:  {len(test)} matches ({test['Date'].dt.year.min()}-{test['Date'].dt.year.max()})")
print(f"\nModel accuracy: {accuracy_score(test['y'], pred):.4f}")
print(f"Model log-loss: {log_loss(test['y'], proba):.4f}")

# Baselines on the identical test set
print(f"\nRank baseline:  {((test['Rank_1'] < test['Rank_2']) == test['y']).mean():.4f}")
print(f"Elo baseline:   {((test['Blend_1'] > test['Blend_2']) == test['y']).mean():.4f}")

odds_ok = test['Odd_1'].notna() & test['Odd_2'].notna()
t = test[odds_ok]
print(f"Market:         {((t['Odd_1'] < t['Odd_2']) == t['y']).mean():.4f}  ({odds_ok.sum()} matches)")

print("\nCoefficients (scaled - directly comparable):")
for f, c in zip(FEATURES, model.coef_[0]):
    print(f"  {f:12s} {c:+.4f}")

# Calibration: do matches predicted at 70% actually resolve 70% of the time?
test['proba'] = proba
bins = pd.cut(test['proba'], [0, .3, .4, .5, .6, .7, 1.0])
print("\nCalibration:")
print(test.groupby(bins, observed=True).agg(
    predicted=('proba', 'mean'),
    actual=('y', 'mean'),
    n=('y', 'size')).round(3))
import joblib
joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES}, 'data/model.pkl')
print("Saved model")