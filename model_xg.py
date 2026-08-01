import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv('data/tennis_elo.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'].dt.year >= 2005].copy()

df['y'] = (df['Winner'] == df['Player_1']).astype(int)

df['elo_diff']  = df['Blend_1'] - df['Blend_2']
df['rank_diff'] = np.log(df['Rank_2']) - np.log(df['Rank_1'])
df['gap_diff']  = df['Gap_1'] - df['Gap_2']
df['form_diff'] = df['Form_1'] - df['Form_2']

FEATURES = ['elo_diff', 'rank_diff', 'gap_diff', 'form_diff']

train = df[df['Date'].dt.year <= 2022]
test  = df[df['Date'].dt.year >= 2023].copy()

# No scaling needed - trees split on thresholds, so feature scale is irrelevant
model = LGBMClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    verbose=-1,
)
model.fit(train[FEATURES], train['y'])

pred  = model.predict(test[FEATURES])
proba = model.predict_proba(test[FEATURES])[:, 1]

print(f"XGBoost accuracy: {accuracy_score(test['y'], pred):.4f}")
print(f"XGBoost log-loss: {log_loss(test['y'], proba):.4f}")
print(f"\n(Logistic was: 0.6543 acc, 0.6159 log-loss)")

print("\nFeature importance:")
for f, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f:12s} {imp:.3f}")

test['proba'] = proba
bins = pd.cut(test['proba'], [0, .3, .4, .5, .6, .7, 1.0])
print("\nCalibration:")
print(test.groupby(bins, observed=True).agg(
    predicted=('proba', 'mean'), actual=('y', 'mean'), n=('y', 'size')).round(3))