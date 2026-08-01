import pandas as pd
import numpy as np

NAME_FIXES = {
    'Andersen J.': 'Andersen J.F.',
    'Aragone J.': 'Aragone J.C.',
    'Aragone JC': 'Aragone J.C.',
    'Bailly G.': 'Bailly G.A.',
    'Bautista R.': 'Bautista Agut R.',
    'Bogomolov Jr.A.': 'Bogomolov Jr. A.',
    'Chela J.': 'Chela J.I.',
    'Del Potro J.': 'Del Potro J.M.',
    'Del Potro J. M.': 'Del Potro J.M.',
    'Etcheverry T.': 'Etcheverry T. M.',
    'Ferrero J.': 'Ferrero J.C.',
    'Galan D.': 'Galan D.E.',
    'Gambill J. M.': 'Gambill J.M.',
    'Gomez F.': 'Gomez F.A.',
    'Guzman J.': 'Guzman J.P.',
    'Herbert P.': 'Herbert P.H.',
    'Herbert P.H': 'Herbert P.H.',
    'Hernandez-Fernandez J': 'Hernandez-Fernandez J.',
    'Huesler M.': 'Huesler M.A.',
    'Jones G.': 'Jones G.D.',
    'Jun W.': 'Jun W.S.',
    'Kim K': 'Kim K.',
    'Kohlschreiber P..': 'Kohlschreiber P.',
    'Kwon S.': 'Kwon S.W.',
    'Lisnard J.': 'Lisnard J.R.',
    'Lu Y.': 'Lu Y.H.',
    'Marin J.A': 'Marin J.A.',
    'Mathieu P.': 'Mathieu P.H.',
    'Moroni G.': 'Moroni G.M.',
    'Qureshi A.': 'Qureshi A.U.H.',
    'Scherrer J.': 'Scherrer J.C.',
    'Schwaerzler J.': 'Schwaerzler J.J.',
    'Silva F.': 'Silva F.F.',
    'Tirante T. A.': 'Tirante T.A.',
    'Tseng C. H.': 'Tseng C.H.',
    'Varillas J. P.': 'Varillas J.P.',
    'Zayid M. S.': 'Zayid M.S.',
    'Zhang Ze': 'Zhang Ze.',
}

df = pd.read_csv('data/atp_tennis.csv')
df['Date'] = pd.to_datetime(df['Date'])
before = len(df)
players_before = pd.concat([df['Player_1'], df['Player_2']]).nunique()

# 1. Drop rows with placeholder ranks (-1 means no ranking available)
df = df[(df['Rank_1'] > 0) & (df['Rank_2'] > 0)]

# 2. Odds <= 1.0 are impossible (-1 placeholders plus a few data errors).
#    Mark missing rather than dropping - the match is still valid for training,
#    it just can't be used for market comparison.
df.loc[df['Odd_1'] <= 1.0, 'Odd_1'] = np.nan
df.loc[df['Odd_2'] <= 1.0, 'Odd_2'] = np.nan

# 3. Standardise player names: strip whitespace, merge known spelling variants.
#    Winner must use the same mapping or Winner == Player_1 breaks.
for col in ['Player_1', 'Player_2', 'Winner']:
    df[col] = df[col].str.strip().replace(NAME_FIXES)

# Verification
print(f"Rows: {before} -> {len(df)} (dropped {before - len(df)})")
print(f"Min rank: {df[['Rank_1','Rank_2']].min().min()} (expected: 1)")
print(f"Min odds: {df[['Odd_1','Odd_2']].min().min()} (expected: >1.0)")
print(f"Usable odds: {(df['Odd_1'].notna() & df['Odd_2'].notna()).sum()}")
print(f"Players: {players_before} -> {pd.concat([df['Player_1'], df['Player_2']]).nunique()}")

winner_valid = (df['Winner'] == df['Player_1']) | (df['Winner'] == df['Player_2'])
print(f"Winner matches a player: {winner_valid.sum()} / {len(df)} (expected: equal)")

df.to_csv('data/clean_tennis.csv', index=False)
print("Saved")