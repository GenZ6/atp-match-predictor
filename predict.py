import json
import pandas as pd
import numpy as np
import joblib

state = json.load(open('data/elo_state.json'))
bundle = joblib.load('data/model.pkl')
model, scaler, FEATURES = bundle['model'], bundle['scaler'], bundle['features']

W_SURF = 0.4
SURFACE_KEY = {'Hard': 'hard', 'Clay': 'clay', 'Grass': 'grass'}

def blended(p, surf_key):
    return W_SURF * p[surf_key] + (1 - W_SURF) * p['general']

def predict(name1, name2, surface='Hard'):
    if name1 not in state:
        raise ValueError(f"Unknown player: {name1}")
    if name2 not in state:
        raise ValueError(f"Unknown player: {name2}")
    if surface not in SURFACE_KEY:
        raise ValueError(f"Surface must be one of {list(SURFACE_KEY)}")

    p1, p2 = state[name1], state[name2]
    sk = SURFACE_KEY[surface]

    elo_diff  = blended(p1, sk) - blended(p2, sk)
    rank_diff = np.log(p2['rank']) - np.log(p1['rank'])
    gap_diff  = 0                      # both assumed match-ready for a live prediction
    form_diff = p1['form'] - p2['form']

    row = pd.DataFrame([[elo_diff, rank_diff, gap_diff, form_diff]], columns=FEATURES)
    prob = model.predict_proba(scaler.transform(row))[0][1]

    print(f"\n{name1}  vs  {name2}   ({surface})")
    print(f"  {name1}: {prob*100:.1f}%")
    print(f"  {name2}: {(1-prob)*100:.1f}%")
    return prob

if __name__ == '__main__':
    predict('Sinner J.', 'Alcaraz C.', 'Clay')
    predict('Sinner J.', 'Alcaraz C.', 'Grass')
    predict('Djokovic N.', 'Sinner J.', 'Hard')