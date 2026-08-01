import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st

@st.cache_resource
def load():
    state = json.load(open('data/elo_state.json'))
    bundle = joblib.load('data/model.pkl')
    return state, bundle

state, bundle = load()
model, scaler, FEATURES = bundle['model'], bundle['scaler'], bundle['features']

W_SURF = 0.4
SURFACE_KEY = {'Hard': 'hard', 'Clay': 'clay', 'Grass': 'grass'}

def blended(p, sk):
    return W_SURF * p[sk] + (1 - W_SURF) * p['general']

def predict(name1, name2, surface):
    p1, p2 = state[name1], state[name2]
    sk = SURFACE_KEY[surface]
    feats = [[
        blended(p1, sk) - blended(p2, sk),
        np.log(p2['rank']) - np.log(p1['rank']),
        0,
        p1['form'] - p2['form'],
    ]]
    row = pd.DataFrame(feats, columns=FEATURES)
    return model.predict_proba(scaler.transform(row))[0][1]

# --- Interface ---
st.title("ATP Tennis Match Predictor")
st.caption("Men's singles outcomes from a from-scratch Elo system. "
           "Trained 2005-2022, validated 2023-2026 (65% accuracy).")

# Only show players active since the cutoff. The model still knows everyone -
# this filters the dropdown for usability, not the ratings underneath.
CUTOFF = "2025-01-01"
active = [n for n in state
          if state[n].get('last_date') and state[n]['last_date'] >= CUTOFF]
players = sorted(active, key=lambda n: -state[n]['general'])

col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Player 1", players, index=0)
with col2:
    p2 = st.selectbox("Player 2", players, index=1)

surface = st.radio("Surface", ["Hard", "Clay", "Grass"], horizontal=True)

if st.button("Predict", type="primary"):
    if p1 == p2:
        st.warning("Pick two different players.")
    else:
        prob = predict(p1, p2, surface)
        st.subheader("Prediction")
        c1, c2 = st.columns(2)
        c1.metric(p1, f"{prob*100:.1f}%")
        c2.metric(p2, f"{(1-prob)*100:.1f}%")
        st.progress(prob)

st.caption("Limitations: no injury/withdrawal data; assumes both players "
           "match-ready; tour-level singles only. Ratings as of July 2026.")