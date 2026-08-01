import pandas as pd
import json
from collections import defaultdict, deque

df = pd.read_csv('data/clean_tennis.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

K = 32
W_SURF = 0.4

general = defaultdict(lambda: 1500.0)
surface = {s: defaultdict(lambda: 1500.0) for s in df['Surface'].unique()}
last_played = {}
recent = defaultdict(lambda: deque(maxlen=10))

cols = {c: [] for c in ['Elo_1', 'Elo_2', 'SElo_1', 'SElo_2',
                        'Blend_1', 'Blend_2', 'Gap_1', 'Gap_2',
                        'Form_1', 'Form_2']}

def update(store, a, b, s_a, k=K):
    ra, rb = store[a], store[b]
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    store[a] = ra + k * (s_a - ea)
    store[b] = rb + k * ((1 - s_a) - (1 - ea))

for row in df.itertuples(index=False):
    p1, p2, surf = row.Player_1, row.Player_2, row.Surface
    s = surface[surf]

    g1, g2 = general[p1], general[p2]
    s1r, s2r = s[p1], s[p2]

    cols['Elo_1'].append(g1);   cols['Elo_2'].append(g2)
    cols['SElo_1'].append(s1r); cols['SElo_2'].append(s2r)
    cols['Blend_1'].append(W_SURF * s1r + (1 - W_SURF) * g1)
    cols['Blend_2'].append(W_SURF * s2r + (1 - W_SURF) * g2)

    cols['Gap_1'].append((row.Date - last_played[p1]).days if p1 in last_played else -1)
    cols['Gap_2'].append((row.Date - last_played[p2]).days if p2 in last_played else -1)

    cols['Form_1'].append(sum(recent[p1]) / len(recent[p1]) if recent[p1] else 0.5)
    cols['Form_2'].append(sum(recent[p2]) / len(recent[p2]) if recent[p2] else 0.5)

    s1 = 1.0 if row.Winner == p1 else 0.0

    update(general, p1, p2, s1)
    update(s, p1, p2, s1)
    last_played[p1] = last_played[p2] = row.Date
    recent[p1].append(s1)
    recent[p2].append(1 - s1)

for c, v in cols.items():
    df[c] = v

df.to_csv('data/tennis_elo.csv', index=False)

# --- Accuracy check ---
ev = df[df['Date'].dt.year >= 2005].copy()
won = ev['Winner'] == ev['Player_1']
print("Blended Elo accuracy:", round(((ev['Blend_1'] > ev['Blend_2']) == won).mean(), 4))

# --- Save final state for the live predictor ---
# df is sorted chronologically, so the last write per player is their latest.
latest = {}
for row in df.itertuples(index=False):
    latest[row.Player_1] = {'rank': int(row.Rank_1), 'date': str(row.Date.date())}
    latest[row.Player_2] = {'rank': int(row.Rank_2), 'date': str(row.Date.date())}

state = {}
for player in general:
    state[player] = {
        'general': general[player],
        'hard':   surface.get('Hard',  {}).get(player, 1500.0),
        'clay':   surface.get('Clay',  {}).get(player, 1500.0),
        'grass':  surface.get('Grass', {}).get(player, 1500.0),
        'form':   sum(recent[player]) / len(recent[player]) if recent[player] else 0.5,
        'rank':   latest.get(player, {}).get('rank', 500),
        'last_date': latest.get(player, {}).get('date', None),
    }

with open('data/elo_state.json', 'w') as f:
    json.dump(state, f)

print(f"Saved state for {len(state)} players")