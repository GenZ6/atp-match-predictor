import pandas as pd

matches = pd.read_csv('data/cleaned_matches.csv', low_memory=False)

# How many matches per surface?
print("Matches by surface:")
print(matches['surface'].value_counts())

# How many matches per year?
matches['tourney_date'] = pd.to_datetime(matches['tourney_date'], format='%Y%m%d')
matches['year'] = matches['tourney_date'].dt.year
print("\nMatches per year (last 5 years):")
print(matches['year'].value_counts().sort_index().tail())
print(matches[matches['year'] == 2024]['tourney_date'].max())