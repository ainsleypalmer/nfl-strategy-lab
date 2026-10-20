"""Officially named public nflverse URLs, plus CSV-upload fallback."""
from __future__ import annotations
from io import BytesIO
import pandas as pd
import requests

URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv'

COLUMNS = [
    'season', 'game_id', 'play_id', 'game_date', 'season_type', 'posteam', 'defteam',
    'play_type', 'down', 'ydstogo', 'yards_gained', 'epa', 'yardline_100',
    'complete_pass','pass_attempt','rush_attempt','receiver_player_name',
    'rusher_player_name','passer_player_name','interception','fumble_lost',
    'touchdown','success','air_yards','pass_length','first_down','qb_scramble','sack',
    'receiving_yards','rushing_yards',
]


def parse_csv(csv):
    # usecols callable allows legitimate variations in source's optional columns.
    return pd.read_csv(csv, usecols=lambda c: c in COLUMNS, low_memory=False)


def fetch_nflverse(years=(2025, 2026)):
    frames = []
    for year in years:
        url = URL.format(year=year)
        r = requests.get(url, timeout=75, headers={'User-Agent': 'NFLStrategyLab-education/0.1'})
        r.raise_for_status()
        if not r.content or len(r.content) < 1000:
            raise RuntimeError(f'Unexpectedly small file for {year}.')
        frame = parse_csv(BytesIO(r.content))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
