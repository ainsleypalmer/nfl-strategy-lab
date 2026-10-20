"""Educational NFL play-by-play analytics and historical-resampling engine.

This software studies play selection and game strategy. It is not a model for
wagering, player props, or independently verified player projections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

REQUIRED = {
    'season', 'game_date', 'season_type', 'posteam', 'defteam',
    'play_type', 'down', 'ydstogo', 'yards_gained', 'epa', 'yardline_100'
}
OPTIONAL = {
    'complete_pass', 'pass_attempt', 'rush_attempt', 'receiver_player_name',
    'rusher_player_name', 'passer_player_name', 'interception', 'fumble_lost',
    'touchdown', 'success', 'game_id', 'play_id', 'air_yards',
    'pass_length', 'first_down', 'qb_scramble', 'sack'
}

class DataQualityError(ValueError):
    pass


def prepare_plays(frame: pd.DataFrame, *, cutoff: str = '2026-10-19', seasons=(2025, 2026)) -> pd.DataFrame:
    """Standardize public nflfastR PBP and prevent look-ahead beyond cutoff.

    cutoff is exclusive. Real games are defined by the supplied game_date.
    For 2026-10-19, October 19 game never enters the model.
    """
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise DataQualityError(f'Missing mandatory nflverse columns: {", ".join(sorted(missing))}')
    df = frame.copy()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    for col in ('season', 'down', 'ydstogo', 'yards_gained', 'epa', 'yardline_100'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in OPTIONAL - set(df.columns):
        df[col] = np.nan
    cut = pd.Timestamp(cutoff)
    df = df.loc[
        df['season'].isin(seasons)
        & df['game_date'].notna()
        & (df['game_date'] < cut)
        & (df['season_type'] == 'REG')
        & df['posteam'].notna()
        & df['defteam'].notna()
        & (df['play_type'].isin(['pass', 'run']))
        & df['down'].between(1, 4)
        & df['ydstogo'].between(1, 35)
        & df['yardline_100'].between(1, 99)
        & df['yards_gained'].notna()
        & df['epa'].notna()
    ].copy()
    if df.empty:
        raise DataQualityError('No usable regular-season plays before the selected cutoff.')
    df['play_type'] = df['play_type'].map({'pass': 'Dropback / pass', 'run': 'Run / scramble'})
    df['successful_play'] = df['epa'] > 0
    df['converted'] = ((df['yards_gained'] >= df['ydstogo']) | (pd.to_numeric(df['touchdown'], errors='coerce').fillna(0) == 1))
    df['explosive'] = (df['yards_gained'] >= 20)
    df['turnover'] = ((pd.to_numeric(df['interception'], errors='coerce').fillna(0) == 1)
                      | (pd.to_numeric(df['fumble_lost'], errors='coerce').fillna(0) == 1))
    return df


def summary(df: pd.DataFrame, offense: str, defense: str) -> pd.DataFrame:
    """Compare observed offensive tendencies and defensive allowances in current year."""
    season = int(df['season'].max())
    rows = []
    for name, column, code in [('Offense', 'posteam', offense), ('Opposing defense', 'defteam', defense)]:
        subset = df.loc[(df[column] == code) & (df['season'] == season)]
        for play in ['Dropback / pass', 'Run / scramble']:
            p = subset[subset['play_type'] == play]
            if p.empty:
                continue
            rows.append({
                'Unit': name, 'Play type': play, 'Plays': int(len(p)),
                'Yards/play': round(float(p['yards_gained'].mean()), 2),
                'EPA/play': round(float(p['epa'].mean()), 3),
                'Positive EPA %': round(float(p['successful_play'].mean()*100), 1),
                'Explosive %': round(float(p['explosive'].mean()*100), 1),
            })
    return pd.DataFrame(rows)


def _context_pool(df: pd.DataFrame, play_type: str, down: int, to_go: int, yardline: int) -> pd.DataFrame:
    sub = df[df['play_type'] == play_type]
    checks = [
        ((sub['down'] == down) & (sub['ydstogo'].between(max(1, to_go-3), to_go+3)) &
         (sub['yardline_100'].between(max(1, yardline-25), min(99, yardline+25)))),
        ((sub['down'] == down) & sub['ydstogo'].between(max(1, to_go-5), to_go+5)),
        (sub['down'] == down),
        pd.Series(True, index=sub.index),
    ]
    for mask in checks:
        matches = sub.loc[mask]
        if len(matches) >= 45:
            return matches
    return sub


def _weights(p: pd.DataFrame, offense: str, defense: str) -> np.ndarray:
    # Heuristic historical resampling weights; NOT calibrated predictive probabilities.
    most_recent = int(p['season'].max())
    year_w = np.where(p['season'].to_numpy() == most_recent, 2.2, 1.0)
    off_w = np.where(p['posteam'].to_numpy() == offense, 2.6, 1.0)
    def_w = np.where(p['defteam'].to_numpy() == defense, 2.0, 1.0)
    w = year_w * off_w * def_w
    return w / w.sum()


@dataclass
class Scenario:
    category: str
    source_plays: int
    sampled_yards: np.ndarray
    sampled_epa: np.ndarray
    sampled_success: np.ndarray
    sampled_convert: np.ndarray
    sampled_explosive: np.ndarray
    yardline: int
    to_go: int
    seed: int

    @property
    def overview(self) -> dict:
        return {
            'Play type': self.category,
            'Historical sample': self.source_plays,
            'Typical gain (median yards)': round(float(np.median(self.sampled_yards)), 1),
            'Middle 80% yards': (round(float(np.quantile(self.sampled_yards, .1)), 1),
                                 round(float(np.quantile(self.sampled_yards, .9)), 1)),
            'Avg EPA/play in resample': round(float(np.mean(self.sampled_epa)), 3),
            'Positive EPA share': round(float(np.mean(self.sampled_success))*100, 1),
            'First down / TD share': round(float(np.mean(self.sampled_convert))*100, 1),
            '20+ yard play share': round(float(np.mean(self.sampled_explosive))*100, 1),
        }


def scenario_sample(df: pd.DataFrame, offense: str, defense: str, *,
                    play_type: str, down: int, to_go: int, yardline: int,
                    n: int = 500, seed: int = 42) -> Scenario:
    if play_type not in ['Dropback / pass', 'Run / scramble']:
        raise ValueError('Unsupported play category')
    if not (1 <= down <= 4 and 1 <= to_go <= 35 and 1 <= yardline <= 99):
        raise ValueError('Invalid game situation')
    if not 1 <= n <= 20000:
        raise ValueError('Sample size outside supported range')
    sub = _context_pool(df, play_type, down, to_go, yardline)
    if len(sub) < 20:
        raise DataQualityError(f'Not enough historical plays for {play_type} ({len(sub)} available).')
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sub), size=n, replace=True, p=_weights(sub, offense, defense))
    plays = sub.iloc[idx]
    gain = plays['yards_gained'].to_numpy(dtype=float)
    # Conversion is situation-dependent; compare sampled gain with CURRENT yards to go.
    to_convert = (gain >= to_go) | (pd.to_numeric(plays['touchdown'], errors='coerce').fillna(0).to_numpy() == 1)
    return Scenario(play_type, len(sub), gain, plays['epa'].to_numpy(dtype=float),
                    (plays['epa'].to_numpy(dtype=float) > 0), to_convert,
                    (gain >= 20), yardline, to_go, seed)


def automatic_comparison(df: pd.DataFrame, offense: str, defense: str, *,
                         down: int, to_go: int, yardline: int, n=500, seed=42):
    results = [scenario_sample(df, offense, defense, play_type=c, down=down,
                               to_go=to_go, yardline=yardline, n=n,
                               seed=seed+i) for i,c in enumerate(['Dropback / pass','Run / scramble'])]
    table = pd.DataFrame([r.overview for r in results])
    table['Note'] = np.where(table['Historical sample'] < 80,
                             'Low context sample', 'Observational only')
    return table, results


def player_involvement(df: pd.DataFrame, offense: str, *, season: Optional[int] = None) -> pd.DataFrame:
    """Observed target/carry involvement. No injury, lineup or player predictions."""
    season = season or int(df['season'].max())
    sub = df[(df['posteam'] == offense) & (df['season'] == season)].copy()
    counts = []
    for col, role, condition in [('receiver_player_name', 'Pass targets', 'Dropback / pass'),
                                 ('rusher_player_name', 'Rushing opportunities', 'Run / scramble')]:
        part = sub.loc[(sub['play_type'] == condition) & sub[col].notna()].copy()
        part = part[part[col].astype(str).str.strip().ne('')]
        if part.empty:
            continue
        ct = part.groupby(col, dropna=True).agg(
            Plays=(col, 'count'),
            Recorded_yards=('yards_gained', 'sum'),
        ).reset_index().rename(columns={col:'Player','Recorded_yards':'Play yards'})
        ct['Role'] = role
        ct['Share of recorded opportunities %'] = (ct['Plays'] / len(part) * 100).round(1)
        counts.append(ct)
    if not counts:
        return pd.DataFrame(columns=['Player','Role','Plays','Play yards','Share of recorded opportunities %'])
    return (pd.concat(counts, ignore_index=True)
            .sort_values(['Role','Plays'], ascending=[True, False]).reset_index(drop=True))


def game_coverage(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    parts = []
    for team in ['SF', 'WAS']:
        for year in sorted(df['season'].dropna().unique()):
            sub = df[(df['season'] == year) & ((df['posteam'] == team) | (df['defteam'] == team))]
            parts.append({'Team': team, 'Season': int(year), 'Games included': sub['game_id'].nunique()
                         if sub['game_id'].notna().any() else 'game_id missing',
                         'Plays included': int(len(sub)), 'Latest game in data':
                         str(sub['game_date'].max().date()) if len(sub) else 'â',
                         'Cutoff (exclusive)': cutoff})
    return pd.DataFrame(parts)
