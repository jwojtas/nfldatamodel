import pandas as pd
import numpy as np
from pathlib import Path

def load_game_logs_from_player_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = player_stats_df.copy()
    # unify common column names
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ['player', 'player_name']:
            rename_map[c] = 'player_name'
        if lc in ['player_id', 'gsis_id']:
            rename_map[c] = 'player_id'
        if lc in ['receiving_yards','rec_yards']:
            rename_map[c] = 'rec_yards'
        if lc in ['receiving_tds','rec_tds']:
            rename_map[c] = 'rec_tds'
        if lc in ['targets']:
            rename_map[c] = 'targets'
        if lc in ['receptions','rec']:
            rename_map[c] = 'receptions'
        if lc in ['rushing_yards','rush_yards']:
            rename_map[c] = 'rush_yards'
        if lc in ['rush_attempts','carries','rush_att']:
            rename_map[c] = 'carries'
        if lc in ['snaps','snap_cnt','snap_count']:
            rename_map[c] = 'snaps'
        if lc in ['snap_pct','snap_percent']:
            rename_map[c] = 'snap_pct'
        if lc in ['routes','route','routes_run']:
            rename_map[c] = 'routes'
    df = df.rename(columns=rename_map)
    # ensure essential numeric columns exist
    for col in ['targets','receptions','rec_yards','rec_tds','carries','rush_yards','snap_pct','snaps','routes']:
        if col not in df.columns:
            df[col] = 0.0
    # ensure game_date exists (may be NaT if not provided)
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        df['game_date'] = pd.NaT
    return df

def add_rolling_features(df: pd.DataFrame, group_col: str = 'player_id', windows=[3,5,8]):
    df = df.sort_values([group_col,'game_date'])
    for w in windows:
        for col in ['targets','receptions','rec_yards','carries','rush_yards','snap_pct','routes']:
            mean_col = f"{col}_roll{w}_mean"
            std_col = f"{col}_roll{w}_std"
            ewm_col = f"{col}_ewm{w}"
            df[mean_col] = df.groupby(group_col)[col].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[std_col] = df.groupby(group_col)[col].rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            df[ewm_col] = df.groupby(group_col)[col].apply(lambda s: s.ewm(span=w, adjust=False).mean()).reset_index(level=0, drop=True)
    return df

def add_opponent_features(df: pd.DataFrame, team_def_df: pd.DataFrame = None):
    df = df.copy()
    if team_def_df is None:
        # add placeholder columns
        df['opp_def_epa_per_play'] = np.nan
        df['opp_yac_allowed'] = np.nan
        df['opp_pass_def_rank'] = np.nan
        df['opp_run_def_rank'] = np.nan
        return df
    team_def_df = team_def_df.copy()
    if 'game_date' in team_def_df.columns:
        team_def_df['game_date'] = pd.to_datetime(team_def_df['game_date'])
        merged = df.merge(team_def_df, left_on=['opp','game_date'], right_on=['team','game_date'], how='left', suffixes=('','_def'))
    else:
        merged = df.merge(team_def_df, left_on='opp', right_on='team', how='left', suffixes=('','_def'))
    # standardize expected names
    rename_map = {}
    for c in merged.columns:
        lc = c.lower()
        if 'epa' in lc and 'def' in lc:
            rename_map[c] = 'opp_def_epa_per_play'
        if 'yac' in lc:
            rename_map[c] = 'opp_yac_allowed'
        if 'pass' in lc and 'rank' in lc:
            rename_map[c] = 'opp_pass_def_rank'
        if 'run' in lc and 'rank' in lc:
            rename_map[c] = 'opp_run_def_rank'
    merged = merged.rename(columns=rename_map)
    return merged

def add_game_environment(df: pd.DataFrame, games_df: pd.DataFrame = None):
    df = df.copy()
    if games_df is None:
        df['game_total'] = np.nan
        df['game_spread'] = np.nan
        df['pace'] = np.nan
        df['weather_wind'] = np.nan
        return df
    g = games_df.copy()
    if 'game_date' in g.columns:
        g['game_date'] = pd.to_datetime(g['game_date'])
        merged = df.merge(g, on=['game_id','game_date'], how='left', suffixes=('','_g'))
    else:
        merged = df.merge(g, on='game_id', how='left', suffixes=('','_g'))
    # normalize common names
    if 'total' in merged.columns:
        merged['game_total'] = merged['total']
    if 'spread' in merged.columns:
        merged['game_spread'] = merged['spread']
    if 'pace' not in merged.columns and 'plays' in merged.columns:
        merged['pace'] = merged['plays']
    return merged

def finalize_features(df: pd.DataFrame, feature_cols: list = None):
    df = df.copy()
    # fill numeric NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    # add some interactions
    df['targets_x_team_total'] = df.get('targets_roll3_mean',0.0) * df.get('team_implied_total',0.0)
    df['routes_x_snap'] = df.get('routes_roll3_mean',0.0) * df.get('snap_pct_roll3_mean',0.0)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if ('roll' in c or 'ewm' in c) and ('mean' in c or 'ewm' in c)]
        for extra in ['team_implied_total','pace','weather_wind','opp_def_epa_per_play','opp_yac_allowed']:
            if extra in df.columns:
                feature_cols.append(extra)
        feature_cols = list(dict.fromkeys(feature_cols))[:200]
    df_features = df[feature_cols + ['player_id','game_id','game_date','player_name']].copy()
    return df_features, feature_cols

def build_features_pipeline_from_player_stats(player_stats_df, team_def_df=None, games_df=None):
    logs = load_game_logs_from_player_stats(player_stats_df)
    logs = add_rolling_features(logs)
    logs = add_opponent_features(logs, team_def_df)
    logs = add_game_environment(logs, games_df)
    features_df, feature_cols = finalize_features(logs)
    return features_df, feature_cols
