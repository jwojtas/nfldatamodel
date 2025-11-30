import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score
import xgboost as xgb
from sklearn.linear_model import PoissonRegressor, LogisticRegression

from .feature_engineering import build_features_pipeline_from_player_stats
from .data_loading import load_player_stats, load_schedule

def train_xgb_reg(X, y, params=None):
    if params is None:
        params = { 'n_estimators':300, 'learning_rate':0.05, 'max_depth':6, 'subsample':0.9, 'colsample_bytree':0.9, 'random_state':42 }
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model

def train_poisson(X, y):
    model = PoissonRegressor(alpha=1e-4, max_iter=500)
    model.fit(X, y)
    return model

def train_logistic(X, y):
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model

def choose_model_type(target):
    t = target.lower()
    if 'yards' in t:
        return 'regression'
    if t in ['receptions','targets','carries','rush_attempts']:
        return 'poisson'
    if 'td' in t or 'touchdown' in t:
        return 'poisson'
    if t.startswith('is_') or t.startswith('has_') or t in ['scored_td']:
        return 'classification'
    return 'regression'

def time_series_cv(X, y, model_type, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if model_type == 'regression':
            m = train_xgb_reg(X_tr, y_tr)
            preds = m.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))
        elif model_type == 'poisson':
            m = train_poisson(X_tr, y_tr)
            preds = m.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))
        else:
            m = train_logistic(X_tr, y_tr)
            probs = m.predict_proba(X_val)[:,1]
            aucs.append(roc_auc_score(y_val, probs))
    return np.mean(maes) if maes else np.nan, np.mean(aucs) if aucs else np.nan

def train_multi_targets(player_stats_df, targets, team_def_df=None, games_df=None, out_dir='models'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    features_df, feature_cols = build_features_pipeline_from_player_stats(player_stats_df, team_def_df, games_df)
    # ensure df sorted chronologically
    features_df = features_df.sort_values('game_date')
    results = {}
    for target in targets:
        # join target column to features by player_id + game_date
        if target not in player_stats_df.columns:
            print(f"Warning: target {target} not in player_stats_df columns; creating zeros")
            player_stats_df[target] = 0.0
        merged = features_df.merge(player_stats_df[['player_id','game_date',target]], on=['player_id','game_date'], how='left')
        X = merged[feature_cols].fillna(0.0)
        y = merged[target].fillna(0.0)
        model_type = choose_model_type(target)
        print(f"Training {target} as {model_type} (N={len(y)})")
        cv_mae, cv_auc = time_series_cv(X, y, model_type)
        print(f"CV MAE: {cv_mae}, CV AUC: {cv_auc}")
        # train final
        if model_type == 'regression':
            model = train_xgb_reg(X, y)
        elif model_type == 'poisson':
            model = train_poisson(X, y)
        else:
            model = train_logistic(X, y)
        out_path = Path(out_dir) / f"model_{target}.joblib"
        joblib.dump({'model': model, 'feature_cols': feature_cols, 'model_type': model_type}, out_path)
        results[target] = { 'path': str(out_path), 'cv_mae': float(cv_mae) if not np.isnan(cv_mae) else None, 'cv_auc': float(cv_auc) if not np.isnan(cv_auc) else None }
        print(f"Saved model for {target} to {out_path}\n")
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--targets', type=str, required=True, help='comma separated targets like rec_yards,receptions,targets')
    parser.add_argument('--out_dir', type=str, default='models')
    args = parser.parse_args()
    targets = args.targets.split(',')
    print('Loading player stats for year', args.year)
    ps = load_player_stats(args.year)
    # optional: load schedules/def stats if available
    train_multi_targets(ps, targets, team_def_df=None, games_df=None, out_dir=args.out_dir)
