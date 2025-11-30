import pandas as pd
import joblib
from pathlib import Path
from .feature_engineering import build_features_pipeline_from_player_stats
from .data_loading import load_player_stats, load_schedule

def load_models(models_dir='models'):
    models = {}
    for p in Path(models_dir).glob('model_*.joblib'):
        name = p.stem.replace('model_','')
        models[name] = joblib.load(p)
    return models

def predict_next_games(models_dict, player_stats_df, next_games_df, team_def_df=None, games_df=None):
    # To compute fresh rolling features, we append next_games rows (with NaN targets) to player_stats_df
    hist = player_stats_df.copy()
    # ensure next_games_df has player_id, player_name, game_id, game_date, team, opp
    combined = pd.concat([hist, next_games_df], ignore_index=True, sort=False)
    features_df, feature_cols = build_features_pipeline_from_player_stats(combined, team_def_df, games_df)
    # filter to next game rows by matching game_id in next_games_df
    next_feat = features_df[features_df['game_id'].isin(next_games_df['game_id'])]
    preds = next_feat[['player_id','player_name','game_id','game_date']].copy()
    for target, meta in models_dict.items():
        model = meta['model']
        feature_cols = meta['feature_cols']
        X = next_feat[feature_cols].fillna(0.0)
        preds[f'pred_{target}'] = model.predict(X)
        # clip negatives for count/regression targets
        preds[f'pred_{target}'] = preds[f'pred_{target}'].clip(lower=0.0)
    return preds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', default='models')
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--next_games_csv', required=True)
    parser.add_argument('--out', default='predictions.csv')
    args = parser.parse_args()
    models = load_models(args.models_dir)
    ps = load_player_stats(args.year)
    next_games = pd.read_csv(args.next_games_csv, parse_dates=['game_date'])
    preds = predict_next_games(models, ps, next_games)
    preds.to_csv(args.out, index=False)
    print('Wrote predictions to', args.out)
