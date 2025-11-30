import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .utils import read_parquet_local_or_temp

log = logging.getLogger(__name__)

# Try optional helpers
_HAS_NFLREADPY = False
_HAS_NFL_DATA_PY = False
_HAS_NFLFASTPY = False
try:
    import nflreadpy
    _HAS_NFLREADPY = True
except Exception:
    pass
try:
    import nfl_data_py
    _HAS_NFL_DATA_PY = True
except Exception:
    pass
try:
    import nflfastpy
    _HAS_NFLFASTPY = True
except Exception:
    pass

NFLVERSE_BASE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"

def _build_release_url(dataset: str, year: int = None) -> str:
    if dataset == 'play_by_play':
        if year is None:
            raise ValueError('year required for play_by_play')
        return f"{NFLVERSE_BASE_RELEASE}/pbp/play_by_play_{year}.parquet"
    if dataset == 'player_stats':
        if year is None:
            raise ValueError('year required for player_stats')
        return f"{NFLVERSE_BASE_RELEASE}/player_stats/player_stats_{year}.parquet"
    if year:
        return f"{NFLVERSE_BASE_RELEASE}/{dataset}/{dataset}_{year}.parquet"
    return f"{NFLVERSE_BASE_RELEASE}/{dataset}/{dataset}.parquet"

def load_player_stats(year: int, cache_path: Optional[str] = 'data/raw/player_stats') -> pd.DataFrame:
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    local_file = Path(cache_path) / f'player_stats_{year}.parquet'
    if _HAS_NFLREADPY:
        try:
            log.info('Loading player stats via nflreadpy')
            # nflreadpy API may vary; attempt common function names
            df = nflreadpy.load_player_stats(seasons=[year])
            return df
        except Exception as e:
            log.warning('nflreadpy failed: %s', e)
    if _HAS_NFL_DATA_PY:
        try:
            log.info('Loading player stats via nfl_data_py')
            df = nfl_data_py.import_stats(season=year)
            return df
        except Exception as e:
            log.warning('nfl_data_py failed: %s', e)
    url = _build_release_url('player_stats', year=year)
    log.info('Downloading player_stats from %s', url)
    df = read_parquet_local_or_temp(url)
    return df

def load_play_by_play(year: int, cache_path: Optional[str] = 'data/raw/pbp') -> pd.DataFrame:
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    local_file = Path(cache_path) / f'play_by_play_{year}.parquet'
    if _HAS_NFLREADPY:
        try:
            log.info('Loading pbp via nflreadpy')
            df = nflreadpy.load_pbp(seasons=[year])
            return df
        except Exception as e:
            log.warning('nflreadpy.load_pbp failed: %s', e)
    if _HAS_NFLFASTPY:
        try:
            log.info('Loading pbp via nflfastpy')
            df = nflfastpy.load_pbp_data(year=year)
            return df
        except Exception as e:
            log.warning('nflfastpy failed: %s', e)
    url = _build_release_url('play_by_play', year=year)
    log.info('Downloading pbp from %s', url)
    df = read_parquet_local_or_temp(url)
    return df

def load_schedule(year:int, cache_path:Optional[str]='data/raw/schedules') -> pd.DataFrame:
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    # try optional loaders
    try:
        if _HAS_NFLREADPY:
            return nflreadpy.load_schedules(seasons=[year])
    except Exception:
        pass
    url = _build_release_url('schedules', year=year)
    try:
        return read_parquet_local_or_temp(url)
    except Exception:
        # fallback: derive from pbp
        pbp = load_play_by_play(year)
        cols = ['game_id','game_date','home_team','away_team']
        if set(cols).issubset(set(pbp.columns)):
            return pbp[cols].drop_duplicates()
        raise RuntimeError('Unable to load schedule for year %s' % year)
