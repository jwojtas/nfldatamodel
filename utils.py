import os
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd

def download_file(url: str, out_path: str, chunk_size: int = 1024*1024):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(out_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=Path(out_path).name) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return out_path

def read_parquet_local_or_temp(url_or_path: str) -> pd.DataFrame:
    if str(url_or_path).startswith('http'):
        tmp_dir = Path(os.getenv('TMPDIR', '/tmp'))
        tmp = tmp_dir / Path(url_or_path).name
        if not tmp.exists():
            download_file(url_or_path, str(tmp))
        df = pd.read_parquet(str(tmp))
    else:
        df = pd.read_parquet(url_or_path)
    return df
