"""Загрузка прогнозов пыли CAMS (Copernicus ADS).

Нужен ключ ADS_API_KEY в .env (ads.atmosphere.copernicus.eu).
Пример: python -m aralguard.data.cams_download --days 5
"""
import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import cdsapi
from dotenv import load_dotenv

from aralguard.config import BBOX

DATA_DIR = Path(__file__).resolve().parents[3] / "data_store" / "cams"

def download(days: int = 5, out_dir: Path = DATA_DIR) -> Path:
    load_dotenv()
    key = os.getenv("ADS_API_KEY")
    if not key:
        raise SystemExit("ADS_API_KEY не задан в .env — см. README")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api", key=key)
    target = out_dir / f"cams_dust_last{days}d.nc"
    client.retrieve(
        "cams-global-atmospheric-composition-forecasts",
        {
            "variable": [
                "dust_aerosol_optical_depth_550nm",
                "particulate_matter_10um",
                "particulate_matter_2.5um",
            ],
            "date": [f"{date.today() - timedelta(days=days)}/{date.today() - timedelta(days=1)}"],
            "time": ["00:00", "12:00"],
            "leadtime_hour": [str(h) for h in range(0, 73, 3)],
            "type": ["forecast"],
            "area": [BBOX["north"], BBOX["west"], BBOX["south"], BBOX["east"]],
            "data_format": "netcdf_zip",
        },
        str(target),
    )
    print("saved:", target)
    return target

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=5)
    download(days=p.parse_args().days)
