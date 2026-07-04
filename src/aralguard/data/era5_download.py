"""ERA5: ветер 10м, порывы, высота погранслоя, влажность почвы (Copernicus CDS).

Нужен CDS_API_KEY в .env. Пример: python -m aralguard.data.era5_download --year 2021
"""
import argparse
import os
from pathlib import Path

import cdsapi
from dotenv import load_dotenv

from aralguard.config import BBOX

DATA_DIR = Path(__file__).resolve().parents[3] / "data_store" / "era5"

VARS = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust", "boundary_layer_height",
    "volumetric_soil_water_layer_1", "total_precipitation",
]

def download(year: int, out_dir: Path = DATA_DIR) -> Path:
    load_dotenv()
    key = os.getenv("CDS_API_KEY")
    if not key:
        raise SystemExit("CDS_API_KEY не задан в .env — см. README")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=key)
    target = out_dir / f"era5_{year}.nc"
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARS,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "area": [BBOX["north"], BBOX["west"], BBOX["south"], BBOX["east"]],
            "data_format": "netcdf",
        },
        str(target),
    )
    print("saved:", target)
    return target

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    download(p.parse_args().year)
