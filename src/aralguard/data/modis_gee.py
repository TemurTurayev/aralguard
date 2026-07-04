"""MODIS MAIAC AOD (MCD19A2) через Google Earth Engine.

Нужны: earthengine authenticate (один раз) и GEE_PROJECT в .env.
Пример: python -m aralguard.data.modis_gee --start 2021-01-01 --end 2021-12-31
Экспортирует среднесуточный AOD по сетке региона в data_store/modis/*.csv
(для MVP хватает CSV по сетке; GeoTIFF-экспорт — раскомментировать блок Export).
"""
import argparse
import os
from pathlib import Path

import ee
from dotenv import load_dotenv

from aralguard.config import BBOX, GRID_DEG

DATA_DIR = Path(__file__).resolve().parents[3] / "data_store" / "modis"

def init():
    """Инициализация EE: сервисный ключ (GEE_SA_KEY) или личные credentials."""
    load_dotenv()
    project = os.getenv("GEE_PROJECT") or None
    sa_key = os.getenv("GEE_SA_KEY")
    if sa_key:
        key_path = Path(__file__).resolve().parents[3] / sa_key
        if key_path.exists():
            creds = ee.ServiceAccountCredentials(None, str(key_path))
            ee.Initialize(creds, project=project)
            return
    ee.Initialize(project=project)

def daily_aod(start: str, end: str):
    region = ee.Geometry.Rectangle(
        [BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]])
    coll = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
            .filterDate(start, end)
            .select("Optical_Depth_055"))

    def per_day(date):
        date = ee.Date(date)
        img = coll.filterDate(date, date.advance(1, "day")).mean()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region,
            scale=GRID_DEG * 111_000, bestEffort=True)
        return ee.Feature(None, {"date": date.format("YYYY-MM-dd"),
                                 "aod": stats.get("Optical_Depth_055")})

    n_days = ee.Date(end).difference(ee.Date(start), "day")
    dates = ee.List.sequence(0, n_days.subtract(1)).map(
        lambda d: ee.Date(start).advance(d, "day").millis())
    return ee.FeatureCollection(dates.map(per_day))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    a = p.parse_args()
    init()
    fc = daily_aod(a.start, a.end)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"aod_{a.start}_{a.end}.csv"
    # маленькие выборки можно тянуть напрямую:
    rows = fc.getInfo()["features"]
    with open(out, "w") as f:
        f.write("date,aod\n")
        for r in rows:
            pr = r["properties"]
            f.write(f"{pr['date']},{pr.get('aod', '')}\n")
    print("saved:", out, f"({len(rows)} days)")
