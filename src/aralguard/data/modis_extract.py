"""Батчевая выгрузка полей MODIS MAIAC AOD (MCD19A2) через ee.data.computePixels.

Один запрос = один месяц (все дни как каналы) => ~72 запроса на 2019-2024,
щадит квоту Community. Выход: data_store/modis/aod_YYYY-MM.npz
  aod   : float32 [days, H, W], NaN = нет данных (облака/нет пролёта)
  dates : список 'YYYY-MM-DD'

Запуск: python -m aralguard.data.modis_extract --start 2019-01 --end 2024-12
"""
import argparse
import calendar
import os
from datetime import date
from pathlib import Path

import numpy as np
import ee
from dotenv import load_dotenv

from aralguard.config import BBOX, GRID_DEG

DATA_DIR = Path(__file__).resolve().parents[3] / "data_store" / "modis"
FILL = -999.0
W = round((BBOX["east"] - BBOX["west"]) / GRID_DEG)   # 80
H = round((BBOX["north"] - BBOX["south"]) / GRID_DEG) # 60


def init():
    load_dotenv()
    project = os.getenv("GEE_PROJECT") or None
    sa_key = os.getenv("GEE_SA_KEY")
    if sa_key:
        key_path = Path(__file__).resolve().parents[3] / sa_key
        if key_path.exists():
            ee.Initialize(ee.ServiceAccountCredentials(None, str(key_path)),
                          project=project)
            return
    ee.Initialize(project=project)


def month_image(year: int, month: int) -> tuple[ee.Image, list[str]]:
    """Изображение с каналом-днём для каждого дня месяца."""
    coll = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
            .select("Optical_Depth_055"))
    ndays = calendar.monthrange(year, month)[1]
    bands, names = [], []
    for d in range(1, ndays + 1):
        day = f"{year}-{month:02d}-{d:02d}"
        nxt = ee.Date(day).advance(1, "day")
        sub = coll.filterDate(day, nxt)
        img = ee.Image(ee.Algorithms.If(
            sub.size().gt(0),
            sub.mean(),
            ee.Image.constant(FILL)))
        name = f"d{d:02d}"
        bands.append(ee.Image(img).rename(name).toFloat().unmask(FILL))
        names.append(day)
    return ee.Image.cat(bands), names


def fetch_month(year: int, month: int) -> Path:
    img, dates = month_image(year, month)
    req = {
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": W, "height": H},
            "affineTransform": {
                "scaleX": GRID_DEG, "shearX": 0, "translateX": BBOX["west"],
                "shearY": 0, "scaleY": -GRID_DEG, "translateY": BBOX["north"],
            },
            "crsCode": "EPSG:4326",
        },
    }
    arr = ee.data.computePixels(req)          # structured ndarray (H, W)
    stack = np.stack([arr[f"d{d:02d}"] for d in range(1, len(dates) + 1)])
    stack = stack.astype(np.float32)
    stack[stack <= FILL + 1] = np.nan
    stack *= 0.001                            # scale factor MAIAC
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"aod_{year}-{month:02d}.npz"
    np.savez_compressed(out, aod=stack, dates=np.array(dates))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01")
    p.add_argument("--end", default="2024-12")
    a = p.parse_args()
    y0, m0 = map(int, a.start.split("-"))
    y1, m1 = map(int, a.end.split("-"))
    init()
    cur, stop = date(y0, m0, 1), date(y1, m1, 1)
    while cur <= stop:
        out = DATA_DIR / f"aod_{cur.year}-{cur.month:02d}.npz"
        if out.exists():
            print(f"skip {out.name} (уже есть)", flush=True)
        else:
            path = fetch_month(cur.year, cur.month)
            with np.load(path) as z:
                cov = 100 * np.mean(~np.isnan(z["aod"]))
            print(f"ok  {path.name}  покрытие {cov:.0f}%", flush=True)
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
