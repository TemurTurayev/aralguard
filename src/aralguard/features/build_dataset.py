"""Сборка обучающего датасета: MODIS AOD (+ ERA5, когда скачан) -> единый тензор.

Выход: data_store/dataset/dust_dataset.npz
  X        : float32 [T, C, H, W] — каналы по дням
  dates    : ['YYYY-MM-DD', ...]
  channels : имена каналов

Каналы: aod (NaN->0), aod_mask (1=есть данные) [+ u10, v10, gust, blh, soil_w, precip]
Запуск: python -m aralguard.features.build_dataset
"""
import glob
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data_store"
OUT = DATA / "dataset"

# Порог «пыльного дня» для статистики (региональный средний AOD)
DUSTY_AOD = 0.35


def load_modis():
    files = sorted(glob.glob(str(DATA / "modis" / "aod_*.npz")))
    if not files:
        raise SystemExit("Нет MODIS-файлов — сначала modis_extract")
    aod_list, dates = [], []
    for f in files:
        with np.load(f) as z:
            aod_list.append(z["aod"])
            dates.extend([str(d) for d in z["dates"]])
    aod = np.concatenate(aod_list, axis=0)          # [T,H,W]
    return aod, dates


def load_era5(dates, shape_hw):
    """Дневные агрегаты ERA5, регрид на сетку MODIS. None если файлов нет."""
    import xarray as xr
    files = sorted(glob.glob(str(DATA / "era5" / "era5_*.nc")))
    if not files:
        return None, []
    ds = xr.open_mfdataset(files, combine="by_coords")
    # переименования на случай разных версий CDS
    ren = {"valid_time": "time", "u10": "u10", "v10": "v10",
           "i10fg": "gust", "fg10": "gust", "blh": "blh",
           "swvl1": "soil_w", "tp": "precip"}
    for old, new in list(ren.items()):
        if old in ds and new != old:
            ds = ds.rename({old: new})
    daily = xr.Dataset({
        "u10": ds["u10"].resample(time="1D").mean(),
        "v10": ds["v10"].resample(time="1D").mean(),
        "gust": ds["gust"].resample(time="1D").max(),
        "blh": ds["blh"].resample(time="1D").mean(),
        "soil_w": ds["soil_w"].resample(time="1D").mean(),
        "precip": ds["precip"].resample(time="1D").sum(),
    })
    from aralguard.config import BBOX, GRID_DEG
    h, w = shape_hw
    lats = np.linspace(BBOX["north"] - GRID_DEG / 2, BBOX["south"] + GRID_DEG / 2, h)
    lons = np.linspace(BBOX["west"] + GRID_DEG / 2, BBOX["east"] - GRID_DEG / 2, w)
    daily = daily.interp(latitude=lats, longitude=lons, method="linear")
    chans, names = [], []
    tindex = daily.indexes["time"].strftime("%Y-%m-%d")
    pos = {d: i for i, d in enumerate(tindex)}
    for name in ["u10", "v10", "gust", "blh", "soil_w", "precip"]:
        arr = daily[name].values.astype(np.float32)   # [Te,H,W]
        full = np.zeros((len(dates), h, w), np.float32)
        for i, d in enumerate(dates):
            j = pos.get(d)
            if j is not None:
                full[i] = arr[j]
        chans.append(full)
        names.append(name)
    return np.stack(chans, axis=1), names             # [T,C,H,W]


def main():
    aod, dates = load_modis()
    t, h, w = aod.shape
    mask = (~np.isnan(aod)).astype(np.float32)
    aod0 = np.nan_to_num(aod, nan=0.0).astype(np.float32)
    X = np.stack([aod0, mask], axis=1)                # [T,2,H,W]
    channels = ["aod", "aod_mask"]

    era5, era5_names = (None, [])
    try:
        era5, era5_names = load_era5(dates, (h, w))
    except Exception as e:
        print("ERA5 пока не подключён:", str(e)[:120])
    if era5 is not None:
        X = np.concatenate([X, era5], axis=1)
        channels += era5_names

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "dust_dataset.npz"
    np.savez_compressed(out, X=X, dates=np.array(dates), channels=np.array(channels))

    # статистика для отчёта
    reg_mean = np.where(mask.sum((1, 2)) > 0.1 * h * w,
                        np.nansum(aod0 * mask, (1, 2)) / np.maximum(mask.sum((1, 2)), 1),
                        np.nan)
    dusty = np.nansum(reg_mean > DUSTY_AOD)
    print(f"Датасет: {out}")
    print(f"Дней: {t} ({dates[0]} … {dates[-1]}) · сетка {h}x{w} · каналы: {channels}")
    print(f"Покрытие AOD: {100 * mask.mean():.0f}% · «пыльных» дней (AOD>{DUSTY_AOD}): {int(dusty)}")
    top = np.argsort(np.nan_to_num(reg_mean, nan=-1))[-10:][::-1]
    print("Топ-10 самых пыльных дней:", ", ".join(f"{dates[i]}({reg_mean[i]:.2f})" for i in top))


if __name__ == "__main__":
    main()
