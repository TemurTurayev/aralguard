# AralGuard

AI-система раннего предупреждения о пыльных бурях с Аралкума и селях —
прогноз 24–72 ч + адресные алерты для клиник, школ и МЧС.

**Контур:** спутник (MODIS/CAMS/ERA5/Sentinel-5P) → ConvLSTM-nowcasting →
DLNM-слой здоровья → карта + Telegram-алерты. Фаза 2: наземный рой сенсоров ZONT.

## Структура

```
src/aralguard/
  config.py            # регион, районы, пороги
  data/                # загрузчики: CAMS (ADS), ERA5 (CDS), MODIS MAIAC (GEE)
  models/              # ConvLSTM (PyTorch) + бейзлайн-коррекция CAMS
  backtest/            # метрики POD/FAR/CSI, skill vs персистентность и vs CAMS
  bot/                 # Telegram-бот адресных алертов (aiogram v3)
dashboard/index.html   # веб-дашборд (пока DEMO-режим на синтетике)
```

## Быстрый старт

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # и заполнить ключи (см. ниже)
python -m aralguard.data.cams_download --days 5      # тест загрузки CAMS
python -m aralguard.models.train --synthetic         # дымовой тест модели
open dashboard/index.html                            # мокап дашборда
```

## Ключи (все бесплатные)

| Сервис | Зачем | Где взять |
|---|---|---|
| Copernicus ADS | прогнозы пыли CAMS | ads.atmosphere.copernicus.eu → профиль → API key |
| Copernicus CDS | ERA5 (ветер, погранслой) | cds.climate.copernicus.eu → профиль → API key |
| Google Earth Engine | MODIS MAIAC AOD архив | earthengine.google.com (Noncommercial) |
| NASA Earthdata | NRT-продукты MODIS/VIIRS | urs.earthdata.nasa.gov |
| Telegram BotFather | токен бота | @BotFather → /newbot |

Ключи кладём ТОЛЬКО в `.env` (в git не попадает). Данные — в `data_store/` (тоже вне git).

## Роадмап MVP (до 1 августа)

- [ ] Неделя 1: датасет 2019–2024 (CAMS+MODIS+ERA5, сетка ~10 км, Приаралье)
- [ ] Неделя 2: бейзлайн (коррекция CAMS) → ConvLSTM; бэктест на бурях 2018/2021
- [ ] Неделя 3: карта с прогнозом + бот; DLNM-слой (литературные RR ВОЗ)
- [ ] Неделя 4: полировка, видео, заявка

Полный контекст проекта: `../ARALGUARD_DECISION.md`.
