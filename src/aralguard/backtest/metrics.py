"""Метрики раннего предупреждения: POD, FAR, CSI + skill против бейзлайнов.

Событие = превышение порога PM10 в ячейке/районе. Сравниваем модель с:
- персистентностью (завтра = сегодня),
- сырым CAMS (наша модель обязана его бить, иначе AI-слой не нужен).
"""
import numpy as np


def contingency(pred: np.ndarray, obs: np.ndarray, thr: float):
    p, o = pred >= thr, obs >= thr
    hits = int(np.sum(p & o))
    misses = int(np.sum(~p & o))
    false_alarms = int(np.sum(p & ~o))
    return hits, misses, false_alarms


def pod(pred, obs, thr):     # вероятность обнаружения (recall)
    h, m, _ = contingency(pred, obs, thr)
    return h / (h + m) if h + m else np.nan


def far(pred, obs, thr):     # доля ложных тревог — целевая метрика alert fatigue
    h, _, f = contingency(pred, obs, thr)
    return f / (h + f) if h + f else np.nan


def csi(pred, obs, thr):     # critical success index
    h, m, f = contingency(pred, obs, thr)
    return h / (h + m + f) if h + m + f else np.nan


def rmse(pred, obs):
    return float(np.sqrt(np.nanmean((pred - obs) ** 2)))


def skill_vs(pred, baseline, obs) -> float:
    """>0 — модель лучше бейзлайна (по RMSE), 1 = идеал."""
    return 1.0 - rmse(pred, obs) / max(rmse(baseline, obs), 1e-9)


def report(pred, obs, cams, thr):
    persist = obs[:-1] if len(obs) > 1 else obs   # сдвиг на 1 шаг
    lines = {
        "POD": pod(pred, obs, thr),
        "FAR": far(pred, obs, thr),
        "CSI": csi(pred, obs, thr),
        "RMSE": rmse(pred, obs),
        "skill_vs_CAMS": skill_vs(pred, cams, obs),
        "skill_vs_persistence": skill_vs(pred[1:], persist, obs[1:]),
    }
    for k, v in lines.items():
        print(f"{k:>22}: {v:.3f}")
    return lines
