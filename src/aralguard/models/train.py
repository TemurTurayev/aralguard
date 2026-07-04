"""Обучение DustNowcaster. Пока датасет не собран — режим --synthetic
(движущиеся гауссовы «облака») как дымовой тест всего цикла обучения.

python -m aralguard.models.train --synthetic --epochs 3
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from aralguard.models.convlstm import DustNowcaster


class SyntheticPlumes(Dataset):
    """Гауссово пятно дрейфует по ветру — суррогат пылевого облака."""

    def __init__(self, n=200, t_in=8, t_out=8, h=32, w=48):
        self.n, self.t_in, self.t_out, self.h, self.w = n, t_in, t_out, h, w

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        cx, cy = rng.uniform(5, 15), rng.uniform(8, 24)
        vx, vy = rng.uniform(0.6, 1.6), rng.uniform(-0.4, 0.4)
        amp, sig = rng.uniform(0.5, 1.5), rng.uniform(2.5, 5.0)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        frames = []
        for t in range(self.t_in + self.t_out):
            g = amp * np.exp(-(((xx - cx - vx * t) ** 2 +
                                (yy - cy - vy * t) ** 2) / (2 * sig ** 2)))
            u = np.full_like(g, vx); v = np.full_like(g, vy)
            noise = rng.normal(0, 0.02, g.shape)
            frames.append(np.stack([g, u, v, g * 0.5, noise]))  # 5 каналов
        arr = np.asarray(frames, dtype=np.float32)
        x = arr[: self.t_in]                       # (T_in, C, H, W)
        y = arr[self.t_in:, :1]                    # (T_out, 1, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=8)
    a = p.parse_args()

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    ds = SyntheticPlumes() if a.synthetic else None
    if ds is None:
        raise SystemExit("Реальный датасет ещё не собран — используйте --synthetic")
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True)

    model = DustNowcaster(in_ch=5, hid=32, t_out=8).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    lossf = torch.nn.SmoothL1Loss()

    for ep in range(a.epochs):
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        print(f"epoch {ep + 1}: loss={tot / len(ds):.4f}")
    torch.save(model.state_dict(), "runs_last.pt")
    print("saved runs_last.pt — дымовой тест пройден")


if __name__ == "__main__":
    main()
