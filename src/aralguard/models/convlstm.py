"""ConvLSTM для nowcasting полей пыли/осадков (PyTorch).

Вход:  (B, T_in, C, H, W)  — последовательность карт (AOD, PM, ветер u/v, BLH...)
Выход: (B, T_out, 1, H, W) — прогнозные карты PM10/AOD на будущие шаги.
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, b, hw, device):
        h = torch.zeros(b, self.hid_ch, *hw, device=device)
        return h, h.clone()


class DustNowcaster(nn.Module):
    """Encoder-forecaster: 2 слоя ConvLSTM, авторегрессивный прогноз T_out шагов."""

    def __init__(self, in_ch: int = 5, hid: int = 48, t_out: int = 24):
        super().__init__()
        self.t_out = t_out
        self.enc1 = ConvLSTMCell(in_ch, hid)
        self.enc2 = ConvLSTMCell(hid, hid)
        self.head = nn.Conv2d(hid, 1, 1)
        self.inp_proj = nn.Conv2d(1, in_ch, 1)  # прогноз -> вход следующего шага

    def forward(self, x):                      # x: (B,T,C,H,W)
        b, t, c, hgt, wid = x.shape
        dev = x.device
        s1 = self.enc1.init_state(b, (hgt, wid), dev)
        s2 = self.enc2.init_state(b, (hgt, wid), dev)
        for i in range(t):                     # encode
            s1 = self.enc1(x[:, i], s1)
            s2 = self.enc2(s1[0], s2)
        outs, last = [], x[:, -1]
        for _ in range(self.t_out):            # forecast (autoregressive)
            s1 = self.enc1(last, s1)
            s2 = self.enc2(s1[0], s2)
            y = self.head(s2[0])               # (B,1,H,W)
            outs.append(y)
            last = self.inp_proj(y)
        return torch.stack(outs, dim=1)        # (B,T_out,1,H,W)


if __name__ == "__main__":  # дымовой тест: python -m aralguard.models.convlstm
    m = DustNowcaster(in_ch=5, hid=16, t_out=8)
    y = m(torch.randn(2, 12, 5, 32, 48))
    print("ok, out:", tuple(y.shape))          # (2, 8, 1, 32, 48)
