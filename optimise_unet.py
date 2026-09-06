"""U-Net optimisation study, run on an inner split of the TRAINING period only.

Why this exists in this form. train_unet_downscaler.py computes its epoch-wise
validation loss on ml_validation_dataset.nc - the withheld 2011-2024 record -
and uses it to early-stop, to schedule the learning rate, and to decide which
epoch's weights are saved. That is model selection on the test set, the same
defect Section 6.12 found and fixed for XGBoost, still live in the U-Net and in
the CNN. Every number this study reports for those two models is therefore
optimistically biased by an unknown amount.

So no configuration here is selected on 2011-2024. The 312 training months are
split 240 / 72 - fit on 1985-2004, select on 2005-2010 - and the withheld record
is touched exactly once per configuration, at the end, to report.

Variants test four hypotheses about why the U-Net damps fine-scale power by a
factor of twelve and has the worst spatial correlation of the four models:

  GP    the composite loss adds mean|grad(y_pred)|, an explicit smoothness
        prior. Minimising a prediction's own gradients flattens it. Matching
        the TARGET's gradients instead is the intended behaviour.
  CLIM  the anomaly is taken against a SCALAR mean of the whole training
        record, so the network must learn the entire spatial pattern from 240
        samples. A per-cell, per-month climatology hands it that pattern.
  DROP  Dropout2d(0.3) fires eight times per encoder level. Spatial dropout
        zeroes whole channels and is a strong smoother.
  CAP   31M parameters on 240 samples.

    KMP_DUPLICATE_LIB_OK=TRUE python optimise_unet.py --variants baseline gp_fix
    KMP_DUPLICATE_LIB_OK=TRUE python optimise_unet.py --all
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data/processed/evaluation/unet_optimisation.csv")
TRAIN = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
TOPO = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CLEAR = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")

COARSE = ["clt", "tas", "ps", "huss", "od550aer"]
INNER_SPLIT_YEAR = 2005
SEED = 42
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


# --------------------------------------------------------------------- data
def load_all():
    tr, va = xr.open_dataset(TRAIN), xr.open_dataset(VAL)
    topo = xr.open_dataset(TOPO)
    fine_lat, fine_lon = tr.fine_lat.values, tr.fine_lon.values
    topo_i = topo.interp(lat=fine_lat, lon=fine_lon, method="linear")
    static = np.stack([np.nan_to_num(topo_i[v].values) for v in ("elevation", "slope", "svf")])
    LON, LAT = np.meshgrid(fine_lon, fine_lat)
    static = np.concatenate([static, LAT[None], LON[None]], axis=0)
    static = (static - static.mean(axis=(1, 2), keepdims=True)) / \
             (static.std(axis=(1, 2), keepdims=True) + 1e-9)

    def pack(ds):
        t = pd.DatetimeIndex(ds.time.values)
        xc = np.stack([np.nan_to_num(ds[v].values) for v in COARSE], axis=1)
        y = np.nan_to_num(ds["clear_sky_index"].values)
        doy = t.dayofyear.values
        astro = np.stack([np.sin(2 * np.pi * doy / 365.25),
                          np.cos(2 * np.pi * doy / 365.25)], axis=1)
        return t, xc.astype(np.float32), y.astype(np.float32), astro.astype(np.float32)

    return pack(tr), pack(va), static.astype(np.float32), fine_lat, fine_lon


class DS(Dataset):
    def __init__(self, xc, y, astro, static, clim, std, augment):
        self.xc, self.y, self.astro = xc, y, astro
        self.static, self.clim, self.std, self.augment = static, clim, std, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        xc = self.xc[i]
        h, w = self.y.shape[1:]
        af = np.broadcast_to(self.astro[i][:, None, None], (2, h, w))
        xf = np.concatenate([self.static, af], axis=0).astype(np.float32)
        tgt = ((self.y[i] - self.clim) / self.std)[None].astype(np.float32)
        if self.augment:
            if np.random.rand() < .5:
                xc, xf, tgt = [np.flip(a, axis=-2).copy() for a in (xc, xf, tgt)]
            if np.random.rand() < .5:
                xc, xf, tgt = [np.flip(a, axis=-1).copy() for a in (xc, xf, tgt)]
        return torch.from_numpy(xc), torch.from_numpy(xf), torch.from_numpy(tgt)


# ------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, i, o, p):
        super().__init__()
        layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)]
        if p > 0:
            layers.append(nn.Dropout2d(p))
        layers += [nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)]
        if p > 0:
            layers.append(nn.Dropout2d(p))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, cc, fc, shape, width=64, depth=4, p=0.3):
        super().__init__()
        self.shape = shape
        m = 2 ** depth
        self.pad = (int(np.ceil(shape[0] / m) * m), int(np.ceil(shape[1] / m) * m))
        ch = [width * 2 ** k for k in range(depth + 1)]
        self.inc = Block(cc + fc, ch[0], p)
        # The deployed model reaches its deepest width through a 1x1 conv, not a
        # pair of 3x3s. Using DoubleConv here instead cost 7.25 GiB and put MPS
        # out of memory on the first variant.
        self.downs = nn.ModuleList(
            [Block(ch[k], ch[k + 1], p) for k in range(depth - 1)]
            + [nn.Sequential(nn.Conv2d(ch[depth - 1], ch[depth], 1),
                             nn.BatchNorm2d(ch[depth]), nn.ReLU(True))])
        self.ups = nn.ModuleList([Block(ch[k + 1] + ch[k], ch[k], p) for k in reversed(range(depth))])
        self.outc = nn.Conv2d(ch[0], 1, 1)

    def forward(self, xc, xf):
        x = torch.cat([F.interpolate(xc, size=self.shape, mode="bilinear", align_corners=True), xf], 1)
        x = F.pad(x, [0, self.pad[1] - self.shape[1], 0, self.pad[0] - self.shape[0]])
        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(F.max_pool2d(feats[-1], 2)))
        x = feats[-1]
        for k, u in enumerate(self.ups):
            skip = feats[-2 - k]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = u(torch.cat([skip, x], 1))
        return self.outc(x)[:, :, :self.shape[0], :self.shape[1]]


# ------------------------------------------------------------------ losses
def grad_penalty_own(pred, tgt):
    """The deployed loss: penalises the PREDICTION's own gradients. This is a
    smoothness prior and it is the wrong sign for a model that already damps."""
    dx = (pred[..., :, :-1] - pred[..., :, 1:]).abs().mean()
    dy = (pred[..., :-1, :] - pred[..., 1:, :]).abs().mean()
    return dx + dy


def grad_match(pred, tgt):
    """Match the target's gradients instead of suppressing the prediction's."""
    px = pred[..., :, :-1] - pred[..., :, 1:]
    tx = tgt[..., :, :-1] - tgt[..., :, 1:]
    py = pred[..., :-1, :] - pred[..., 1:, :]
    ty = tgt[..., :-1, :] - tgt[..., 1:, :]
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def spectral_loss(pred, tgt):
    """Match the log power spectrum, which targets the damping directly."""
    P = torch.fft.rfft2(pred.squeeze(1)).abs() ** 2
    T = torch.fft.rfft2(tgt.squeeze(1)).abs() ** 2
    return (torch.log1p(P) - torch.log1p(T)).abs().mean()


def make_loss(kind, lam):
    def f(pred, tgt):
        mse = F.mse_loss(pred, tgt)
        if kind == "none":
            return mse
        if kind == "own":
            return mse + lam * grad_penalty_own(pred, tgt)
        if kind == "match":
            return mse + lam * grad_match(pred, tgt)
        if kind == "spectral":
            return mse + lam * spectral_loss(pred, tgt)
        raise ValueError(kind)
    return f


# ------------------------------------------------------------------ metrics
def spectral_ratio(pred, truth, cut=10):
    """Retained share of fine-scale power, prediction over truth, on the
    time-mean field - the same quantity compute_power_spectra.py reports."""
    def tail(a):
        f = np.fft.fftshift(np.abs(np.fft.fft2(a - a.mean())) ** 2)
        cy, cx = np.array(f.shape) // 2
        y, x = np.ogrid[:f.shape[0], :f.shape[1]]
        r = np.hypot(y - cy, x - cx).astype(int)
        tot = f.sum()
        return f[r >= cut].sum() / tot if tot > 0 else np.nan
    return tail(pred.mean(0)) / tail(truth.mean(0))


def evaluate(model, xc, xf_static, astro, y_true, clim, std, clear, months):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(y_true)):
            h, w = y_true.shape[1:]
            af = np.broadcast_to(astro[i][:, None, None], (2, h, w))
            xf = np.concatenate([xf_static, af], axis=0)[None].astype(np.float32)
            p = model(torch.from_numpy(xc[i][None]).to(DEV), torch.from_numpy(xf).to(DEV))
            preds.append(p.cpu().numpy()[0, 0])
    csi = np.clip(np.array(preds) * std + clim, 0, 1.1)
    ghi_p = csi * clear[months - 1]
    ghi_t = np.clip(y_true, 0, 1.1) * clear[months - 1]
    err = ghi_p - ghi_t
    tm_p, tm_t = ghi_p.mean(0), ghi_t.mean(0)
    return dict(
        rmse=float(np.sqrt((err ** 2).mean())),
        mbe=float(err.mean()),
        r=float(np.corrcoef(ghi_p.ravel(), ghi_t.ravel())[0, 1]),
        spatial_r=float(np.corrcoef(tm_p.ravel(), tm_t.ravel())[0, 1]),
        centred_rmse=float((tm_p - tm_t).std()),
        spec_ratio=float(spectral_ratio(csi, np.clip(y_true, 0, 1.1))),
    )


# ---------------------------------------------------------------- variants
BASE = dict(loss="own", lam=0.001, clim="scalar", dropout=0.3, width=64,
            depth=4, lr=2e-4, wd=1e-4, epochs=100, patience=20, batch=8)

VARIANTS = {
    # the deployed configuration, but selected on the inner split
    "baseline":        {},
    # H1: the smoothness prior
    "gp_none":         dict(loss="none"),
    "gp_match":        dict(loss="match"),
    "gp_match_strong": dict(loss="match", lam=0.05),
    "spectral":        dict(loss="spectral", lam=0.01),
    # H2: the scalar climatology
    "clim_percell":    dict(clim="percell"),
    # H3: dropout
    "drop_0":          dict(dropout=0.0),
    "drop_10":         dict(dropout=0.1),
    # H4: capacity
    "small":           dict(width=32, depth=3),
    "tiny":            dict(width=16, depth=3),
    # combinations of whatever the single factors show
    "combo_a":         dict(loss="match", clim="percell", dropout=0.1),
    "combo_b":         dict(loss="match", clim="percell", dropout=0.1, width=32, depth=3),
    "combo_c":         dict(loss="none", clim="percell", dropout=0.1, width=32, depth=3),
    "combo_spec":      dict(loss="spectral", lam=0.01, clim="percell", dropout=0.1,
                            width=32, depth=3),
}


def run(name, cfg_over, data, report_test=True, seed=SEED):
    cfg = dict(BASE); cfg.update(cfg_over)
    set_seed(seed)
    (t_tr, xc_tr, y_tr, a_tr), (t_te, xc_te, y_te, a_te), static, flat, flon = data
    clear = xr.open_dataset(CLEAR).clearsky_ghi.values

    inner = t_tr.year < INNER_SPLIT_YEAR
    xc_f, y_f, a_f = xc_tr[inner], y_tr[inner], a_tr[inner]
    xc_s, y_s, a_s = xc_tr[~inner], y_tr[~inner], a_tr[~inner]

    if cfg["clim"] == "scalar":
        clim = np.float32(y_f.mean()); std = np.float32(y_f.std())
    else:                                   # per-cell, per-calendar-month
        mm = t_tr[inner].month.values
        clim = np.stack([y_f[mm == m].mean(0) for m in range(1, 13)]).astype(np.float32)
        std = np.float32((y_f - clim[mm - 1]).std())

    def climfor(times):
        return clim if cfg["clim"] == "scalar" else clim[times.month.values - 1]

    # Standardise the coarse atmospheric channels on the FIT FOLD's statistics.
    # Omitting this was a real bug in the first run of this sweep: ps enters at
    # ~90,000 Pa and tas at ~290 K against huss at ~0.01, straight into a Conv2d
    # that precedes any BatchNorm, so the pressure channel swamped the rest and
    # nothing learned. Train loss sat at 1.0 - the variance of a unit-normalised
    # target - for every variant, which is what made ten configurations produce
    # the same meaningless RMSE.
    cm = xc_f.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    cs = xc_f.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    cs[cs == 0] = 1.0
    xc_f = ((xc_f - cm) / cs).astype(np.float32)
    xc_s = ((xc_s - cm) / cs).astype(np.float32)
    xc_te_n = ((xc_te - cm) / cs).astype(np.float32)

    tr_ds = DS(xc_f, y_f, a_f, static, climfor(t_tr[inner]), std, True)
    dl = DataLoader(tr_ds, batch_size=cfg["batch"], shuffle=True, drop_last=False)

    model = UNet(len(COARSE), static.shape[0] + 2, y_tr.shape[1:], cfg["width"],
                 cfg["depth"], cfg["dropout"]).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    lossf = make_loss(cfg["loss"], cfg["lam"])

    best, best_state, wait, t0 = np.inf, None, 0, time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        for xcb, xfb, yb in dl:
            opt.zero_grad()
            l = lossf(model(xcb.to(DEV), xfb.to(DEV)), yb.to(DEV))
            l.backward(); opt.step()
        # selection on the INNER split, never on 2011-2024
        if DEV.type == "mps":
            torch.mps.empty_cache()
        sel = evaluate(model, xc_s, static, a_s, y_s, climfor(t_tr[~inner]), std,
                       clear, t_tr[~inner].month.values)
        if DEV.type == "mps":
            torch.mps.empty_cache()
        sched.step(sel["rmse"])
        if sel["rmse"] < best:
            best, wait = sel["rmse"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    row = dict(variant=name, params=nparam, epochs_run=ep + 1,
               minutes=round((time.time() - t0) / 60, 1),
               inner_rmse=round(best, 4), **{("cfg_" + k): v for k, v in cfg.items()})
    if report_test:
        te = evaluate(model, xc_te_n, static, a_te, y_te, climfor(t_te), std,
                      clear, t_te.month.values)
        row.update({("test_" + k): round(v, 4) for k, v in te.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    names = list(VARIANTS) if args.all else args.variants
    if args.epochs:
        BASE["epochs"] = args.epochs

    print("device %s | inner split: fit <%d, select >=%d | test 2011-2024 touched once"
          % (DEV, INNER_SPLIT_YEAR, INNER_SPLIT_YEAR), flush=True)
    data = load_all()
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT).variant)
        if done:
            print("  resuming; already done: %s" % ", ".join(sorted(done)), flush=True)
    rows = []
    for n in names:
        if n in done:
            continue
        try:
            r = run(n, VARIANTS[n], data)
        except RuntimeError as e:
            print("  %-16s FAILED: %s" % (n, str(e)[:110]), flush=True)
            if DEV.type == "mps":
                torch.mps.empty_cache()
            continue
        rows.append(r)
        print("  %-16s inner %.3f | test RMSE %.3f  spatial_r %.4f  spec %.3f  "
              "(%.1f min, %d ep, %.1fM par)"
              % (n, r["inner_rmse"], r["test_rmse"], r["test_spatial_r"],
                 r["test_spec_ratio"], r["minutes"], r["epochs_run"],
                 r["params"] / 1e6), flush=True)
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        if os.path.exists(OUT):
            old = pd.read_csv(OUT)
            df = pd.concat([old[~old.variant.isin(df.variant)], df], ignore_index=True)
        df.to_csv(OUT, index=False)
    print("\nSaved %s" % OUT)


if __name__ == "__main__":
    main()
