"""Pooled 50-stock windowed dataset for the DeformTime/TSLib harness.

WHY THIS EXISTS
---------------
Neither Crossformer's nor DeformTime's stock loaders fit our setup, for two independent reasons:

  1. TSLib's Dataset_Custom reads ONE csv as ONE long series and imposes its own 70/10/20 ROW split
     (data_loader.py: `num_train = int(len(df_raw) * 0.7)`). That silently discards our walk-forward
     calendar splits and would train on 2022-2023 data we intend to hold out.

  2. We train POOLED: 50 stocks, one shared model, no stock identifier. Vertically concatenating the
     50 stocks into one csv and letting TSLib slide over it would emit windows SPANNING STOCK
     BOUNDARIES -- a window containing the tail of AKAM and the head of ATO. Those samples are
     meaningless and there would be 49 of them per epoch, silently.

So windows are built per stock and only then pooled.

THE CONTRACT (matches the protocol EXAMM is scored under)
---------------------------------------------------------
  input  t[i .. i+L-1]  ->  target RET at t[i+L]
  L = seq_len = 20, pred_len = 1, label_len = 0

TSLib's own windowing already produces exactly this, so the target is just the raw RET column with no
manual shifting: with label_len=0 and pred_len=1, seq_y = data[s_end : s_end+1], i.e. the row
immediately after the input window. Do NOT pre-shift the csv -- that would double-shift.

Sample counts (cohort 2020). Only TRAIN follows the bare len - seq_len - pred_len + 1, because
nothing precedes it to borrow lookback from; val and test each prepend the tail of the split before
them (see __read_data__) and so yield rows - pred_len windows, independent of seq_len:
    train 3,290 rows                -> 3,270/stock -> 163,500 pooled
    val     252 rows + 19 from train ->   251/stock ->  12,550 pooled

TEST IS DIFFERENT: __read_data__ prepends the last (seq_len - pred_len) rows of val to test before
windowing, so the first test window's lookback doesn't have to be burned entirely inside the test
file. This is leak-free (val is already fully in the past relative to test; it is used only as
INPUT context, never as a target) and it is why test has 500/stock, not 481: targets now cover
t[2]..t[501], the SAME range EXAMM's own predictions cover (EXAMM: 500 predictions from 501 rows,
time_offset=1). No truncation/alignment step is needed when comparing to EXAMM any more -- both
emit one prediction per test row after the first, covering the identical (ticker, date) set.
(Before this fix, test windows never crossed the val/test boundary, so the first prediction needed
20 full days already inside the test file: only 481/stock, targets t[21]..t[501], and EXAMM's own
predictions had to be truncated by 19 to align. That older data lives in the "_trading_shadow"
matched-window comparisons and any results built before this docstring changed.)

TARGET COLUMN ORDER
-------------------
TSLib's `features='MS'` convention takes the LAST dataframe column as the prediction target
(f_dim = -1 in the exp loop). RET is both an input and the target, so it is moved to the end:
    [VOL_CHANGE, BA_SPREAD, ILLIQUIDITY, sprtrn, TURNOVER, RET]
enc_in / dec_in = 6, c_out = 1.

NORMALIZATION -- READ THIS BEFORE CHANGING IT
---------------------------------------------
EXAMM's `--normalize avg_std_dev` is NOT a z-score, despite the name. In time_series.cxx:1064 the
combined *variance* is assigned to a variable called `std_dev` with no sqrt (contrast
calculate_statistics, which does take the sqrt). Lines 1072-1075 then set
norm_max = (max - avg) / std_dev, and the per-value transform at :162 is
((x - avg) / std_dev) / norm_max. The variance cancels identically:

    ((x - avg) / V) / ((max - avg) / V)  ==  (x - avg) / (max - avg)

so the effective transform is MEAN-CENTERED MAX-SCALING, bounded, with the maximum observation
mapping to exactly +1.0. (normalize_avg_std_dev even logs "doing min/max normalization" at :1010.)

This matters, and z-scoring instead would NOT be a harmless difference. Measured on cohort_2020
train union val, RET lands in [-0.686, +1.000] under EXAMM versus [-15.8, +23.0] under a z-score;
BA_SPREAD reaches +115.6 z-scaled. Under an MSE objective the worst per-sample target term would be
~530x larger for the transformer than for EXAMM -- a first-order difference in gradient dynamics,
not a shared imperfection. Rank IC is invariant to a single pooled affine map, but MSE and RMSE are
not, and the paper reports both.

So ExammScaler below reproduces (x - avg) / (max - avg) exactly. Statistics are pooled across all 50
stocks (one scaler, not 50), matching a single pooled model. Scope defaults to `train_val` because
EXAMM pools training and validation filenames before computing them (time_series.cxx:764-771); the
test split never contributes to its own scaling under either scope. Set STOCK_NORM_SCOPE=train for
train-only statistics -- but then EXAMM must be re-run to match, or the comparison is confounded.

EVALUATION KEYS
---------------
Cross-sectional rank IC needs to know which (ticker, date) each prediction belongs to, and the
harness's test loop only emits a flat array. So on construction this writes `<split>_index.csv`
(sample_idx, ticker, target_date) to STOCK_META_DIR. Row i of that file describes prediction i, in
dataloader order. Two properties of data_factory.py make that alignment hold for the test split and
must not be changed: shuffle_flag=False, and batch_size=1 (so its drop_last=True drops nothing,
since any count % 1 == 0). The val split IS shuffled, so val_index.csv is written for completeness but
its row order is not meaningful -- use it only via a (ticker, date) join, never positionally.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 6 predictors, RET last (see TARGET COLUMN ORDER above)
FEATURES = ["VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "sprtrn", "TURNOVER", "RET"]
TARGET = "RET"
SPLIT_FILE = {0: "train", 1: "val", 2: "test"}


class ExammScaler:
    """EXAMM's `avg_std_dev` transform, exactly: (x - avg) / (max - avg). See module docstring for
    the derivation showing why the variance in time_series.cxx cancels. Column-wise, pooled."""

    def fit(self, x):
        self.avg_ = x.mean(axis=0).astype(np.float64)
        self.max_ = x.max(axis=0).astype(np.float64)
        self.denom_ = self.max_ - self.avg_
        if not np.all(self.denom_ > 0):
            bad = [FEATURES[j] for j in np.where(self.denom_ <= 0)[0]]
            raise ValueError(f"max == avg for {bad}; column is constant, cannot normalize")
        return self

    def transform(self, x):
        return (x - self.avg_) / self.denom_

    def inverse_transform_target(self, y):
        """Un-scale predictions of the target column back to raw RET units."""
        j = FEATURES.index(TARGET)
        return y * self.denom_[j] + self.avg_[j]


class Dataset_StockPooled(Dataset):
    def __init__(self, root_path, data_path=None, flag="train", size=None, features="MS",
                 target="RET", scale=True, timeenc=0, freq="d", seasonal_patterns=None):
        assert size is not None, "size=[seq_len, label_len, pred_len] is required"
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ("train", "val", "test")
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.flag = flag
        self.root_path = root_path
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.norm_scope = os.environ.get("STOCK_NORM_SCOPE", "train_val")
        assert self.norm_scope in ("train", "train_val")
        self.__read_data__()

    # ------------------------------------------------------------------ loading

    def _load_split(self, split):
        """{ticker: (values[n,6] float32, dates[n] datetime)} for one split, tickers sorted."""
        # `not f.startswith(".")` is load-bearing, not hygiene. os.listdir returns dotfiles (a glob
        # would not), and a tarball built on macOS carries an AppleDouble "._NAME" sidecar beside
        # every file. "._ADM_train.csv" ends with "_train.csv", so without this filter a panel
        # unpacked from such an archive silently doubles: 100 "tickers" for 50 stocks, half of them
        # 163-byte binary metadata blobs parsed as price data. Observed on this project when the
        # portfolio bundle was built on a Mac and unpacked on the pod.
        paths = sorted(f for f in os.listdir(self.root_path)
                       if f.endswith(f"_{split}.csv") and not f.startswith("."))
        if not paths:
            raise FileNotFoundError(f"no *_{split}.csv under {self.root_path}")
        out = {}
        for p in paths:
            ticker = p[: -len(f"_{split}.csv")]
            df = pd.read_csv(os.path.join(self.root_path, p))
            missing = [c for c in FEATURES + ["date"] if c not in df.columns]
            if missing:
                raise ValueError(f"{p}: missing columns {missing}")
            out[ticker] = (df[FEATURES].values.astype(np.float32),
                           pd.to_datetime(df["date"]).values)
        return out

    def __read_data__(self):
        data = self._load_split(SPLIT_FILE[self.set_type])

        # ---- scaler: fit on the configured scope, pooled over all stocks, never on test
        self.scaler = ExammScaler()
        if self.scale:
            fit_rows = [v for _, (v, _) in sorted(self._load_split("train").items())]
            if self.norm_scope == "train_val":
                fit_rows += [v for _, (v, _) in sorted(self._load_split("val").items())]
            self.scaler.fit(np.concatenate(fit_rows, axis=0))

        # ---- VAL AND TEST: prepend the tail of the PRECEDING split as lookback context, so the
        # first target of a split does not require seq_len-1 of that split's own days to be burned
        # building the first window. Leak-free: the preceding split lies entirely in the past
        # (verified across all 700 stocks / 14 datasets -- train ends 2020-12-31, val starts
        # 2021-01-04), and it is used only as INPUT, never as a target. That last part is
        # structural, not incidental: borrowed rows occupy combined indices [0, ctx) with
        # ctx = seq_len - pred_len, while the earliest target sits at index seq_len, so for any
        # seq_len the two ranges cannot overlap.
        #
        # WHY ctx IS seq_len - pred_len AND NOT seq_len. With this ctx, window s=0's target lands
        # on the split's SECOND row -- exactly the first row EXAMM can score, since time_offset=1
        # predicts row t+1 from row t and so can never score a split's first row either.
        # Crossformer's upstream loader borrows the full in_len (border1s = train_num - in_len),
        # which lets it predict the split's first row; that yields 252 val / 501 test windows
        # against EXAMM's 251 / 500 and breaks the row-for-row comparison the paper depends on.
        # Borrowing one fewer row is deliberate.
        #
        # VALIDATION WAS PREVIOUSLY EXCLUDED FROM THIS, and it was a real defect. Validation burned
        # its first seq_len rows, so at L=20 it began 2021-02-02 rather than 2021-01-05 (232
        # windows/stock vs EXAMM's 251), and at L=96 it would not have begun until 2021-05-21 (156
        # windows). That under-fed early stopping, hid every January from model selection while
        # January is fully present at test time, and made validation size depend on seq_len -- so a
        # longer-context probe would have silently competed on a third less selection data.
        if self.flag in ("val", "test"):
            prev_split = "train" if self.flag == "val" else "val"
            prev_data = self._load_split(prev_split)
            ctx = self.seq_len - self.pred_len
            for ticker in list(data):
                if ticker not in prev_data:
                    raise ValueError(f"{ticker}: in {self.flag} but not {prev_split} -- cohorts not aligned")
                pvals, pdates = prev_data[ticker]
                if len(pvals) < ctx:
                    raise ValueError(f"{ticker}/{prev_split}: {len(pvals)} rows < {ctx} needed "
                                     f"for {self.flag} lookback")
                cvals, cdates = data[ticker]
                data[ticker] = (np.concatenate([pvals[-ctx:], cvals], axis=0),
                                np.concatenate([pdates[-ctx:], cdates], axis=0))

        # ---- per-stock windows; a window never spans two stocks
        self.series, self.stamps, self.tickers = [], [], []
        self.index = []          # (stock_idx, window_start)
        self.meta = []           # (ticker, target_date) parallel to self.index
        n_win = self.seq_len + self.pred_len          # rows consumed by one sample
        for si, ticker in enumerate(sorted(data)):
            values, dates = data[ticker]
            if len(values) < n_win:
                raise ValueError(f"{ticker}/{self.flag}: {len(values)} rows < {n_win} needed")
            self.series.append(self.scaler.transform(values).astype(np.float32)
                               if self.scale else values)
            self.stamps.append(self._time_features(dates))
            self.tickers.append(ticker)
            for s in range(len(values) - n_win + 1):
                self.index.append((si, s))
                # label_len=0 => target row is exactly s + seq_len
                self.meta.append((ticker, str(pd.Timestamp(dates[s + self.seq_len]).date())))

        self._write_meta()

    def _time_features(self, dates):
        """Four daily calendar features scaled to [-0.5, 0.5]. Note data_factory.py sets timeenc=1
        (because --embed defaults to timeF); we ignore that and always emit these four, which is
        self-consistent across splits and models. Channel-mixing models that concatenate x_mark to
        the variate axis will therefore see 6 + 4 tokens -- record that in the results table."""
        d = pd.DatetimeIndex(dates)
        return np.stack([d.month / 12.0 - 0.5,
                         d.day / 31.0 - 0.5,
                         d.dayofweek / 6.0 - 0.5,
                         d.dayofyear / 366.0 - 0.5], axis=1).astype(np.float32)

    def _write_meta(self):
        """Row i describes prediction i in dataloader order. Requires shuffle=False for eval.
        Also persists the scaler so predictions can be returned to raw RET units afterwards --
        without this, saved preds are in normalized space and MSE is not comparable to EXAMM."""
        out_dir = os.environ.get("STOCK_META_DIR", self.root_path)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(self.meta, columns=["ticker", "target_date"]).to_csv(
            os.path.join(out_dir, f"{self.flag}_index.csv"), index_label="sample_idx")
        if self.scale:
            j = FEATURES.index(TARGET)
            with open(os.path.join(out_dir, "scaler.json"), "w") as f:
                json.dump({"scope": self.norm_scope, "features": FEATURES, "target": TARGET,
                           "avg": self.scaler.avg_.tolist(), "max": self.scaler.max_.tolist(),
                           "target_avg": float(self.scaler.avg_[j]),
                           "target_denom": float(self.scaler.denom_[j])}, f, indent=2)

    # ------------------------------------------------------------------ access

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, s = self.index[i]
        x, mark = self.series[si], self.stamps[si]
        s_end = s + self.seq_len
        r_beg = s_end - self.label_len
        r_end = r_beg + self.label_len + self.pred_len
        return (torch.from_numpy(x[s:s_end]), torch.from_numpy(x[r_beg:r_end]),
                torch.from_numpy(mark[s:s_end]), torch.from_numpy(mark[r_beg:r_end]))

    def inverse_transform(self, data):
        """Only ever called on single-column target arrays here (--inverse is not passed by our
        job, but guard anyway rather than crash on a 6-feature scaler)."""
        return self.scaler.inverse_transform_target(np.asarray(data))
