# Results Tables — current state

All values MEASURED from artifacts on disk. PENDING = no transformer trained yet.
Regenerate with (bash, **not** zsh — zsh does not word-split and collapses the config triples):

```
python3 scripts/transformer_bench/compare_to_examm.py --cohort cohort_2020_aligned \
    --examm-root test_output/mse_cohort_2020_aligned \
    --tf-root results/transformer_bench/cohort_2020_aligned --year 2022
bash scripts/stock_run/regen_ls_sweep.sh        # long/short book-size sweep
bash scripts/stock_run/regen_trading_table.sh   # per-run dispersion, 10 runs
```

## Scope: only true walk-forward cells

Train through year N−1, validate on year N, **trade year N+1**. Exactly two cells satisfy it:

| Cohort | Train | Validate | Trade |
|---|---|---|---|
| 2020 | ≤ 2020-12-31 | 2021 | **2022** |
| 2021 | ≤ 2021-12-31 | 2022 | **2023** |

Cohort 2020 traded on 2023 is excluded — a two-year-stale model selected on a non-adjacent
validation year. That exclusion removes the only significant accuracy cell we had (IC +0.0284,
CI [+0.0022, +0.0546]).

---

## Table 1. Main results

Accuracy is the 10-run ensemble on the identical row set (targets t[21]…t[N]; EXAMM truncated by 19
to match an L=20 window). Trading uses the same ensemble's predictions, dollar-neutral long/short,
daily rebalance, **gross of transaction costs**. Returns are percent over the trading year.

| Model | Cohort | Trade year | Params | Ensemble IC | MSE | MAE | L/S 5 | L/S 10 | L/S 15 | L/S 20 | Equal-wt B&H | S&P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXAMM** | 2020 | **2022** | 55 | +0.0077 | 4.600e−4 | 0.01558 | −21.73 | **−15.20** | −17.21 | −6.55 | −13.99 | −19.75 |
| **EXAMM** | 2021 | **2023** | 77 | +0.0162 | 2.876e−4 | 0.01189 | +41.74 | **+33.90** | +24.98 | +15.25 | +7.91 | +25.08 |
| Crossformer | 2020 | 2022 | 512,588 | PENDING | — | — | — | — | — | — | −13.99 | −19.75 |
| Crossformer | 2021 | 2023 | 512,588 | PENDING | — | — | — | — | — | — | +7.91 | +25.08 |
| DeformTime | 2020 | 2022 | 107,243 | PENDING | — | — | — | — | — | — | −13.99 | −19.75 |
| DeformTime | 2021 | 2023 | 107,243 | PENDING | — | — | — | — | — | — | +7.91 | +25.08 |
| PatchTST | 2020 | 2022 | 67,969 | PENDING | — | — | — | — | — | — | −13.99 | −19.75 |
| PatchTST | 2021 | 2023 | 67,969 | PENDING | — | — | — | — | — | — | +7.91 | +25.08 |

L/S 10 bolded as the pre-committed headline book size.

**Net-of-cost figures exist and should be the paper's headline** — see Table 2. Gross is shown here as
requested, but a reviewer will ask for net, and in 2022 the two differ materially: gross −15.20 is
near parity with B&H's −13.99, while net −18.78 is a clear 4.8pp loss.

## Table 2. Transaction-cost sensitivity

| Trade year | Book | Gross | Net | Drag |
|---|---|---|---|---|
| 2022 | L/S 5 | −21.73 | −25.85 | 4.12 |
| 2022 | L/S 10 | −15.20 | −18.78 | 3.58 |
| 2022 | L/S 15 | −17.21 | −20.15 | 2.94 |
| 2022 | L/S 20 | −6.55 | −9.33 | 2.78 |
| 2023 | L/S 5 | +41.74 | +41.10 | 0.64 |
| 2023 | L/S 10 | +33.90 | +33.02 | 0.88 |
| 2023 | L/S 15 | +24.98 | +23.96 | 1.02 |
| 2023 | L/S 20 | +15.25 | +14.41 | 0.84 |

Drag is 2.8–4.1pp in 2022 but only 0.6–1.0pp in 2023 at every book size — the 2022 book turns over
far more, independently consistent with an unstable signal that year.

## Table 3. Accuracy detail

| Trade year | Days | Mean single run | Seed sd | Run range | Ensemble IC | HAC 95% CI | Hit rate |
|---|---|---|---|---|---|---|---|
| 2022 | 231 | +0.0056 | 0.0078 | [−0.0119, +0.0176] | +0.0077 | [−0.0226, +0.0380] | **48.9%** |
| 2023 | 230 | +0.0038 | 0.0096 | [−0.0086, +0.0209] | +0.0162 | [−0.0071, +0.0395] | 53.9% |

## Table 4. Per-run trading dispersion (L/S 10, net)

| Trade year | n | Mean single run | sd | Run range | Positive | Ensemble |
|---|---|---|---|---|---|---|
| 2022 | 10 | −20.07 | 5.60 | [−27.26, −6.75] | **0/10** | −18.78 |
| 2023 | 10 | +9.15 | 13.32 | [−14.77, +25.18] | 7/10 | +33.02 |

## Table 5. Parameter efficiency

Parameters are reported both per model and for the deployed 10-member ensemble, because every
accuracy and trading figure in this document is an ensemble result. Quoting a single-model count next
to an ensemble number understates what produced it by 10x.

| Model | Per model | Ensemble (×10) | × EXAMM |
|---|---|---|---|
| **EXAMM** 2020 cohort | **55** (23–90) | **549** | 1× |
| **EXAMM** 2021 cohort | **77** (40–131) | **768** | 1× |
| PatchTST | 67,969 | 679,690 | ~1,030× |
| DeformTime | 107,243 | 1,072,430 | ~1,625× |
| Crossformer | 512,588 | 5,125,880 | ~7,767× |

Ratios are stable whether computed model-to-model or ensemble-to-ensemble, since both sides are
ensembled over 10 runs; they are quoted against EXAMM's cross-cohort mean of 66.

### RETRACTED: "independent windows per parameter"

An earlier version of this table carried a column dividing the training-window count by the sequence
length, on the reasoning that stride-1 windows overlap by 19/20 and so carry only ~8,175 windows of
independent information. That statistic was used to argue the transformers are starved of data here.
**It does not survive contact with the benchmark these models were published on.**

Measured from the harness's own loader (`data_loader.py:35-36`), ETTh1's training split is
12×30×24 = 8,640 rows, giving **8,209 training windows** at Crossformer's published settings
(seq_len 336, pred_len 96). Our pooled training set is **163,500 windows — 19.9× more.** Crossformer
trains its 512,588 parameters on ETTh1 without difficulty. Applying the same overlap discount to
ETTh1 would imply ~24 independent windows for a half-million-parameter model, which is absurd
precisely because the model demonstrably works there. Each window carries a distinct target, so
information scales closer to the window count than to window-count/L.

The correct statement is that **sample count is not the binding constraint on this venue —
signal-to-noise is.** Daily equity returns are near-unpredictable (best daily IC here ~0.02), whereas
ETTh1's transformer temperature is strongly autocorrelated. A large model will overfit near-noise
regardless of how many windows it is shown.

Lyu et al. report 6,227 parameters for EXAMM on this universe, but that is the **sum of 50 per-stock
models** (98–188 each). Ours is one model. Different quantity; never the same column.

---

## Reading notes

**Book size is monotone in 2023 and that is the strongest evidence here.** +41.74 → +33.90 → +24.98 →
+15.25 as the book widens from 5 to 20 names, converging toward B&H's +7.91. That is the signature of
a genuine ranking signal concentrated in the tails: the extremes carry it, and diluting with
mid-ranked names washes it out. (At L/S 3, not shown in Table 1, it reaches +46.84 — the trend
continues.)

**2022 shows the same tail concentration with the sign reversed.** L/S 3 reaches −39.11 gross while
L/S 20 is only −6.55: the more concentrated the book, the worse. The extremes were anti-predictive
that year. Both years therefore say the ranking carries information at the tails; in 2022 the sign
was wrong. That is a more informative claim than "one year up, one down", and it matches the hit
rates (48.9% vs 53.9%).

**Neither walk-forward cell has a significant IC** (+0.0077 and +0.0162, both intervals spanning
zero). The supportable claims are parameter efficiency and the tail-concentration structure, not
accuracy superiority.

**Pre-commit to L/S 10 and to dollar-neutral.** L/S 5 gives the best 2023 number and a bad 2022 one;
L/S 20 gives the best 2022 number and the worst 2023 one. Reporting a different book size per year
would be indefensible. Hybrid (not shown) rescues 2022 at +14.40 while dollar-neutral loses — same
hazard, same answer: fix the choice in advance and report the rest as sensitivity.

## What to leave out

**Lyu et al.'s published transformer numbers as table rows.** Their models are per-stock on 70/15/15
with ~7,333 rows; ours is pooled on walk-forward splits with 3,290. Cite in text, never as a row.

**Long-only.** Loses in both years (−36.69, gross) and only confirms the edge sits on the short side.

**Accuracy win/lose language against the transformers.** Neither walk-forward cell is significant.
