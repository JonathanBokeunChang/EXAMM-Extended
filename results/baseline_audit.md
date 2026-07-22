# Baseline Audit — Walk-Forward EXAMM Stock Forecasting

**Date:** 2026-07-20
**Scope:** the standard per-stock EXAMM recipe (50 independent genomes, raw next-day
return target, MSE training) evaluated on a 4-cohort calendar walk-forward.
This is the *baseline* against which all Paper 2 treatments (shared-genome
estimation, relative-return targets, ranking loss) are compared.

Every number below was independently re-derived from the raw prediction files and
checked with control/plumbing sanity tests plus a moving-block bootstrap. See
**§6 Soundness** for the verification battery. Two comparators were added on
2026-07-20 and are load-bearing for the conclusion: a **trivial linear baseline**
(§3) on the identical features/target/days, and a **replication of the source
paper's long-only strategy** (§5.2) on this universe.

---

## 1. Campaign integrity

Four cohorts, 50 stocks × 10 runs each (2,000 training jobs), Anvil (gcc 11.2.0,
OpenMPI 4.1.6), binary at commit `a62ae61`.

| cohort | train ≤ | val | test span | runs | failures | weight_decay | VERIFY OK | sweep corrupted |
|---|---|---|---|---|---|---|---|---|
| cohort_2018 | 2018 | 2019 | 2020–2023 | 500/500 | 0 | 0 | 500/500 | **0/500** |
| cohort_2019 | 2019 | 2020 | 2021–2023 | 500/500 | 0 | 0 | 500/500 | **0/500** |
| cohort_2020 | 2020 | 2021 | 2022–2023 | 500/500 | 0 | 0 | 500/500 | **0/500** |
| cohort_2021 | 2021 | 2022 | 2023 | 500/500 | 0 | 0 | 500/500 | **0/500** |

Sixth consecutive fully-clean campaign since the depth-sort corruption fix
(baseline v2 + grow-shrink + these 4 cohorts = 3,000 verified genomes, 0 corrupted).
Selection protocol: per stock, the run with the best **validation-year** MSE among
verified-clean genomes (identical to all prior campaigns). No test-set information
enters selection.

## 2. Primary endpoint — cross-sectional Information Coefficient (IC)

**Definition.** Each trading day, rank all 50 stocks by predicted next-day return
and by realized next-day return; take the Spearman rank correlation. IC = mean of
that daily number over a test period. IC measures the one thing a forecaster
controls — *did it sort winners from losers* — independent of strategy knobs and
single-name luck.

**Endpoint.** Pooled IC over each cohort's **fresh year** (its first test year, i.e.
models ~1 year old), 2020–2023, 1,005 trading days. Two model views are reported:
the **selected** genome (deployment-realistic) and the **10-run ensemble** (mean of
all 10 runs' predictions — noise cancellation). The **reversal rule** (short
yesterday's winners) is the signal-ceiling reference, computed on identical days.

| view | pooled fresh-year IC | naive t | block-bootstrap 95% CI | significant? |
|---|---|---|---|---|
| baseline, selected | −0.0045 | −0.93 | [−0.0146, +0.0046] | **no** (CI straddles 0) |
| baseline, ensemble | −0.0024 | −0.43 | [−0.0112, +0.0066] | **no** (CI straddles 0) |
| reversal (bar) | **+0.0186** | **+2.35** | **[+0.0027, +0.0344]** | **yes** (CI excludes 0) |

**Result.** On identical days, in the identical universe, a cross-sectional signal
worth IC ≈ +0.019 exists and is significant even under the autocorrelation-robust
block bootstrap. The standard EXAMM recipe captures none of it — statistically
indistinguishable from zero whether one takes the best run or averages all ten.
Ensembling shifts the estimate toward zero (as noise-cancellation should) and
confirms there is no hidden signal to recover.

### Per-cohort fresh-year IC

| cohort | fresh year | model age | selected | ensemble | reversal |
|---|---|---|---|---|---|
| cohort_2018 | 2020 | ~1.5 y | +0.0103 | +0.0095 | +0.0015 |
| cohort_2019 | 2021 | ~1.4 y | +0.0031 | −0.0025 | **+0.0419** |
| cohort_2020 | 2022 | ~1.4 y | −0.0180 | −0.0075 | +0.0064 |
| cohort_2021 | 2023 | ~1.4 y | −0.0135 | −0.0092 | +0.0249 |

The 2021 row is the thesis in one line: in the **peak-signal year** (reversal
+0.042, its highest), the baseline captured **−0.003** — maximum available signal,
zero capture. Conversely in 2020, the one year the ceiling itself was dead
(reversal +0.002), the model "matched" it at ≈0. The recipe's capture is
uncorrelated with how much signal exists.

## 3. Linear baseline — does neuroevolution earn its keep?

**Question.** Everything in §2 is measured against zero-training rules. That leaves
open the reviewer's first question: on the identical 6 features, identical next-day
target, identical days, would a *trivial* fitted model do better, the same, or worse
than the neuroevolution search? A per-stock **OLS** (and val-α-selected **Ridge**)
answers it. Parity is exact — same inputs (`RET VOL_CHANGE BA_SPREAD ILLIQUIDITY
sprtrn TURNOVER`), `time_offset 1` target, fit on **train only** (val used solely for
Ridge α, mirroring EXAMM's val-only genome selection), predictions in raw RET units
so the cross-section is comparable exactly as EXAMM's denormalized output is. The
per-stock↔shared axis is also probed with a single **pooled ("shared") OLS**.

| view | pooled fresh-year IC | naive t | block-bootstrap 95% CI | significant? |
|---|---|---|---|---|
| baseline EXAMM, selected | −0.0045 | −0.93 | [−0.0146, +0.0047] | no |
| baseline EXAMM, ensemble | −0.0024 | −0.43 | [−0.0114, +0.0072] | no |
| **per-stock OLS** | **+0.0196** | **+3.04** | **[+0.0064, +0.0333]** | **yes** |
| per-stock Ridge (val-α) | +0.0194 | +2.99 | [+0.0061, +0.0335] | **yes** |
| shared OLS (pooled) | +0.0193 | +2.47 | [+0.0042, +0.0357] | **yes** |
| reversal (bar) | +0.0186 | +2.35 | [+0.0021, +0.0348] | **yes** |

**Result — the finding inverts the naive expectation.** A trivial linear regression
recovers **essentially the full available signal** (IC +0.0196, CI excludes 0, sitting
right at the +0.019 reversal ceiling), while the entire EXAMM neuroevolution recipe
recovers **none** of it (≈0, CI straddles 0). Ridge and shared-OLS agree with
per-stock OLS, so it is neither an OLS-overfit artifact nor a per-stock↔shared
structure effect. **The 6-feature set is *not* the ceiling** — the signal is linearly
learnable; EXAMM's failure is EXAMM-specific.

**Mechanism (measured, not asserted).** Adversarial hardening
(`scratchpad/hardening_checks.py`) pins down *why*:

- **Reversal loading.** OLS's daily cross-section ranks like the reversal factor at
  rank-correlation **+0.358**; EXAMM's at **+0.032**. OLS *found* the one robust
  linear factor (median fitted coef on RET = −0.038, negative in 149/200 stock-fits =
  short-term reversal); EXAMM never locked onto it.
- **Not a collapse.** EXAMM's predictions are actually *more* dispersed cross-
  sectionally than OLS's (41% vs 9% of realized dispersion) — it makes confident,
  varied predictions that are simply **misranked** (dispersed noise), the textbook
  signature of a flexible model overfitting weak signal.
- **Placebo.** OLS refit on shuffled training targets → pooled IC +0.004 ≈ 0; the
  +0.020 is genuine signal, not a pipeline that returns positive regardless.

**Capacity does not rescue it (grow-shrink).** If the mechanism is overfitting, model
shrinkage should help. Measured directly on the 70/15/15 all-50-present window (607
days, 2021-08 → 2023-12; `scratchpad/gs_ic.py`), grow-shrink's cross-sectional IC —
an endpoint it had never been scored on — is:

| model (70/15/15, 607-day full cross-section) | cross-sectional IC | 95% CI |
|---|---|---|
| baseline_v2 (full size) | −0.0101 | [−0.0203, +0.0006] straddles 0 |
| grow-shrink (half size) | −0.0001 | [−0.0123, +0.0125] straddles 0 |
| per-stock OLS | +0.0204 | [+0.0058, +0.0410] excludes 0 |
| reversal (bar) | +0.0221 | [+0.0078, +0.0432] excludes 0 |

Halving the model (grow-shrink/baseline daily rank agreement only +0.26 — a large
reshuffle) moves IC from mildly negative to dead zero, still ~0.02 below the linear
ceiling. Direction is consistent with "less capacity → less noise," magnitude is
nowhere near enough: you would have to shrink to near-linear rigidity to reach the
ceiling. **Structure (per-stock↔shared) and capacity (full↔half) both fail to close
the gap** — the two levers most people would try first are ruled out.

## 4. No staleness effect (replication of the pilot GO — refuted)

A 10-stock pilot had suggested fresh models carried a tradable edge (IC +0.023) that
decayed with age. At 50-stock scale this did not replicate, and the failure is not a
dilution artifact:

- **Fresh vs stale, identical CY2023 days (50 stocks):** fresh (cohort_2021, age
  ~1.4 y) IC −0.0135; stale (v2 baseline, age ~2.4 y) IC −0.0159. Paired daily
  difference ≈ 0 (fresh > stale on 119/249 days — a coin flip).
- **Cross-vintage age ladder, identical CY2023 days:** young (1.4 y) −0.0135 vs old
  (4.4 y from cohort_2018) −0.0070 — statistically indistinguishable, no gradient.
- **On the pilot's own 10 stocks**, new runs scored IC −0.0059 vs the pilot
  campaign's +0.0233 — same stocks, same days, same recipe. The pilot's edge was
  run-to-run selection luck (single-run IC spread across 10 runs: sd ≈ 0.008), not
  a real effect.

There is no decay because there was never cross-sectional signal to decay.

## 5. Economic co-primary — trading (CY2023, identical days)

### 5.1 Daily long-short (market-neutral)

Daily long-short via the Financial_toolbox driver. FRESH = cohort_2021 (age ~1.4 y),
STALE = v2 baseline restricted to CY2023 (age ~2.4 y), REVERSAL = short yesterday's
winners. Benchmarks over the same 249 days: equal-weight buy-and-hold **+7.91%**,
S&P **+25.08%**.

| cell | FRESH | STALE | REVERSAL |
|---|---|---|---|
| long 5 / short 5 | −10.68% | −11.45% | **+57.56%** |
| long 10 / short 10 | −14.10% | −13.81% | **+22.13%** |
| long 15 / short 15 | −11.82% | −10.16% | **+12.67%** |
| 10/10, net of transaction costs | −13.67% | — | **+22.37%** |

The economic picture matches the IC picture exactly: the standard recipe loses at
every setting, fresh ≈ stale, while the reversal signal — computable from a column
these models receive as an **input** — earns +22% *net of costs* on the same days.

### 5.2 Long-only — replication of the source paper's headline (Algorithm 1)

The source paper (Lyu et al., arXiv 2410.17212) reports its strongest result with a
**long-only** strategy (its Algorithm 1) on the **30 DJI** stocks: 2023 EXAMM
**+39.05%** vs equal-weight buy-and-hold **+3.23%**, DJI +13.70%, S&P +26.29% — EXAMM
beats B&H by +35.8 pp and beats the S&P. Running *that exact strategy* (the toolbox's
`portfolio_simple_return`) on **this 50-stock universe**, identical 249-day 2023
window, using the fresh cohort_2021 models (train ≤2021 / val 2022 / test 2023, the
same convention as the paper's 2023 dataset):

| strategy (long-only, 2023, 50 stocks) | return | vs EW B&H (+7.91%) | vs S&P (+25.08%) |
|---|---|---|---|
| **Fresh EXAMM** (cohort_2021) | **+10.29%** | +2.4 pp — edges it | loses |
| Stale EXAMM (v2, ~2.4 y) | −9.22% | loses | loses |
| **Reversal rule** (zero training) | **+17.44%** | +9.5 pp — beats it | loses |
| Oracle (perfect foresight) | +1214.22% | — | — |

**Reading.** The paper's pattern reproduces only in *direction* (fresh EXAMM edges
equal-weight B&H) and collapses in *magnitude* (+2.4 pp here vs the paper's +35.8 pp),
and unlike the paper it **loses to the S&P**. The decisive control: a **zero-training
reversal rule beats EXAMM at its own long-only game** (+17.44% vs +10.29%). So the
small long-only edge is not selection skill — it is a weak partial capture of the same
reversal factor a one-liner captures better. The fresh≫stale gap here (+10.29 vs
−9.22) is a *directional/market-timing* effect (long-only carries beta), fully
consistent with the market-neutral finding that fresh ≈ stale ≈ 0 once direction is
stripped out (§2, §4). The oracle's +1214% confirms the *strategy mechanism* has vast
headroom — the binding constraint is prediction quality, not the strategy.

The headline in the source paper is thus a **long-only, directional, single-universe,
no-reversal-control** result; under the stricter, market-neutral, reversal-benchmarked
test applied here, the selection skill is ~0. This audit's methodology is a superset of
the paper's, not a contradiction of it.

## 6. Soundness / verification battery

Run 2026-07-20 (`scratchpad/verify_baseline.py`, `linear_baseline.py`,
`hardening_checks.py`, `gs_ic.py`), independent re-derivation from raw prediction
files:

| check | result |
|---|---|
| Calendar identity (all 50 stocks share the test calendar, per cohort) | OK, all 4 cohorts |
| Alignment/denorm: `expected_RET` == raw next-day RET | max diff **0.00e+00** |
| Linear-baseline parity: OLS's target == EXAMM's `expected_RET`, same days | max diff **~1e-19**, all 4 cohorts |
| Control: random predictions → IC | −0.009 (≈ 0, single draw) |
| Plumbing: oracle (actual returns as prediction) → IC | **+1.0000** exactly |
| Linear placebo: OLS on shuffled train targets → IC | +0.004 (≈ 0) |
| Trading alignment: `expected_RET` == RET[t+1] per stock (toolbox driver) | 50/50 stocks OK |
| Inference: naive-t vs moving-block bootstrap (block 20, B 2000) | both reported above |

The control (random ≈ 0), plumbing (oracle = +1), and placebo (shuffled ≈ 0) checks
confirm the IC estimator and the linear pipeline are both correct; the exact
`expected_RET` identities confirm predictions are denormalized and time-aligned across
EXAMM, the linear models, and the trading driver; the block bootstrap confirms
significance conclusions survive daily-IC autocorrelation.

## 7. Caveats (disclosed, not fixed)

- **Survivorship bias:** the 50-stock universe was selected as companies alive
  through 2023, inflating absolute returns. Mitigated because all headline
  comparisons are *within-universe on identical days* (the bias largely cancels),
  but "beats the market" claims are weakened by it.
- **Feature set, corrected:** the 6 predictors contain essentially one exploitable
  factor (short-term reversal), so the reversal bar (+0.019) is a realistic upper
  bound *on this feature set*. It is **not** a modeling ceiling — a linear model
  reaches it (§3); the gap is EXAMM's, not the features'. Richer features (a higher
  ceiling for *any* model) remain named future work.
- **Validation-year embargo:** the train ≤Y / validate Y+1 / trade Y+2 convention
  (matching Lyu et al. arXiv 2410.17212) means models are ~1 year old on their first
  trading day — a conservative choice that understates fresh performance.
- **Control single-draw:** the random-IC control used one seed (−0.009); its purpose
  is machinery validation, not a distributional claim.
- **Grow-shrink window:** §3's grow-shrink IC is measured on the 70/15/15 all-50
  window (607 days, 2021–2023), a different construction than the walk-forward
  fresh-year endpoint; it is directly comparable to baseline_v2 on that same window
  (identical per-stock test windows), not to the pooled §2 number.

## 8. Conclusion

The standard per-stock, raw-target, MSE EXAMM recipe has **no cross-sectional
forecasting signal** at 50-stock scale — fresh or stale, selected or ensembled, in
any regime. The audit's new comparators locate *where* that failure lives:

1. **It is not the features/objective.** A trivial per-stock OLS on the identical
   features and target captures essentially the full available signal (IC +0.020 ≈
   the reversal ceiling, CI excludes 0); EXAMM captures ~0. The signal is linearly
   learnable — EXAMM leaves it on the table, producing dispersed predictions
   uncorrelated with the one robust factor (reversal loading +0.03 vs OLS's +0.36).
2. **It is not estimation structure or capacity.** Per-stock↔shared (shared-OLS ≈
   per-stock OLS) and full↔half size (grow-shrink IC ≈ 0) both fail to close the gap.
   Four axes tried — fresh/stale, selected/ensemble, structure, capacity — four nulls.
3. **The source paper's headline was direction, not skill.** Its long-only strategy
   replicates only weakly on this universe (+2.4 pp over B&H vs its +35.8 pp), loses
   to the S&P, and is beaten by a zero-training reversal rule — the benchmark it never
   ran. Stripped of beta/direction, the selection skill is ~0.

This is a clean, statistically-sound negative result and a precise motivation for the
Paper 2 treatments: since structure and capacity are ruled out, the remaining lever
with a mechanism is the **objective** — MSE-selection over a flexible search space
cannot distinguish the factor-capturing genome from noise-fitters (the factor is too
weak to move MSE), so a **cross-sectional ranking / IC-aligned objective** is the
principled next experiment. The four ablations above are its motivation section.
