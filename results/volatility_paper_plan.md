# Paper Plan — Pooled Ensemble Neuroevolution for Volatility Forecasting

**Status:** drafted 2026-07-21, after the pooled+ensemble result beat HAR on all four metrics.
**Working title:** *Evolving Recurrent Networks for Realized Volatility Forecasting:
Pooled Training, Ensembling, and Efficient Deployment*

---

## 1. Thesis

Neuroevolution (EXAMM) **beats the domain-standard HAR model** for realized-volatility
forecasting — but only with the right estimation recipe. Naive per-stock evolution *loses
badly*; the fix is **(a) pooled multi-series training** (cures data scarcity/variance) and
**(b) ensembling across runs** (cures selection overfitting / winner's curse). We diagnose
why the naive approach fails, demonstrate the fix, and then answer the deployment
questions: how to **maintain** these models cheaply over time (warm-start continual
evolution) and how **small** they can be made (grow-shrink pruning).

**Why this works where returns failed:** volatility is high-SNR, autocorrelated, persistent
and temporal — EXAMM's native habitat. Returns are none of those (documented in §7).

## 1b. Framing (ICAIF) — the causal arc, not a list

Frame the paper as **one argument**, where the efficiency work exists *because* the
accuracy fix is expensive — not as three separate results:

> naive per-stock neuroevolution fails → **diagnose why** (winner's curse + variance) →
> **fix it** (pooling + ensembling) → *the fix costs N× training and deploys one model
> across ~300 stocks* → **make it deployable** (warm-start cuts training cost, pruning cuts
> model cost).

Ensembling is what beats HAR, and ensembling is *what creates* the N-run cost that
warm-start solves; pooling is what beats HAR, and pooling is *what creates* the
one-big-model problem that pruning solves. Keep the **diagnosis (C2) as the intellectual
core** — it is the most novel and most domain-general piece — and present warm-start and
grow-shrink honestly as *known techniques applied to the specific cost problem this recipe
creates*, not as new methods.

## 2. Contributions

- **C1 (headline).** Pooled+ensembled EXAMM beats HAR on realized-vol forecasting across
  **all four standard metrics**, on 120 stocks, with p from 2e-09 to 3e-20.
- **C2 (mechanism).** A quantified diagnosis of *why* per-stock neuroevolution fails:
  selection overfitting (val→test drop 0.154 vs HAR's 0.004) plus variance from data
  scarcity. Domain-general finding about best-of-N selection in evolutionary NAS.
- **C3 (recipe).** Pooling and ensembling are the two levers, each mapped to a diagnosed
  cause, each with a controlled ablation.
- **C4 (deployment).** Warm-start continual evolution maintains quality at a fraction of the
  search budget; grow-shrink pruning reduces model size at parity.

## 3. Established results (already run)

| finding | numbers |
|---|---|
| Vol is forecastable (range-based Parkinson, HAR) | OOS R² **+0.22** median, positive in 87% of stocks |
| Per-stock EXAMM **loses** to HAR | +0.128 (10-run ens) vs **+0.206** |
| Diagnosis: selection overfitting | EXAMM val **+0.269** → test **+0.115** (drop .154); HAR drop **.004** |
| Saturation ruled out | clamping HAR to ±1 train-SD *improves* it (+0.230) → not the cause |
| Pooling control (20 stk) | pooling gain: EXAMM **+0.100** vs HAR **+0.011** (9×) |
| **MAIN RESULT — 120 stk, 3-run pooled ensemble vs HAR (best config = pooled HAR)** | **MSE .1337** vs .1447 (99/120, p=4e-12) · **MAE .2484** vs .2593 (100/120, p=2e-10) · **QLIKE −7.187** vs −7.021 (111/120, p=1.5e-20) · **R² .2922** vs .2473 (99/120, p=2e-11) — **EXAMM wins all four metrics, 82–92% of stocks** |
| Ensemble-size curve (R², partial E3) | 1 run **.2065** → 2 runs **.2863** (+.080) → 3 runs **.2922** (+.006); most gain from the 2nd member = 1/n variance scaling, as the winner's-curse diagnosis predicts |

## 4. Experiments

### E1 — Main comparison (extend what's done)
- **Universe:** expand 120 → **all liquid CSI300 stocks (~300)**. More stocks = more pooled
  data, which our mechanism predicts *helps EXAMM further* — a built-in test of the thesis.
- **Target:** forward 5-day mean log Parkinson vol. Robustness: forward 1-day and 22-day.
- **Baselines (must broaden beyond HAR):**
  - HAR (per-stock, pooled), HAR-with-leverage
  - **GARCH(1,1), EGARCH** (asymmetry), **EWMA/RiskMetrics**
  - **Fixed-architecture LSTM / GRU** ← isolates the value of *architecture search* vs "just use an RNN"
- **Metrics:** MSE, MAE, **QLIKE**, R² — report all four (the metric choice provably flips
  single-run verdicts; reporting one would be cherry-picking).
- **Stats:** paired Wilcoxon + t-test per stock; **Diebold–Mariano**; **Model Confidence Set**
  (the standard for vol-model comparison).

### E2 — Diagnosis (done; formalize)
Per-stock vs pooled; val→test decomposition; the 9× pooling-gain control. This is C2.

### E3 — Ensemble-size ablation
Ensemble of **1, 2, 3, 5, 10, 20** runs → performance curve on all four metrics.
Quantifies how much of the winner's curse each additional member removes, and where it
saturates (deployment-relevant: how many runs must you actually pay for?).
*Have: 1 → R² .2065, 2 → .2863. Need the rest.*

### E4 — Warm-start continual evolution (deployment)
- Walk-forward cohorts: train ≤Y (pooled), val Y+1, test Y+2, for Y = 2016…2019.
- **COLD:** re-evolve from scratch each year. **WARM:** seed cohort Y+1 run *r* from cohort
  Y run *r*'s best genome (`--genome_bin … --transfer_learning_version v1
  --epigenetic_weights --start_filled` — the validated flag set). **WARM-CHEAP:** warm at
  ⅕ budget.
- **Endpoints:** TOST equivalence on forecast quality (pre-specified margin) +
  **genomes-to-match-cold** (efficiency curve).
- *Note:* the existing warm-start machinery transfers directly; only the data/target change.

### E5 — Grow-shrink pruning (deployment)
- Standard vs grow-shrink (G:50/S:200), pooled, ensembled.
- **Endpoints:** model size (nodes/edges/weights) **and** all four forecast metrics.
- Hypothesis: parity or better at reduced size. Suggestive prior: the search already
  converges to compact genomes unaided (best genome observed at **7 nodes / 15 edges**).
- Pruning also matters for the pooled setting: one deployed model serving ~300 stocks.

### E6 — Temporal robustness
Repeat E1 across multiple walk-forward test years (not just 2019–20) to show the result
isn't window-specific. Reuses E4's cohort infrastructure.

### E7 — Economic evaluation (the "so what")
**VaR backtesting** — convert vol forecasts to 1%/5% VaR; **Kupiec** (unconditional
coverage) and **Christoffersen** (conditional coverage/independence) tests.
Motivated by a measured property: EXAMM makes **fewer severe under-forecasts** of vol
(4.6% vs HAR 6.2%), and under-forecasting risk is precisely the costly error in risk
management. This is where EXAMM's error profile should shine.

## 5. Data

- **Primary:** Qlib CSI300 daily OHLCV (2015–2020 in hand; bundle extends to 2008 — extend
  for more walk-forward cohorts).
- **Vol proxy:** Parkinson range estimator from daily high/low (Tier-2). Robustness:
  Garman–Klass.
- **Secondary (optional, strengthens generality):** US universe *if* the OHLC re-pull from
  Zimeng lands (`OPENPRC/ASKHI/BIDLO`), or Qlib US data. A second market answers "is this
  China-specific?"
- **Gold standard (future work):** intraday realized volatility (Qlib 1-min bundle).

## 6. Execution order

| phase | work | status |
|---|---|---|
| P0 | Main result on 120 stocks, 4 metrics, paired stats | **done** |
| P1 | Expand to ~300 stocks; add GARCH/EGARCH/EWMA/LSTM/GRU baselines; MCS + DM tests | to do |
| P2 | Ensemble-size ablation (E3) | partial |
| P3 | Grow-shrink (E5) | **running now** |
| P4 | Walk-forward cohorts + warm-start chains (E4, E6) | to do — biggest build |
| P5 | VaR backtesting (E7) | to do |
| P6 | Write up | — |

## 7. The returns work — how it fits in

The failed returns investigation becomes the paper's **motivation//contrast section**, not
wasted effort: it establishes *why volatility is the right target*. Documented nulls —
US returns (EXAMM ≈0 vs OLS +0.02), richer features (GBM ≈0), Qlib Alpha158 (EXAMM −0.021
vs GBM +0.024) — show the signal in returns is weak and, where it exists, tabular rather
than temporal. One clean sentence: *neuroevolution's value in finance is governed by
whether the target has exploitable temporal structure.*

## 8. Risks & mitigations

| risk | mitigation |
|---|---|
| Result is China-specific | add a second market (US w/ OHLC, or Qlib US) — E1 |
| HAR isn't a strong enough bar | add GARCH/EGARCH/EWMA **and** fixed LSTM/GRU; use MCS |
| Ensemble = more compute than HAR | disclose explicitly; report the ensemble-size curve (E3) so the cost/benefit is transparent; HAR is deterministic and cannot be ensembled |
| Single test window | E6 walk-forward across years |
| Warm-start fails at pooled scale | canary before full waves; fallback = COLD-only + E1/E3/E5 (paper still stands) |
| Grow-shrink hurts accuracy | report honestly as size/accuracy trade-off curve |

## 9. Non-negotiables (methodology discipline earned this session)

1. **Report all four metrics, always.** Single-metric reporting flips verdicts.
2. **Small-subset wins don't count** until they replicate at scale (bit us twice: the
   returns pilot GO, and the 20-stock pooled result that reversed at 120).
3. **Validate any new harness against a known benchmark** before trusting it (the Qlib
   rank-label bug produced a 4×-off number until caught).
4. **Pre-register endpoints** before treatment runs.
5. Disclose the compute asymmetry of ensembling.
