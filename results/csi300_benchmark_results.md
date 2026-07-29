# CSI300 MASTER-Replica Benchmark — Measured Results

**Session: 2026-07-26/27.** All numbers are measured, not estimated. Every model scores the SAME
frozen `eval_index.csv` row set with Newey–West HAC standard errors (`scripts/vol_baselines/ic_stats.py`).
Metric is daily cross-sectional Spearman (rank) IC vs the raw `LABEL`, averaged over test dates.

---

## 1. Datasets built

| dataset | features | label | eval rows | stocks | dates |
|---|---|---|---|---|---|
| `csi300_master_replica_invdata` | 158 Alpha158 + 42 market | `Ref(-5)/Ref(-1)-1` (H=5) | 251,217 | 451 | 871 |
| `csi300_master_replica_cndata` | same (replication feed) | same | 249,874 | 450 | 867 |
| `csi300_master_replica_invdata_alpha360` | 360 (60 lags × 6 fields) | `Ref(-2)/Ref(-1)-1` (H=2) | 250,614 | 434 | 871 |
| `csi300_master_replica_invdata_seq` | 6 fields/row, 60-day recurrence | `Ref(-2)/Ref(-1)-1` (H=2) | 235,277 | 429 | 871 |

Splits (MASTER's): train 2008-01-01–2014-12-31, val 2015-01-01–2016-12-31, test 2017-01-01–2020-08-01.
Source bundle pinned: `chenditc/investment_data` tag **2026-07-26**, sha256 `f4df2e16…`.

---

## 2. Harness validation — the load-bearing check

**First time in this project that any result was checked against an independently computed number.**

| | Rank IC |
|---|---|
| Published qlib GRU, Alpha360 (20-seed mean) | **0.0584** |
| Our GRU, Alpha360 flat (2 seeds) | **+0.0581** |
| Our GRU, gate `base` arm (different standardisation) | **+0.0580** |

Three routes, agreeing to ~1bp. Supporting checks:

- **Cross-source replication:** ridge on invdata **+0.0704** vs cndata **+0.0704** (spread 0.0001, HAC CIs overlap)
- **Placebo** (train labels shuffled within date): **+0.0019, HAC t 0.38** — CI spans zero
- **Determinism:** identical `content_sha256` across two independent builds
- **HAC self-check:** ratio **0.96×** on the non-overlapping next-day label vs **1.55×** on the 4-day
  overlapping label — the correction responds to real autocorrelation, not applied blindly

---

## 3. Venue matters — Alpha158 vs Alpha360

| venue | our linear ceiling | published GRU | recurrent headroom |
|---|---|---|---|
| **Alpha158** (engineered factors) | **+0.0739** | 0.0428 | **−0.031** (GRU LOSES to linear) |
| **Alpha360** (raw sequences) | **+0.0329** | 0.0584 | **+0.0255** (GRU WINS) |

On qlib's own leaderboard a plain **linear** model scores Rank IC 0.0472 on Alpha158, beating
LightGBM (0.0469), MLP (0.0429), LSTM (0.0435), GRU (0.0428), Transformer (0.0407), TCN (0.0421).
Alpha360 inverts this. **Alpha158 is the wrong venue for any recurrent/NAS claim.**

Alpha158 detail (invdata, HAC lag 4):
- full 200-feature ridge, CSRankNorm target: **+0.0704** (HAC t 8.59)
- Alpha158-only ridge: **+0.0739** (HAC t 8.87)
- raw-return target: **+0.0427** (HAC t 6.21) → **CSRankNorm target worth +0.0277, HAC t +3.97**
- market block contribution: **−0.0035** (HAC t −1.51) — see §7

---

## 4. GRU capacity sweep — the frontier

Sequence-form data, 306-stock shared training universe, 1 seed per size (h=64 is 2 seeds).

| model | params | Rank IC | HAC t |
|---|---|---|---|
| GRU 2×2 | 99 | +0.0295 | 5.54 |
| GRU 2×4 | 269 | +0.0407 | 7.92 |
| GRU 2×8 | 825 | +0.0364 | 6.61 |
| GRU 2×16 | 2,801 | +0.0453 | 9.14 |
| GRU 2×32 | 10,209 | +0.0488 | 10.70 |
| GRU 2×64 | 38,849 | **+0.0564** | 11.71 |

h=8 dipping below h=4 is 1-seed noise; CIs are ~±0.011 wide. Trend, not a precise predictor.

**GRU on the full 515-stock universe: +0.0571.** Restricting to the shared 306-stock universe
(−30.6% train rows, required for EXAMM fairness) cost only **7bp**.

---

## 5. EXAMM results — sequence form, pooled, depth 45

All at `bp_iterations=10`, `max_recurrent_depth=45`, 10,000 genomes, 306-stock universe,
`LABEL_CSRANK` target, scored once on the validation-selected `global_best_genome`.

| run | weights | nodes | edges | rec | Rank IC | HAC t |
|---|---|---|---|---|---|---|
| d45 run_1 (canary) | 49 | 9 | 14 | 5 | +0.0277 | 5.18 |
| d45 run_2 | 102 | 11 | 35 | 5 | +0.0278 | 4.93 |
| d45 run_3 | 78 | 9 | 24 | 9 | +0.0246 | 4.19 |
| **3-seed ensemble** | **mean 76** | | | | **+0.0289** | **5.06** |
| bpi30 (3× training) | 74 | 11 | 20 | 7 | **+0.0297** | 5.17 |

**Paired vs GRU 2×64: −0.0275, HAC t −5.66.** CIs do not overlap.

### THE CENTRAL FINDING

**EXAMM sits ON the GRU capacity curve, not above it.**

| | params | Rank IC |
|---|---|---|
| EXAMM 3-seed ensemble | 76 | +0.0289 |
| EXAMM bpi30 | 74 | +0.0297 |
| **GRU 2×2** | **99** | **+0.0295** |

Three independent EXAMM configurations, all ~75 weights, all ~+0.029, all statistically identical
to a hand-designed GRU of the same size. **Neuroevolution buys nothing here that simply shrinking a
GRU does not** — same accuracy, same size, no search required.

This kills the framing "EXAMM matches neural baselines at 100–500× fewer parameters." The 793×
ratio is real but meaningless without a capacity-matched baseline, and against one it evaporates.

---

## 6. Diagnosis of the gap — what was ruled out

| hypothesis | test | verdict |
|---|---|---|
| Genome undertrained | `finetune_rnn`, 100 epochs | **RULED OUT** — val MSE delta exactly **0.0000000000** |
| Training budget limits growth | bpi30 arm (3× bp_iterations) | **RULED OUT** — stayed 11 nodes / 74 weights, IC +0.0297 |
| Ensembling would rescue it | 3-seed ensemble | **RULED OUT** — +8% over mean seed (+0.0267→+0.0289), not the 4.5× seen elsewhere |
| Capacity binds | GRU capacity sweep | **CONFIRMED** — GRU@99 ≈ EXAMM@76; entire gap is size |
| Search/selection is broken (val r=+0.065 vs test) | implied by sweep | **WEAKENED** — EXAMM is ON the frontier, so selection isn't crippling it |
| **Mutation rates limit growth** | grow3 arm | **CONFIRMED** — see §8 |

---

## 7. Cross-sectional features HURT (nonlinear gate)

GRU ± 9 peer features (k=20 train-selected), Alpha360, matched architecture/lr/seeds/eval index:

| arm | Rank IC | HAC t |
|---|---|---|
| base | +0.0580 | 13.25 |
| base+CS | +0.0502 | 10.81 |
| **paired delta** | **−0.0078** | **−2.96** |

Within-seed (same init, only inputs differ): s0 −0.0061 (t −2.22), s1 −0.0084 (t −2.76), both same sign.

**Mechanism:** CS features made val MSE *better* (0.993286→0.991461) and test rank IC *worse*
(+0.0552→+0.0491).

> **CORRECTION (2026-07-29).** The explanation previously given here — *"peer aggregates carry the
> common component, which cancels in cross-sectional ranking by construction"* — is **FALSE**, and is
> refuted by `_cs_features.csv` in this repo. Date-fixed-effect variance decomposition of the nine
> features:
>
> | feature | between-date | within-date |
> |---|---|---|
> | `CS_PEER{1,5,20}` | 0.81 | 0.18 |
> | `CS_REL{1,5,20}` | 0.01–0.02 | **0.98–0.99** |
> | `CS_BETAADJ{1,5,20}` | 0.00 | **1.00** |
>
> **Six of nine features are 98–100% cross-sectionally varying**; nothing cancels. Only three are
> aggregates, and even those retain 18% within-date variance. `csi300_cs_operator_gate.py:178-188`
> asserts non-cancellation as a hard precondition of the very test that produced the −0.0078.
>
> Untested alternatives: input-width dilution (6→15 channels on a fixed 2×64 GRU, the CS block
> broadcast identically across all 60 timesteps); objective mismatch (val MSE improved while test IC
> fell — see §12); and redundancy, since `span{own, peer} = span{own, own−peer}` and own cumulative
> returns are linearly recoverable from the 60 close-ratio lags.
>
> Note also these were **k=20 train-CORRELATION-selected** peers, not concept/industry-membership
> peers. This result does not speak to concept-based grouping.

**Contrast:** HIST's *learned* relational structure = **+0.0083**; our *hand-specified* peer features
= **−0.0078**. Near-equal magnitude, opposite sign. The *learning* of relational structure appears to
be the essential ingredient; you cannot shortcut it with fixed peer averages.

Also verified: MASTER's 63 market features are **date-constant** (max within-date std across stocks
= exactly **0.0**), so for a linear model scored by rank IC their contribution is **zero by
construction**. An earlier "+0.0200, t=3.95" ablation of ours was a ridge-alpha artifact.

---

## 8. `max_recurrent_depth` and mutation rates (EXAMM internals)

**Depth bisect** (700 genomes each): depths **10 / 20 / 30 / 45 all clean**; depth 45 additionally
survived 2,116 genomes at bp_iterations=10 with zero errors. **Depth 60 CRASHES** after ~120–300
genomes:
```
ERROR: inputs_fired on RNN_Node 7 at time 55 is 7 and total_inputs is 6
```
EXAMM edge-bookkeeping bug — `total_inputs` fixed at edge construction, fatals on mismatch.

**Mutation rates were the binding constraint on genome size.** EXAMM's defaults are SYMMETRIC
(`add_node_rate` = `disable_node_rate` = 1.0, likewise edges), so the search sits under balanced
growth/shrink pressure and plateaued at **9–11 nodes** in every d45 seed and in bpi30.

With `ADD_NODE_RATE=3.0 ADD_EDGE_RATE=3.0 ADD_REC_EDGE_RATE=3.0` + `BP_ITERATIONS=30` (`grow3`):

| arm | genomes | nodes | edges | rec |
|---|---|---|---|---|
| d45 finals | 10,031 | 9–11 | 14–35 | 5–9 |
| bpi30 final | 10,031 | 11 | 20 | 7 |
| **grow3 @ 6,400** | **6,400** | **16** | **70** | **15** |

grow3 passed every prior arm's *final* structure using ~8% of the budget. Rates were exposed as CLI
flags in commit `d2beb32` (`--add_node_rate` etc.); misspelled `--*_rate` args are a hard error, since
a silently ignored flag would produce a clean-looking null.

**grow3 IC result: PENDING.** Curve prediction at ~200–250 weights: **+0.035–0.040**.
Within-EXAMM evidence (49w→+0.0277 vs 102w→+0.0278, no size effect) argues lower.

---

## 9. Reproduced: IC-vs-MSE campaign trading result (2023)

Archive `~/Downloads/ic_mse_results.tar.gz`, arm `mse_cohort_2021_aligned/run_1..10`,
50 US stocks, `cohort_2021_aligned`, 10-run ensemble, test year 2023.

| | memory note | **reproduced** |
|---|---|---|
| ensemble IC | +0.0195 | **+0.019496** |
| ICIR | 1.71 | **+1.712** |
| strategy return (long10/short10) | +33.0% | **+33.90%** |
| S&P (sprtrn) | +25.1% | **+25.08%** |
| equal-weight B&H | +7.9% | **+7.91%** |

**Caveats that still apply:** a trivial pooled OLS on the same cohort got a *higher* IC (+0.0257 vs
+0.0195); at 50 stocks × 1 year the IC is **not significant** (ICIR 1.71); single up-market year.
The market-relative **+8.8pts** is the signal, not the raw +33.9%.

**Useful implication:** IC → P&L leverage is real. IC +0.0195 produced +8.8pts over the S&P. The
CSI300 EXAMM ensemble is **+0.0289** on a venue with 306 stocks / 871 dates where the signal IS
resolvable — so a TopkDropoutStrategy backtest alongside IC is worth running.

---

## 10. Claims retracted this session (all mine)

1. **"The 63 market features add real signal (+0.0200, t=3.95)"** — FALSE. Date-constant features
   cannot reorder a cross-section. The gain was a ridge-alpha (shrinkage) artifact.
2. **"All t-stats"** — inflated ~1.5–2× by the 4-day overlapping label. Ridge t=13.50 → **8.59** HAC.
3. **"Linear CS gate FAILED → stop the CS-operator project"** — NOT SUPPORTED. The linear test is
   structurally blind to gating/attention mechanisms. All three prior CS gates were linear: one piece
   of evidence repeated three times with a shared blind spot.
4. **"EXAMM is undertrained / training budget is the bottleneck"** — ruled out by finetune (delta 0.0)
   and bpi30 (no change).
5. **"Peer aggregates cancel in cross-sectional ranking by construction"** (§7) — FALSE; 6 of 9
   features are 98–100% within-date varying. See the correction box in §7.
6. **"Seeded EXAMM degrades the seed by −0.0023"** (2026-07-29) — OVERSTATED. Every date-level
   paired contrast has |t| < 2. Supported: *fails to improve*, run-level t(5) = −4.25, sign 6/6
   p = 0.031. Not supported: any magnitude claim.
7. **"The seeded arms fell off the capacity curve"** (2026-07-29) — FALSE. The curve's own GRU
   residual sd is **0.00305**, so −0.0023 is 0.75 sd, inside its scatter. Relatedly, the seed's
   "exact" agreement with the curve (+0.04524 vs +0.04521 predicted) is coincidence at a resolution
   a 6-point, 1-seed fit cannot support.
8. **"Seeded degradation is a 20:1 train-budget mismatch (200 pretrain epochs vs 10 bp_iterations)"**
   (2026-07-29) — REFUTED three ways: lineages accumulate **17,500–17,770 total BP epochs**, not 10;
   corr(Δweights, ΔIC) = **+0.28**, positive when damage predicts negative; and `gs_run_2` shows the
   full deficit at **+12 weights** from the seed, i.e. with nothing to damage. Validation also
   improved *population-wide*, not just for the selected genome.

**The recurring error:** running a test structurally incapable of showing the effect, then reading
its null as evidence against the effect. *Before trusting any null, ask: could this test have
produced a positive result if the hypothesis were true?*

**A second recurring error, added 2026-07-29:** quoting a **date-level** HAC t for an effect whose
unit of randomisation is the **seed**. The attention gate's "attn − mean = −0.0036, HAC t −2.93" is
t = **−0.82** across its 3 seeds (one seed at −0.0124 against two zeros). Report both, and gate on
the seed-level statistic.

---

## 11. What the paper is now

The efficiency framing is dead on this venue. What survives is a **methodological contribution**:

> **Capacity-matched baselines.** NAS efficiency claims routinely compare an evolved model against a
> *default-sized* baseline and report the parameter ratio. Performing the obvious control — shrinking
> the baseline to matched capacity — made a 793× "advantage" disappear entirely. Supported here by a
> six-point capacity curve on an externally-validated harness.

Ranked next tests:
1. **grow3 → eval vs the curve** (running) — does EXAMM beat a GRU of equal size?
2. **Depth-10 control arm** — isolates multi-scale, EXAMM's one distinctive capability
3. **grow3 seeds 2–3** — ensemble + error bars
4. Val-IC selection *(demoted — EXAMM is on the frontier)*
5. Multi-objective NSGA-II *(heavily demoted — would retrace the GRU curve)*
6. Rank-IC fitness *(5/5 prior failures, incl. soft-rank Spearman at −0.0001)*
7. Learned relational operator *(only path to SOTA; weeks of C++)*

---

## 12. Validation does not predict test rank IC — four independent inversions

The single most reproducible finding on this venue. Across EXAMM configurations the measured
correlation is **r = +0.065 (n=6)**. Worse than uninformative, it repeatedly ranks *backwards*:

| # | experiment | validation says | test rank IC says |
|---|---|---|---|
| 1 | grow3 (192w) vs pretrained seed (3,079w) | grow3 better (0.332738 vs 0.333237) | grow3 **worse** by 0.0123 |
| 2 | §7 CS features | better (0.993286 → 0.991461) | **worse** (+0.0552 → +0.0491) |
| 3 | attention gate readouts | `last` **worst** of three | `last` **best** of three |
| 4 | seeded EXAMM (2026-07-29) | improved 0.333237 → 0.332578 | **fell** +0.0452 → +0.0429 |

**Why it matters beyond this project:** in any NAS system validation fitness *is* the search signal.
Case 4 is the cleanest demonstration — 1,800 genomes and ~17,600 accumulated BP epochs moved
validation **down** 0.00058 and test rank IC **down** 0.0042 *across the whole population*, while
selection contributed nothing: the three validation-selected global bests (mean +0.0405) are
indistinguishable from a random island champion (+0.0411).

**Resolution floor.** Six seeded architectures differing by up to 4 nodes were separated by **3e-5**
on validation MSE — thirty to sixty times smaller than differences already shown here to carry no
test information. Selection at that spread is selection on noise.

**Readout dependence.** The apparent seeded deficit is partly an artifact of the narrowest readout:

| readout | plain | grow-shrink |
|---|---|---|
| 3 validation-selected global bests | +0.0429 | +0.0431 |
| 30 island champions | **+0.0440** | **+0.0451** |
| seed | +0.0452 | +0.0452 |

**The grow-shrink island ensemble ties the seed (+0.0451 vs +0.0452).** So "all six runs below the
seed" is a property of the 3-genome readout, not of the arm: picking three champions with a selector
whose val→test correlation is +0.065 is a winner's-curse operation, and sampling the population more
broadly recovers nearly the whole deficit. Report the seeded arms as **matching, not improving,** the
seed — and report the readout dependence as part of the finding, since it is what demonstrates the
search added variance rather than signal.

---

## 13. Controls run 2026-07-29

**LR control — the GRU baseline is correctly tuned.** `LR["gru"] = 2e-4` is sourced from
`workflow_config_gru_Alpha158.yaml` but used on Alpha360, so it was checked directly:

| | epochs | test IC | HAC t |
|---|---|---|---|
| lr 2e-4 (banked config) | 27, 30 | **+0.0581** | +13.24 |
| lr 1e-3 | 13, 15 | +0.0565 | +12.88 |
| paired (1e-3 − 2e-4) | | **−0.0015** | **−0.92** |

The banked +0.0581 reproduced to four decimals, and the higher rate is *not* better. **§4, §5 and §7
stand as written.** Mechanism: the slower rate trains ~2× longer before validation turns.

**Still unexplained:** the attention gate's no-attention control arm scored **+0.0616** — the highest
number measured on this venue, above published GRU (0.0584). Learning rate is now ruled out.
Remaining confounds: an `fc_in` Linear(6,64)+Tanh projection, a doubled readout, 3 seeds vs 2, no
input/target standardisation, batch 2000 vs 512. Worth a controlled follow-up.

**Ensemble breadth does not pay.** Volatility venue, 238 stocks, 10 → 100 → 1000 members: median R²
moves ±0.002, and in one arm the widest ensemble was *worse* than the narrowest. Today's global-best
ensembles gained +2.9% (gs) and +6.2% (plain) over their mean single run — under the +8% precedent.

**Attention/pooling — no effect detected, underpowered.** `last` +0.0616, `mean` +0.0611,
`attn` +0.0575. Seed-level t: attn−mean **−0.82**, mean−last **−0.05**. Report as underpowered, never
as a negative effect.
