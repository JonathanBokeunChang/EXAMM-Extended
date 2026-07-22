# ICAIF 2026 — Full Experimental Design (8 pages, deadline Aug 2)

**Working title:** *Deploying Evolved Recurrent Networks for Volatility Forecasting:
Pooling, Ensembling, and the Cost of Making Neuroevolution Work*

**Date:** 2026-07-21 · **Days remaining:** 12 · **Supersedes:** icaif_deployment_plan.md
(returns version) and extends volatility_paper_plan.md into an exact, day-by-day protocol.

---

## 0. Thesis, contributions, and what is already banked

**Thesis (one causal arc, not a list):** naive per-stock neuroevolution loses badly to HAR
→ we diagnose exactly why (estimation variance + selection winner's curse) → two recipe
changes fix it (pooled multi-series training; cross-run ensembling) and flip the result to
a significant win over HAR on all four standard metrics → the recipe creates real
deployment costs (N training runs; one big shared model) → warm-start continual evolution
and grow-shrink pruning pay those costs.

**Contributions:**
- **C1** Pooled+ensembled EXAMM beats HAR across MSE/MAE/QLIKE/R² (banked: 120 stocks,
  82–92% win rates, p = 4e-12…1.5e-20), to be confirmed walk-forward.
- **C2** Quantified diagnosis: val→test generalization drop 0.154 (EXAMM) vs 0.004 (HAR);
  9× differential pooling gain; 1/n ensemble-curve recovery. The winner's-curse anatomy of
  best-of-N selection in evolutionary NAS — domain-general.
- **C3** The recipe, with each lever causally mapped to a diagnosed failure and ablated.
- **C4** Deployment: warm-start (training cost) + grow-shrink (model cost), presented as
  known techniques applied to the cost problem the recipe creates.

**Banked results (do not re-run; cite from logs):**
| item | numbers |
|---|---|
| Main single-split result | 3-run pooled ens vs pooled HAR, 120 stks: MSE .1337/.1447 (99/120, p=4e-12) · MAE .2484/.2593 (100/120, 2e-10) · QLIKE −7.187/−7.021 (111/120, 1.5e-20) · R² .2922/.2473 (99/120, 2e-11) |
| Diagnosis | per-stock ens .128 vs HAR .206; val→test drop .154 vs .004; saturation ruled out (clamped HAR improves to .2295) |
| Pooling control | pooling gain EXAMM +.100 vs HAR +.011 (20-stk) |
| Ensemble curve (partial) | 1→.2065, 2→.2863, 3→.2922 |
| Grow-shrink per-stock | −12% nodes / −25% edges / −50% rec edges, all metrics parity (p>.6) |
| In flight | pooled grow-shrink 120 stks ×3 runs (run 1 ~38%) |

**Strategic advantages to exploit in the paper:** fully public data (Qlib) + public code =
complete reproducibility (rare in finance ML; say it loudly); diagnosis-first structure
(reviewers reward mechanism); every claim pre-registered before the confirming run.

---

## 1. Non-negotiable methodology (earned this session; a reviewer-proofing contract)

1. **Report all four metrics everywhere** (MSE, MAE, QLIKE, R²). Single-run verdicts were
   metric-dependent; one-metric reporting is cherry-picking.
2. **Fairness protocol for every learned baseline:** identical inputs, identical
   train/val/test rows, identical 3-seed ensembling, identical val-based early stopping /
   model selection. No baseline is handicapped; HAR gets both per-stock and pooled fits and
   we compare against its **best** config.
3. **No small-subset claims.** Nothing asserted from <100 stocks or a single window unless
   labeled pilot. (Burned twice: returns pilot GO; 20-stock pooled edge that reversed.)
4. **Validate every new harness against a known number** before trusting its output
   (the rank-label bug rule). LSTM harness must reproduce ~HAR-level R² before its EXAMM
   comparison counts; GARCH harness must produce sane persistence (α+β ≈ 0.9–0.99).
5. **Pre-register endpoints** (§5) in this file BEFORE the walk-forward campaign runs.
6. **Disclose compute asymmetry** (3 evolutionary runs vs closed-form HAR) with the
   ensemble-size curve and warm-start cost table as the mitigation.
7. Clock-immune progress tracking (genome counts, not wall time); caffeinate attached to
   every multi-hour run; never co-schedule two heavy jobs (the 5× contention lesson).

---

## 2. Task and data specification (exact)

- **Data:** Qlib `cn_data` daily OHLCV, 1999-11-10 → 2020-09-25 (4,943 trading days),
  CSI300 membership. Public; version-pin the bundle in the repro statement.
- **Vol estimator:** Parkinson daily range σ_P = sqrt( ln(H/L)² / (4 ln 2) ), floored at
  1e-9; work in **log** vol: LV = ln(σ_P + 1e-4). Robustness (R2): Garman–Klass.
- **Target:** TARGET(t) = mean(LV[t+1..t+H]), **H = 5** primary; H = 1 and H = 22
  robustness (single split only). Forward-aligned to row t → **`--time_offset 0`**.
- **Inputs (5):** LV, MA5(LV), MA22(LV), RET (log close-to-close), LOGVOL (log volume).
  All backward-looking; ffill then 0-fill NaN; drop stocks with <600 clean rows or any
  split <60 rows.
- **Universe:** all CSI300 members passing filters — target **n≈280–300** ("CSI300-full"),
  superset of the banked 120. The mechanism predicts more pooled data helps EXAMM;
  this doubles as a scale test (E1b).
- **Splits:**
  - *Fixed split (banked + ablations):* train ≤2017 / val 2018 / test 2019-01→2020-09.
  - *Walk-forward (primary for the paper):* expanding-window cohorts, test years
    **2016, 2017, 2018, 2019, 2020p** (2020 partial ≈ 180 days, disclosed):
    cohort Y = train ≤ Y−2 / val Y−1 / test Y. 5 cohorts.

## 3. Model zoo (exact configs)

| model | spec |
|---|---|
| **EXAMM (ours)** | pooled all-stock training via multi-file `--training_filenames`; 10 islands × 10; **10,000 genomes**; bp_iterations 10; num_mutations 1; normalize avg_std_dev; offset 0; **3 independent runs → prediction-mean ensemble of global bests** (no val selection — ensembling IS the selection story); `--save_genome_option none` (keeps global best only) |
| EXAMM-GS | + `--growth_phase_genomes 50 --reduction_phase_genomes 200` |
| EXAMM-WARM | cohort Y+1 run r seeded from cohort Y run r's global best: `--genome_bin <bin> --transfer_learning_version v1 --epigenetic_weights --start_filled` (validated flag set; never `--tl_epigenetic_weights`) |
| EXAMM-WARM-CHEAP | warm at **max_genomes 2000** (⅕ budget) |
| **HAR** | OLS on (LV, MA5, MA22); per-stock AND pooled; report best |
| HAR-X | HAR + leverage terms (neg-return magnitude, signed RET) — the stronger linear bar |
| **LSTM / GRU** | PyTorch CPU; 1 layer, hidden 32 (≈ EXAMM weight-count parity; report param counts); same 5 inputs, pooled training, Adam 1e-3, early stop on val (patience 10); **3 seeds → same ensemble protocol**. THE "did you need NAS?" baseline |
| GARCH(1,1), EGARCH | `arch` package, per-stock on RET; H-day variance aggregation → log-vol forecast; document the mapping |
| EWMA | RiskMetrics λ=0.94 on σ_P²; H-day flat projection |
| Naive | TARGET(t−1) persistence (random-walk-in-vol) — the floor |

## 4. Experiments

**E0 — Setup (D0):** `pip install torch --index-url cpu` + `arch`; extend
`qlib_vol_dataset_har.py` with `--cohort-year` and `--n-stocks all`; build CSI300-full
fixed-split + 5 cohort datasets; harness-validation runs (rule 4).

**E1a — GATE: PASSED (2026-07-21).** ✅ On identical test rows, identical recipe (pooled,
3-seed ensembles, val early-stopping), 120 stocks: **EXAMM R2 .2989 / QLIKE -7.194** vs
HAR .2371/-7.013, LSTM-32 .2109/-7.126, GRU-32 .2104/-7.133, LSTM-4 .2145/-7.140.
EXAMM wins **101-119/120** vs every comparator, p **8e-13 ... 2e-21**. Not a capacity story:
LSTM-4 (181 params) and LSTM-32 (5,025) both lose to EXAMM's ~tens of weights.
**Framing unlocked: "architecture search earns its keep."** Still to do: replicate on the
walk-forward cohorts (E2) before it is the primary claim; extend to CSI300-full.
*(Methodological note: the LSTM harness windows drop the first 22 test rows/stock; EXAMM was
re-evaluated on the SAME rows before comparison — the raw tables are not comparable.)*
Original gate spec (retained for reference):
- EXAMM > LSTM significantly → strongest paper ("architecture search earns its keep").
- EXAMM ≈ LSTM → honest reframe: "evolved networks match hand-tuned recurrent baselines
  **without architecture engineering**, and beat econometric standards"; deployment story
  carries the paper. Acceptable.
- EXAMM < LSTM → diagnosis+deployment paper with LSTM as the accuracy reference; rethink
  headline. (Prior: unlikely — but we test before writing, not after.)

**E1b — Scale replication (D1, parallel):** rerun banked comparison on CSI300-full fixed
split (3 runs). Prediction on record: EXAMM's margin grows or holds with more pooled data.

**E2 — Walk-forward campaign (D2–3, the paper's primary table):** COLD arm, 5 cohorts ×
3 runs, CSI300-full, all baselines refit per cohort. Primary endpoint evaluated here.

**E3 — Warm-start chain (D3–4):** WARM + WARM-CHEAP along 2016→2020p (4 links, 3 lineages),
seeded from COLD-2016. Canary one link before the chain (gate G3). Endpoints: TOST
equivalence vs COLD (margin 5% relative val MSE; also test-metric parity) +
**genomes-to-match-cold** efficiency curve + wall-clock table.

**E4 — Grow-shrink: DONE, NEGATIVE in the deployed regime (2026-07-21).** ✅
Pooled (120 stks, 3-run ens): GS is *worse on all four metrics* (MSE .1363 vs .1337 p=7e-4;
MAE .2554 vs .2484 p=1e-18; QLIKE -7.166 vs -7.187 p=3e-18; R2 .2915 vs .2922 p=6e-4) and is
NOT smaller (12 nodes/30 edges vs 10/21). Per-stock it *was* smaller at parity (-25% edges,
all p>0.6). **Interpretation:** pruning helps only where estimation variance binds; once
pooling removes the variance, the forced grow/shrink cycling is pure search disruption.
**Decision: pruning is EXCLUDED from the deployed recipe** and reported as a ~4-sentence
ablation (skeleton 6.3) that independently corroborates the C2 diagnosis. **No further GS
runs** — the 120-stock per-stock control would tidy the contrast but is a side finding, and
the effort belongs on E2/E3. Model-efficiency claim moves to E1a's parameter comparison.

**E5 — Ensemble-size ablation + the n=3 deployment decision (D4).**
Runs exist at **n=10 for every learned arm** (EXAMM COLD per cohort, EXAMM-GS, LSTM, GRU);
deterministic baselines (HAR/GARCH/EWMA/naive) are n=1 by construction — state this so the
asymmetry reads as inherent, not as a favour to EXAMM.
- **Curve** at n = 1,2,3,5,10, computed **per cohort** (not just the fixed split), for
  **R², QLIKE, and VaR coverage** — not R² alone. Reporting QLIKE/VaR by ensemble size is
  what earns any *tail-risk* language; an R²-only curve does not support a tail claim.
- **Subset bootstrap (the point of running 10):** over all **C(10,3)=120** three-run
  subsets, report the distribution of the win-rate vs best baseline. This upgrades the claim
  from *"our 3 runs beat HAR"* to *"**any** 3 runs beat HAR"* — and it is only possible
  because 10 were run. Report median and 5th percentile of the subset distribution.
- **Headline choice:** the paper deploys **n=3**, justified as the robustness/cost frontier
  (live risk systems have retraining-latency budgets), with the curve showing the asymptote
  and the bootstrap showing 3 is reliable rather than a lucky draw.
- *Why not economise to 3 runs everywhere:* on Anvil the 50-job array costs ~1% of the SU
  balance and the same wall-clock as 15 (all parallel), while 3-only would forfeit the
  subset bootstrap **on the primary walk-forward table** and force the curve to be measured
  on the fixed split but applied to walk-forward.

**E6 — Robustness (D5):** H=1 and H=22 targets; Garman–Klass estimator (fixed split,
EXAMM vs HAR/LSTM only).

**E7 — Economic significance (D5):** 1-day-ahead vol (from E6 H=1) → 1% and 5% VaR
(normal and Student-t(5) quantiles); **Kupiec** unconditional coverage + **Christoffersen**
independence/conditional coverage per stock; report violation rates + % stocks passing,
EXAMM vs HAR vs GARCH. Motivated by the measured asymmetry (EXAMM severe vol
under-forecasts 4.6% vs HAR 6.2%).

**E8 (stretch, only if D6 is free) — Second market:** Qlib US bundle, one fixed-split
replication, EXAMM vs HAR/LSTM. Kills "China-specific." If skipped → limitations + future work.

## 5. Pre-registered endpoints and decision rules (locked before E2 runs)

- **Ensemble size, locked:** all learned arms are run at **n=10**; the **main table compares
  at matched n** (no method under-ensembled); the **deployed/headline configuration is n=3**,
  with the C(10,3) subset bootstrap as its pre-registered robustness check. Deterministic
  baselines are n=1 by construction and this is stated in the text.
- **Primary:** walk-forward pooled-over-cohorts per-stock **QLIKE** and **R²**, EXAMM
  ensemble vs each baseline's best config; paired per stock×cohort Wilcoxon (primary)
  + paired t (secondary); significance α=0.05 with Holm correction across the baseline set.
  **Success claim requires:** EXAMM significantly better than *every* non-EXAMM baseline on
  QLIKE, and ≥ statistical parity on MSE/MAE/R² (no significant loss).
- **Model Confidence Set** (arch.bootstrap.MCS, QLIKE and MSE loss series pooled per
  stock-day): report which models survive at 90%; target: EXAMM-ens in MCS, ideally alone.
- **Diebold–Mariano** per stock (QLIKE differential, HAC small-sample variant): report %
  of stocks rejecting in EXAMM's favor at 5%.
- **Warm≈cold:** TOST, 5% relative margin on final val MSE, paired per cohort×lineage; plus
  genomes-to-match (median + IQR). Claim shape: "equivalent quality at X% of the search."
- **GS:** parity = no metric significantly worse (Wilcoxon p>0.05 with N≥100 stocks) at
  ≥20% edge reduction.
- No endpoint changes after E2 launches; anything post-hoc is labeled exploratory.

## 6. Paper skeleton (8 pp ACM sigconf, page budget)

| § | content | pp |
|---|---|---|
| 1 | Intro: deployment question, causal-arc summary, C1–C4 | 1.0 |
| 2 | Related work: HAR/RV lit; DL-for-vol; EXAMM/neuroevolution; NAS-in-finance | 0.75 |
| 3 | Task, data, models, fairness protocol (Table 1: model zoo + params + compute) | 1.25 |
| 4 | **Diagnosis** (Fig 1: val→test drop bars; Fig 2: ensemble curve; pooling-control table) | 1.0 |
| 5 | **Main results** (Table 2: walk-forward, all models × 4 metrics + MCS/DM; Fig 3: per-year win rates; scale test) | 1.75 |
| 6 | **Deployment** (Fig 4: genomes-to-match; Table 3: warm/cheap/GS cost-vs-quality) | 1.25 |
| 7 | Economic significance (Table 4: VaR coverage tests) | 0.5 |
| 8 | Limitations (single market unless E8; daily proxy vs intraday RV; compute asymmetry; 2020 partial) + conclusion | 0.5 |

Figures built with a single consistent style; every table cites its generating script.

## 7. Day-by-day (Jul 21 → Aug 2)

| day | work | gate |
|---|---|---|
| **D0 Jul 21** | finish pooled-GS runs; install torch/arch; cohort+full dataset builder; harness validations; **lock §5 pre-registration** | — |
| **D1 Jul 22** | **E1a LSTM/GRU gate** + econometric baselines on fixed split; launch E1b (300-stk, 3 runs, bg) | **G1: NAS-vs-LSTM verdict → sets framing** |
| **D2 Jul 23** | E2 COLD walk-forward launch (5 cohorts × 3; sequential locally w/ caffeinate, or Anvil if ported in ≤half a day); baselines per cohort (fast, CPU) | — |
| **D3 Jul 24** | E2 finishes; **G2:** walk-forward replicates main result. E3 warm canary → chain launch | **G3: warm canary** |
| **D4 Jul 25** | E3 finishes; E4 GS eval; E5 ensemble curve (7 extra runs bg) | — |
| **D5 Jul 26** | E6 robustness; E7 VaR; MCS/DM/TOST analysis code finalized | — |
| **D6 Jul 27** | full analysis freeze; all figures/tables generated from scripts; (E8 stretch if clean) | **G4: results frozen** |
| **D7–9 Jul 28–30** | write (methods → results → diagnosis → deployment → related → **intro last**); repro appendix | — |
| **D10 Jul 31** | full draft → Zimeng; her pass | — |
| **D11 Aug 1** | revisions, anonymization check (double-blind), buffer | — |
| **D12 Aug 2** | final read, **submit** | — |

Compute sanity: ~45 min/pooled 10k run when machine is free → E2 (15 runs) ≈ 11 h serial
(overnight D2→D3), E3 ≈ 9 h full + 2 h cheap, E5 ≈ 5 h, E1b ≈ 2.5 h. All fits locally with
overnight scheduling; Anvil optional accelerator, not a dependency.

## 8. Risk register

| risk | mitigation / fallback |
|---|---|
| **LSTM ≈ or > EXAMM (G1)** | pre-planned reframes in E1a; diagnosis + deployment + econometric wins still carry an honest paper |
| torch install friction (py3.9) | pin `torch==2.2.x` CPU wheel; fallback GRU in numpy (small, 1-layer) or sklearn MLP w/ lag stack + disclosed limitation |
| GARCH→H-day log-vol mapping contested | document formula; EWMA + naive corroborate; GARCH is secondary baseline |
| walk-forward shrinks the margin | report honestly; per-year figure shows where it holds; diagnosis unchanged |
| warm-start fails at pooled scale | G3 canary catches D3; paper falls back to COLD + GS + ensemble-curve deployment story |
| 2020 partial year | disclose; robustness excluding 2020p in appendix |
| time crunch | drop order: E8 → E6(H=22) → GS-cohort → E5(n=10 tail). Never drop: E1a, E2, E3, pre-registration |
| single machine dies | everything is scripted + logs in scratchpad; Anvil port docs in repo; caffeinate on all runs |

## 9. Reproducibility checklist (paper appendix)

- [ ] Qlib bundle version + download command; dataset builder script + exact filters
- [ ] Every table/figure regenerable by one script each (`results/paper/` scripts)
- [ ] EXAMM commit hash + full CLI for every arm; seeds for baseline runs
- [ ] Pre-registration file (this §5) committed before E2, with git timestamp
- [ ] Compute disclosure: runs × genomes × wall-clock, vs closed-form baselines
