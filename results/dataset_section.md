# 4 CRSP Dataset

The Center for Research in Security Prices (CRSP) dataset is a comprehensive collection of data on
the U.S. stock market. It contains all stocks on the NYSE, AMEX, and NASDAQ, and covers a wide range
of market indexes, open-end mutual funds, exchange-traded funds (ETFs), U.S. Treasury securities, and
corporate bonds. CRSP data is known for its high level of accuracy and is highly reliable for
academic research and financial modeling. Its historical range extends from December 1925 to the
present, at daily, monthly, quarterly, and annual frequencies, and it contains variables central to
financial research such as stock prices, returns, trading volumes, and dividend yields.

## 4.1 Stock Selection

We adopt the stock selection of Lyu et al., in which 50 mid-cap companies<sup>1</sup> were drawn from
the S&P 500 index for return forecasting. Mid- and small-cap stocks present a particular opportunity
for return forecasting because they are less efficiently priced than large-caps. Large-cap stocks are
extensively analyzed by institutional investors and market analysts, so new information is
incorporated into their prices quickly; mid- and small-caps receive less scrutiny, leaving patterns
and inefficiencies that a forecasting model may exploit. Because the S&P 500 primarily comprises
large- and mid-cap U.S. companies, restricting attention to its 50 mid-cap constituents isolates the
more forecastable segment of the index.

Retaining the same universe allows our results to be placed directly alongside the transformer
baselines reported in that work, since any difference in performance cannot be attributed to a
difference in the stocks being forecast.

## 4.2 Predictors

From this dataset we use the same six economic predictors associated with stock returns, shown in
Table 1. These were either obtained directly from CRSP or computed from CRSP variables.

**Table 1.** Economic predictors for stock return prediction.

| Predictors | Description |
|---|---|
| Return | Percentage change in stock price |
| Volume Change | Percentage change in trading volume |
| Bid-Ask Spread | (*AskPrice* − *BidPrice*) / *Price* |
| Illiquidity | *Return* / (*Volume* × *Price*) |
| Turn Over | *Volume* / *SharesOutstanding* |
| S&P 500 Return | S&P 500 index return |

Return is the percentage change of a stock's price from time [*t*−1] to time [*t*], which for this
data is day-to-day; Return is also the forecasting output parameter. Volume Change is the percentage
change of the stock's trading volume over the same interval. Bid-Ask Spread measures the difference
between the highest price buyers are willing to pay (Bid) and the lowest price sellers are willing to
accept (Ask), with the Ask price usually being higher than the Bid. Illiquidity measures how readily
the stock can be sold for cash. Turn Over measures the rate at which a stock is sold during a given
period and is a further measure of liquidity. S&P 500 Return is the return of the entire S&P 500
index and serves as a benchmark for overall market performance. Together these predictors carry both
firm-level and market-wide information, and each has been shown to be associated with stock returns.

Because our files retain the underlying CRSP columns from which these quantities are derived
(*AskPrice*, *BidPrice*, *Price*, and *SharesOutstanding*), we were able to confirm each definition
directly rather than assume it. The Bid-Ask Spread and Illiquidity identities in Table 1 reproduce
the stored values exactly on all 3,290 rows tested.

Two properties of the data are worth stating explicitly. First, Illiquidity as defined here is
*signed*: it is constructed from Return rather than |Return|, and therefore departs from the standard
Amihud (2002) illiquidity measure. We confirmed that the sign of Illiquidity agrees with the sign of
Return on every row. We retain the original definition for comparability with prior work. Second,
Bid-Ask Spread takes negative values on approximately 0.3% of rows, arising from crossed quotes in
which the recorded Bid exceeds the recorded Ask. This is a known CRSP artifact, and we leave the
affected rows unmodified rather than filter them.

## 4.3 Experimental Stock Datasets

The forecasting task is one-step-ahead prediction: a model observes the six predictors at day [*t*]
and predicts Return at day [*t*+1]. This shift is applied within each stock's file, so the final
observation of any file has no target and is unused.

Where prior work trained 50 separate per-stock models, and additionally a single model on a
horizontally combined dataset of all 50 stocks, we instead train **one pooled model** across the
universe. Each stock remains a separate sequence, but a single network — with a single set of weights
— is fitted across all 50 simultaneously. No stock identifier is supplied among the inputs, so the
network cannot distinguish one ticker from another and must learn a single forecasting function
applied uniformly to every stock.

This is the standard *global model* formulation in contemporary forecasting and in empirical asset
pricing, and it is the configuration used throughout Qlib's model zoo. It carries three advantages
relevant to this work. The model sees fifty times the training data available to any individual
model, which is a substantial regularizer at the parameter counts EXAMM produces. Its size is
independent of universe size, so stocks may be added or removed without retraining a new model or
reshaping an existing one. And because all predictions come from one function, they lie on a common
scale, which is a prerequisite for the daily cross-sectional rank correlation we use to evaluate
forecast quality.

Rather than a fixed proportional split, we divide the data by calendar year to mirror the deployment
setting in which a model is trained on history, validated on the following year, and traded in the
year after that. Two such walk-forward cohorts are used, summarized in Table 2. Row counts are per
stock and are identical across all 50 stocks within a cohort.

**Table 2.** Walk-forward cohort definitions. Counts are rows per stock.

| Cohort | Training | Validation | Test |
|---|---|---|---|
| 2020 | 3,290 rows, 2007-12-07 to 2020-12-31 | 252 rows, 2021 | 501 rows, 2022–2023 |
| 2021 | 3,542 rows, 2007-12-07 to 2021-12-31 | 251 rows, 2022 | 250 rows, 2023 |

Because the one-day target shift is applied inside each file and the three splits are stored as
separate files, no label can reach across a split boundary: the last usable training target is the
final training day itself. At a one-day forecast horizon, no purging or embargo period is therefore
required.

For the 2020 cohort, the resulting pooled training tensor comprises 50 sequences of 3,290 timesteps
over 6 features, or 164,500 observations; validation and test comprise 12,600 and 25,050
observations respectively. Training proceeds by shuffling the order of the 50 sequences at each
epoch, performing backpropagation through time across each complete sequence, and applying one
weight update per sequence. Validation error is computed across all 50 validation sequences at the
end of each epoch and the best-performing parameters retained.

Inputs are normalized by mean-centred maximum scaling, *(x − μ) / (max − μ)*, with μ and max computed
once over the pooled data at load time. The transform is bounded, mapping the largest observation to
exactly 1.0, and is applied per feature across all 50 stocks jointly rather than per stock. The
resulting statistics are stored inside the evolved model and re-applied unchanged at inference, so
the test period never contributes to its own scaling. Every model in the comparison uses this
identical transform; because it is a single affine map per feature, rank correlation is invariant to
it, while squared-error magnitudes are not, which is why matching it across models matters.

We use all available history for each stock. Listing dates vary, with the earliest series beginning
in January 1990 and the latest in December 2007; 44 of the 50 stocks begin before 2000, and all end
in December 2023. Aligning the stocks to a common calendar, which the cross-sectional evaluation
requires, truncates every series to the latest of these start dates.

Each file retains twelve further columns beyond the six predictors. Five are identifiers, and the
remaining seven — including price, shares outstanding, market capitalization, and the quoted bid and
ask — support the downstream trading evaluation, where the one-way transaction cost is taken as the
half-spread (*Ask* − *Bid*)/2.

## 4.4 Data Preparation and Validation

The cohorts are generated from the CRSP extract by a deterministic pipeline that reassembles each
stock's full chronological series, partitions it by calendar year, and aligns the resulting files to
a shared trading calendar. Each stage asserts its own preconditions and aborts on violation rather
than emitting a file that would silently corrupt a downstream experiment.

The calendar of each stock is required to be strictly increasing. This check identified nine stocks
containing duplicated dates. Inspection showed the duplicates to be adjacent and byte-identical
across every column except Volume Change, where the second copy carried a value of zero introduced by
the duplication itself. Duplicate removal therefore keeps the first occurrence, but is guarded: any
duplicate pair not matching this pattern aborts the build. The pipeline further verifies that each
validation and test year contains between 240 and 260 rows, consistent with a full trading year; that
column headers agree across all three splits; and that the splits are provably non-overlapping. After
alignment, the trading calendars of all 50 stocks are compared element-wise and required to be
identical. Data rows are copied verbatim rather than parsed and re-serialized, so no rounding is
introduced.

## 4.5 Limitations

Several properties of this dataset bear on the interpretation of our results and are stated here
rather than left implicit.

**Survivorship.** Every series terminates on the same date, 2023-12-29, because the universe was
selected among firms surviving to the end of the sample. Any constituent delisted, acquired, or
bankrupted during the period is absent by construction. For comparisons of forecast accuracy this is
a modest bias, since all models are evaluated on an identical universe and the comparison remains
internally valid. For the trading results it is material: returns are earned on a portfolio of known
survivors and should not be read as an achievable live strategy.

**Normalization scope.** The scaling statistics described in Section 4.3 are computed over the
training and validation periods jointly. The test period is excluded and is transformed using the
stored statistics, so held-out results are unaffected; validation-based model selection, however, is
mildly optimistic. A second caveat follows from the form of the transform: because the denominator is
*max − μ*, the scale of each feature is set by a single extreme observation, and the mapping is
therefore more sensitive to outliers than a variance-based standardization would be.

**Index membership.** The universe is a fixed list of 50 tickers. Point-in-time S&P 500 constituency
is not tracked, so a stock's entry into or exit from the index during the sample is not reflected.

**Cross-sectional power.** With 50 stocks per date, the daily cross-sectional rank correlation is
dominated by sampling noise, and the minimum detectable effect on this universe is considerably
larger than the effect sizes typically of interest in this literature. Cross-sectional statistics
should accordingly be read as point estimates rather than as tests.

**Cost of calendar alignment.** Truncating all stocks to a common start date reduces the pooled
training set from 328,919 observations, a median of 6,900 per stock, to 164,500, or 3,290 per stock.
Alignment is required only for cross-sectional evaluation, which joins predictions by date after the
fact, and not for training itself.

**Provenance.** The CRSP extract is carried as a set of pre-split files without an accompanying
manifest, checksum, or record of the originating query. The derived cohorts are reproducible from the
extract deterministically, but the extract itself cannot at present be independently re-derived.

---

<sup>1</sup> A mid-cap company is a company with a market capitalization between $2 billion and $10
billion.
