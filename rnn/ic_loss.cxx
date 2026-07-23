#include "ic_loss.hxx"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
using std::string;
using std::vector;

// Defaults (see header). Kept as globals so a CLI flag / test can override them.
double IC_SOFTRANK_TAU = 0.1;
double IC_VARIANCE_FLOOR = 1e-12;
double IC_SPREAD_FLOOR = 0.02;  // selection guard (reject near-constant genomes)
double IC_VAR_FLOOR = 0.1;      // tau: target per-date cross-sectional std (VICReg)

// Numerical floor inside std = sqrt(var + eps) for the variance-floor gradient.
static const double IC_VAR_EPS = 1e-8;

IcMode ic_mode_from_string(const string& s) {
    string lower;
    lower.resize(s.size());
    std::transform(s.begin(), s.end(), lower.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (lower == "spearman") {
        return IcMode::SPEARMAN;
    }
    return IcMode::PEARSON;
}

const char* ic_mode_to_string(IcMode mode) {
    switch (mode) {
        case IcMode::SPEARMAN:
            return "spearman";
        case IcMode::PEARSON:
        default:
            return "pearson";
    }
}

// Numerically stable logistic sigmoid.
static inline double stable_sigmoid(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    double e = std::exp(z);
    return e / (1.0 + e);
}

// Average-rank (1-based, ties share the mean of their positions), like
// scipy.stats.rankdata(method="average"). Used for the target ranks in the
// Spearman surrogate and for both vectors in the hard Spearman IC.
static void average_ranks(const vector<double>& v, vector<double>& ranks) {
    int32_t n = (int32_t) v.size();
    ranks.assign(n, 0.0);

    vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&v](int32_t a, int32_t b) { return v[a] < v[b]; });

    int32_t i = 0;
    while (i < n) {
        int32_t j = i;
        while (j + 1 < n && v[order[j + 1]] == v[order[i]]) {
            j++;
        }
        // positions i..j (0-based) tie -> average of 1-based ranks (i+1 .. j+1)
        double avg_rank = ((double) (i + 1) + (double) (j + 1)) / 2.0;
        for (int32_t k = i; k <= j; k++) {
            ranks[order[k]] = avg_rank;
        }
        i = j + 1;
    }
}

// Pearson correlation of (u, v) with, optionally, the gradient wrt the FIRST
// argument u (v treated as constant). Returns the correlation, or 0 for a
// degenerate (near-constant) u or v (in which case d_u, if requested, is zeroed).
//
//   a = u - mean(u), b = v - mean(v)
//   Saa = sum a^2, Sbb = sum b^2, Sab = sum a*b
//   corr = Sab / (sqrt(Saa) sqrt(Sbb))
//   dcorr/du_i = ( b_i - (Sab/Saa) a_i ) / (sqrt(Saa) sqrt(Sbb))
//
// The mean-centering cross-terms cancel because b is centered (sum b = 0), which
// is why this is O(N) rather than O(N^2). Verified vs finite-difference to ~2e-10.
static double pearson_grad_wrt_first(
    const vector<double>& u, const vector<double>& v, double variance_floor, vector<double>* d_u
) {
    int32_t n = (int32_t) u.size();
    if (d_u != nullptr) {
        d_u->assign(n, 0.0);
    }
    if (n < 2) {
        return 0.0;
    }

    double u_mean = std::accumulate(u.begin(), u.end(), 0.0) / n;
    double v_mean = std::accumulate(v.begin(), v.end(), 0.0) / n;

    double Saa = 0.0, Sbb = 0.0, Sab = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double a = u[i] - u_mean;
        double b = v[i] - v_mean;
        Saa += a * a;
        Sbb += b * b;
        Sab += a * b;
    }

    if (Saa < variance_floor || Sbb < variance_floor) {
        return 0.0;  // degenerate: undefined correlation, zero gradient
    }

    double denom = std::sqrt(Saa) * std::sqrt(Sbb);
    double corr = Sab / denom;

    if (d_u != nullptr) {
        double sab_over_saa = Sab / Saa;
        for (int32_t i = 0; i < n; i++) {
            double a = u[i] - u_mean;
            double b = v[i] - v_mean;
            (*d_u)[i] = (b - sab_over_saa * a) / denom;
        }
    }

    return corr;
}

// Soft ranks: sr_i = 1 + sum_{k != i} sigmoid((p_k - p_i)/tau). As tau -> 0 this
// approaches the hard (descending-comparison) rank; larger p -> larger soft rank.
static void soft_ranks(const vector<double>& p, double tau, vector<double>& sr) {
    int32_t n = (int32_t) p.size();
    sr.assign(n, 1.0);
    for (int32_t i = 0; i < n; i++) {
        double s = 0.0;
        for (int32_t k = 0; k < n; k++) {
            if (k == i) {
                continue;
            }
            s += stable_sigmoid((p[k] - p[i]) / tau);
        }
        sr[i] += s;
    }
}

// Differentiable IC for a single date. p = predictions, y = targets (both length
// N). Returns IC_j; if d_p != nullptr, fills dIC_j/dp.
static double ic_single_date(
    const vector<double>& p, const vector<double>& y, IcMode mode, double variance_floor, double tau,
    vector<double>* d_p
) {
    int32_t n = (int32_t) p.size();
    if (d_p != nullptr) {
        d_p->assign(n, 0.0);
    }
    if (n < 2) {
        return 0.0;
    }

    if (mode == IcMode::PEARSON) {
        return pearson_grad_wrt_first(p, y, variance_floor, d_p);
    }

    // SPEARMAN: Pearson( softrank(p), rank(y) ), gradient chained through softrank.
    vector<double> sr;
    soft_ranks(p, tau, sr);

    vector<double> yrank;
    average_ranks(y, yrank);  // constant wrt p

    vector<double>* dP_ptr = (d_p != nullptr) ? new vector<double>() : nullptr;
    double ic = pearson_grad_wrt_first(sr, yrank, variance_floor, dP_ptr);

    if (d_p != nullptr) {
        const vector<double>& dP = *dP_ptr;  // dPearson/dsr_m
        // dIC/dp_i = sum_m dP[m] * dsr_m/dp_i
        //   dsr_m/dp_i (i != m) = sigmoid'((p_i - p_m)/tau) * (1/tau)
        //   dsr_m/dp_m         = sum_{k != m} sigmoid'((p_k - p_m)/tau) * (-1/tau)
        for (int32_t i = 0; i < n; i++) {
            double gi = 0.0;
            for (int32_t m = 0; m < n; m++) {
                if (m == i) {
                    double self = 0.0;
                    for (int32_t k = 0; k < n; k++) {
                        if (k == m) {
                            continue;
                        }
                        double sg = stable_sigmoid((p[k] - p[m]) / tau);
                        self += sg * (1.0 - sg) * (-1.0 / tau);
                    }
                    gi += dP[m] * self;
                } else {
                    double sg = stable_sigmoid((p[i] - p[m]) / tau);
                    gi += dP[m] * (sg * (1.0 - sg) * (1.0 / tau));
                }
            }
            (*d_p)[i] = gi;
        }
        delete dP_ptr;
    }

    return ic;
}

// Gather column j (one date) across all stocks into p (predictions) and y (targets).
static void gather_date(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, int32_t j, vector<double>& p,
    vector<double>& y
) {
    int32_t n_stocks = (int32_t) preds.size();
    p.resize(n_stocks);
    y.resize(n_stocks);
    for (int32_t i = 0; i < n_stocks; i++) {
        p[i] = preds[i][j];
        y[i] = targets[i][j];
    }
}

double cross_sectional_ic(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode
) {
    if (preds.empty() || preds[0].empty()) {
        return 0.0;
    }
    int32_t n_dates = (int32_t) preds[0].size();

    vector<double> p, y;
    double ic_sum = 0.0;
    for (int32_t j = 0; j < n_dates; j++) {
        gather_date(preds, targets, j, p, y);
        ic_sum += ic_single_date(p, y, mode, IC_VARIANCE_FLOOR, IC_SOFTRANK_TAU, nullptr);
    }
    return ic_sum / n_dates;
}

void cross_sectional_ic_gradient(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode, double& loss,
    vector<vector<double> >& d_preds
) {
    int32_t n_stocks = (int32_t) preds.size();
    d_preds.assign(n_stocks, vector<double>(n_stocks > 0 ? preds[0].size() : 0, 0.0));

    if (preds.empty() || preds[0].empty()) {
        loss = 0.0;
        return;
    }
    int32_t n_dates = (int32_t) preds[0].size();

    vector<double> p, y, d_p;
    double ic_sum = 0.0;
    double inv_dates = 1.0 / (double) n_dates;

    for (int32_t j = 0; j < n_dates; j++) {
        gather_date(preds, targets, j, p, y);
        double ic_j = ic_single_date(p, y, mode, IC_VARIANCE_FLOOR, IC_SOFTRANK_TAU, &d_p);
        ic_sum += ic_j;

        // loss = -mean_j IC_j  ->  dloss/dp_ij = -(1/D) * dIC_j/dp_ij
        for (int32_t i = 0; i < n_stocks; i++) {
            d_preds[i][j] = -inv_dates * d_p[i];
        }
    }

    loss = -(ic_sum * inv_dates);
}

double cross_sectional_mse(const vector<vector<double> >& preds, const vector<vector<double> >& targets) {
    if (preds.empty() || preds[0].empty()) {
        return 0.0;
    }
    int32_t n_stocks = (int32_t) preds.size();
    int32_t n_dates = (int32_t) preds[0].size();
    double sse = 0.0;
    for (int32_t i = 0; i < n_stocks; i++) {
        for (int32_t j = 0; j < n_dates; j++) {
            double e = preds[i][j] - targets[i][j];
            sse += e * e;
        }
    }
    return sse / ((double) n_stocks * (double) n_dates);
}

double cross_sectional_variance_penalty(const vector<vector<double> >& preds) {
    if (preds.empty() || preds[0].empty()) {
        return 0.0;
    }
    int32_t n_stocks = (int32_t) preds.size();
    int32_t n_dates = (int32_t) preds[0].size();
    double sum = 0.0;
    for (int32_t j = 0; j < n_dates; j++) {
        double mean = 0.0;
        for (int32_t i = 0; i < n_stocks; i++) {
            mean += preds[i][j];
        }
        mean /= n_stocks;
        double var = 0.0;
        for (int32_t i = 0; i < n_stocks; i++) {
            double d = preds[i][j] - mean;
            var += d * d;
        }
        var /= n_stocks;
        double std_j = std::sqrt(var + IC_VAR_EPS);
        double hinge = IC_VAR_FLOOR - std_j;
        if (hinge > 0.0) {
            sum += hinge * hinge;  // squared hinge -> smooth (C^1) for the gradient check
        }
    }
    return sum / n_dates;
}

double cross_sectional_objective(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode, double lambda
) {
    // L = -mean_j IC_j + lambda * variance_penalty
    return -cross_sectional_ic(preds, targets, mode) + lambda * cross_sectional_variance_penalty(preds);
}

void cross_sectional_objective_gradient(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode, double lambda,
    double& loss, double& mean_ic, double& var_penalty, vector<vector<double> >& d_preds
) {
    // d(-IC)/dp via the finite-diff-verified IC gradient (also (re)sizes d_preds).
    double ic_loss = 0.0;
    cross_sectional_ic_gradient(preds, targets, mode, ic_loss, d_preds);
    mean_ic = -ic_loss;  // cross_sectional_ic_gradient's loss is -mean_ic

    var_penalty = cross_sectional_variance_penalty(preds);

    if (lambda != 0.0 && !preds.empty() && !preds[0].empty()) {
        int32_t n_stocks = (int32_t) preds.size();
        int32_t n_dates = (int32_t) preds[0].size();
        double inv_dates = 1.0 / (double) n_dates;
        // penalty_j = max(0, tau - std_j)^2,  std_j = sqrt(var_j + eps)
        // d(penalty_j)/dp_ij = -2*(tau - std_j) * (p_ij - mean_j)/(N*std_j)   for std_j < tau
        // d(L_var)/dp_ij     = (1/D) * d(penalty_j)/dp_ij
        for (int32_t j = 0; j < n_dates; j++) {
            double mean = 0.0;
            for (int32_t i = 0; i < n_stocks; i++) {
                mean += preds[i][j];
            }
            mean /= n_stocks;
            double var = 0.0;
            for (int32_t i = 0; i < n_stocks; i++) {
                double d = preds[i][j] - mean;
                var += d * d;
            }
            var /= n_stocks;
            double std_j = std::sqrt(var + IC_VAR_EPS);
            double hinge = IC_VAR_FLOOR - std_j;
            if (hinge <= 0.0) {
                continue;  // this date already meets the spread floor -> no penalty gradient
            }
            double coef = lambda * inv_dates * (-2.0) * hinge / ((double) n_stocks * std_j);
            for (int32_t i = 0; i < n_stocks; i++) {
                d_preds[i][j] += coef * (preds[i][j] - mean);
            }
        }
    }

    loss = ic_loss + lambda * var_penalty;  // = -mean_ic + lambda*var_penalty
}

double cross_sectional_spread(const vector<vector<double> >& preds) {
    if (preds.empty() || preds[0].empty()) {
        return 0.0;
    }
    int32_t n_stocks = (int32_t) preds.size();
    int32_t n_dates = (int32_t) preds[0].size();
    double spread_sum = 0.0;
    for (int32_t j = 0; j < n_dates; j++) {
        double mean = 0.0;
        for (int32_t i = 0; i < n_stocks; i++) {
            mean += preds[i][j];
        }
        mean /= n_stocks;
        double var = 0.0;
        for (int32_t i = 0; i < n_stocks; i++) {
            double d = preds[i][j] - mean;
            var += d * d;
        }
        spread_sum += std::sqrt(var / n_stocks);
    }
    return spread_sum / n_dates;
}

void spearman_ic_stats(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, double& mean_ic, double& std_ic,
    int32_t& n_used
) {
    mean_ic = 0.0;
    std_ic = 0.0;
    n_used = 0;
    if (preds.empty() || preds[0].empty()) {
        return;
    }
    int32_t n_dates = (int32_t) preds[0].size();

    vector<double> p, y, pr, yr;
    vector<double> ics;
    ics.reserve(n_dates);
    double ic_sum = 0.0;
    for (int32_t j = 0; j < n_dates; j++) {
        gather_date(preds, targets, j, p, y);
        average_ranks(p, pr);
        average_ranks(y, yr);
        // Pearson of the two rank vectors (no gradient); degenerate dates contribute 0.
        double ic_j = pearson_grad_wrt_first(pr, yr, IC_VARIANCE_FLOOR, nullptr);
        ics.push_back(ic_j);
        ic_sum += ic_j;
    }
    n_used = (int32_t) ics.size();
    if (n_used == 0) {
        return;
    }
    mean_ic = ic_sum / n_used;
    if (n_used > 1) {
        double ss = 0.0;
        for (double v : ics) {
            double d = v - mean_ic;
            ss += d * d;
        }
        std_ic = std::sqrt(ss / (n_used - 1));  // Bessel-corrected
    }
}

double spearman_ic_hard(const vector<vector<double> >& preds, const vector<vector<double> >& targets) {
    double mean_ic, std_ic;
    int32_t n_used;
    spearman_ic_stats(preds, targets, mean_ic, std_ic, n_used);
    return mean_ic;
}

double icir_from(double mean_ic, double std_ic, int32_t n) {
    if (n < 2 || std_ic < 1e-6) {
        return 0.0;
    }
    return mean_ic / std_ic * std::sqrt((double) n);
}
