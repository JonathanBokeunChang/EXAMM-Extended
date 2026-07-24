#ifndef EXAMM_IC_LOSS_HXX
#define EXAMM_IC_LOSS_HXX

// Cross-sectional Information-Coefficient (IC) loss for EXAMM.
//
// This is a *ranking* objective: at each date the model's predictions across the
// whole universe of stocks are scored against the realized targets, and the
// per-date scores are averaged over time. It replaces the pointwise MSE objective
// for return/rank forecasting, where what matters is getting the cross-sectional
// ORDER right (long the top, short the bottom), not the absolute level.
//
// Data layout everywhere in this module is [stock][date]: preds[i][j] is stock i's
// prediction on date j, targets[i][j] the realized target. All stocks must be
// calendar-aligned so that index j is the same date for every i (the caller is
// responsible for this; the pooled EXAMM path enforces equal series lengths).
//
// Two differentiable surrogates are provided (selected by IcMode):
//   PEARSON  - IC_j = Pearson correlation of (preds[:,j], targets[:,j]).
//              Closed-form gradient, O(N) per date. Default.
//   SPEARMAN - IC_j = Pearson correlation of (softrank(preds[:,j]), rank(targets[:,j])),
//              i.e. a differentiable Spearman rank correlation via soft ranks.
//              O(N^2) per date; trivial at universe sizes of tens-to-hundreds.
//
// The loss is L = -mean_j IC_j (we MAXIMIZE IC, so we minimize -IC to fit EXAMM's
// minimize-fitness convention). cross_sectional_ic_gradient fills d_preds with
// dL/dpreds so it can be injected directly into RNN::backward_pass.
//
// spearman_ic_hard() computes the TRUE (non-differentiable, hard-rank) Spearman IC
// and is what should be used for validation/fitness and reporting -- train on a
// smooth surrogate, select on the real metric.

#include <string>
#include <vector>
using std::vector;

enum class IcMode { PEARSON, SPEARMAN };

// Which cross-sectional training objective the pooled batch path optimizes:
//   IC     - L = -mean_j IC_j + ic_var_lambda * variance_floor_penalty (the ranking arm)
//   MSEVAR - L = MSE + csvar_lambda * mean_j Var_cs_j(pred) (dispersion-PENALIZED MSE;
//            fitness = validation MSE, collapse guard disabled -- shrinkage is intended)
enum class CsObjective { IC, MSEVAR };

// Parse "pearson"/"spearman" (case-insensitive). Unknown -> PEARSON.
IcMode ic_mode_from_string(const std::string& s);
const char* ic_mode_to_string(IcMode mode);

// Soft-rank temperature for SPEARMAN (smaller -> closer to hard ranks, sharper
// gradients). 0.1 matches the value the gradient check was validated at.
extern double IC_SOFTRANK_TAU;

// A date whose prediction variance is below this is treated as degenerate:
// it contributes 0 to the IC and 0 to the gradient (avoids the 0/0 blow-up when
// a genome emits constant predictions on a date).
extern double IC_VARIANCE_FLOOR;

// Collapse guard: if a genome's mean cross-sectional prediction spread (per-date std,
// averaged over dates) falls below this, it has degenerated to near-constant output
// and its (scale-invariant) IC is meaningless. The training variance-floor term keeps
// spread up; the guard is the selection-side safety net (see
// RNN_Genome::backpropagate_cross_sectional). Default well below IC_VAR_FLOOR so
// genuinely-varied genomes pass while collapsed ones are rejected.
extern double IC_SPREAD_FLOOR;

// Target per-date cross-sectional std for the anti-collapse variance floor (tau). The
// penalty pushes each date's prediction std up toward this value; a constant prediction
// (std 0) gets the maximum penalty. This is VICReg's variance term (Bardes/Ponce/LeCun,
// ICLR 2022) applied per date. IC is scale-invariant, so pinning the spread to tau does
// not constrain the achievable ranking -- it only forbids the degenerate zero-spread
// optimum. Unlike an MSE anchor it targets SPREAD (not level), which is the actual
// collapse; MSE fails here because on a near-unpredictable signal the MSE-optimal
// predictor is itself a near-constant.
extern double IC_VAR_FLOOR;

// Mean daily differentiable IC over all dates (higher is better). preds/targets
// are [stock][date] and must be rectangular (all rows the same length).
double cross_sectional_ic(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode
);

// Fills loss = -mean_j IC_j and d_preds[i][j] = dloss/dpreds[i][j].
// d_preds is (re)sized to match preds. Degenerate dates contribute zero gradient.
void cross_sectional_ic_gradient(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode, double& loss,
    vector<vector<double> >& d_preds
);

// Pointwise MSE over all (stock, date) entries: mean of (pred - target)^2.
// Kept as a diagnostic (reported by verify_fullprec_ic); NOT part of the objective.
double cross_sectional_mse(const vector<vector<double> >& preds, const vector<vector<double> >& targets);

// Anti-collapse variance-floor penalty (VICReg-style, per date):
//   (1/D) * sum_j max(0, IC_VAR_FLOOR - std_j)^2,   std_j = sqrt(var_j + eps).
// Zero once every date's cross-sectional std is >= IC_VAR_FLOOR; grows as predictions
// collapse toward a constant. Does not depend on the targets.
double cross_sectional_variance_penalty(const vector<vector<double> >& preds);

// The ANTI-COLLAPSE training objective: L = -mean_j IC_j + lambda * variance_penalty.
// The pure IC term is scale/shift-invariant, so it admits a degenerate optimum where
// predictions collapse to a near-constant (a documented failure of correlation losses;
// see Kwiatkowski & Chudziak 2025). The variance-floor term forbids the zero-spread
// solution directly. lambda = 0 recovers pure IC.
double cross_sectional_objective(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode, double lambda
);

// Fills d_preds[i][j] = d(L)/d(pred_ij) for L = -mean_j IC_j + lambda*variance_penalty,
// and returns loss = L along with its components mean_ic and var_penalty (for logging).
// Reuses the finite-diff-verified cross_sectional_ic_gradient for the IC part and adds
// the closed-form variance-floor gradient.
void cross_sectional_objective_gradient(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode, double lambda,
    double& loss, double& mean_ic, double& var_penalty, vector<vector<double> >& d_preds
);

// DISPERSION-PENALIZED MSE (the "negative dose" on the dispersion axis):
//   L = (1/(N*T)) * sum_ij (pred_ij - target_ij)^2
//       + lambda * (1/T) * sum_j Var_i(pred_ij)         (population 1/N variance)
// The returns campaign measured that objectives AMPLIFYING cross-sectional dispersion
// overfit in proportion (raw-MSE generalizes < z-score-MSE 11x < ICIR 500x val->test).
// This objective extends that dose-response curve in the negative direction by
// PENALIZING dispersion -- the exact inverse of the variance-floor above. Finance
// grounding: Grinold-Kahn forecast shrinkage (alpha = IC * vol * score) moved into
// the training objective. lambda = 0 is the pure batch-MSE control arm that isolates
// the optimizer-pathway effect (batch cross-sectional vs stochastic per-series).
double cross_sectional_msevar_objective(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, double lambda
);

// Fills d_preds[i][j] = d(L)/d(pred_ij) for the dispersion-penalized MSE and returns
// loss = L plus its components (for logging):
//   d(L)/d(pred_ij) = 2*(pred_ij - target_ij)/(N*T) + lambda * 2*(pred_ij - mean_i pred_ij)/(N*T)
void cross_sectional_msevar_gradient(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, double lambda, double& loss,
    double& mse_part, double& var_part, vector<vector<double> >& d_preds
);

// Per-date hard-rank Spearman IC statistics: fills mean_ic (the average IC that
// spearman_ic_hard returns), std_ic (Bessel/(n-1) std of the per-date ICs), and
// n_used (number of dates). Degenerate dates contribute IC 0. For validation/fitness.
void spearman_ic_stats(
    const vector<vector<double> >& preds, const vector<vector<double> >& targets, double& mean_ic, double& std_ic,
    int32_t& n_used
);

// True Spearman rank-IC (hard ranks, average tie ranks), averaged over dates.
// Thin wrapper over spearman_ic_stats. Non-differentiable; reporting/fitness only.
double spearman_ic_hard(const vector<vector<double> >& preds, const vector<vector<double> >& targets);

// IC information ratio: ICIR = mean_ic / std_ic * sqrt(n). "Sharpe ratio of IC" --
// a standard quant metric (Qlib Rank ICIR) that rewards temporally-consistent IC.
// Returns 0 if std_ic is below a small floor (degenerate; the spread collapse guard
// handles that case upstream) or n < 2.
double icir_from(double mean_ic, double std_ic, int32_t n);

// Mean per-date cross-sectional prediction spread (std over stocks, averaged over
// dates). A collapse monitor: values near 0 mean the model emits near-constant output.
double cross_sectional_spread(const vector<vector<double> >& preds);

#endif
