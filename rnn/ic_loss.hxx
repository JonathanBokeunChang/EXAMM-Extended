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

// True Spearman rank-IC (hard ranks, average tie ranks), averaged over dates.
// Non-differentiable; for validation/fitness/reporting only.
double spearman_ic_hard(const vector<vector<double> >& preds, const vector<vector<double> >& targets);

// Mean per-date cross-sectional prediction spread (std over stocks, averaged over
// dates). A collapse monitor: values near 0 mean the model emits near-constant output.
double cross_sectional_spread(const vector<vector<double> >& preds);

#endif
