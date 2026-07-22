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

// True Spearman rank-IC (hard ranks, average tie ranks), averaged over dates.
// Non-differentiable; for validation/fitness/reporting only.
double spearman_ic_hard(const vector<vector<double> >& preds, const vector<vector<double> >& targets);

#endif
