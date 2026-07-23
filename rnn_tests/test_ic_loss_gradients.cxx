// Finite-difference gradient check for the cross-sectional IC loss.
//
// This is the correctness GATE for rnn/ic_loss.cxx: it verifies that the
// analytic gradient produced by cross_sectional_ic_gradient() matches a central
// finite-difference of the loss for BOTH ic_modes (pearson, spearman), over a
// grid of random universes/dates. Exits non-zero if any case exceeds tolerance.
//
// Build: linked directly against rnn/ic_loss.cxx (no other deps).
// Run:   ./rnn_tests/test_ic_loss_gradients

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using std::vector;

#include "rnn/ic_loss.hxx"

// The training objective L = -mean_j IC_j + lambda*MSE (lambda=0 -> pure -IC).
static double loss_of(const vector<vector<double> >& preds, const vector<vector<double> >& targets, IcMode mode,
                      double lambda) {
    return cross_sectional_objective(preds, targets, mode, lambda);
}

// Returns the max abs difference between analytic and finite-difference gradient
// of the objective over all (stock, date) entries, for one random problem instance.
static double check_instance(int32_t n_stocks, int32_t n_dates, IcMode mode, double lambda, std::mt19937& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);

    vector<vector<double> > preds(n_stocks, vector<double>(n_dates));
    vector<vector<double> > targets(n_stocks, vector<double>(n_dates));
    for (int32_t i = 0; i < n_stocks; i++) {
        for (int32_t j = 0; j < n_dates; j++) {
            preds[i][j] = gauss(rng);
            targets[i][j] = gauss(rng);
        }
    }

    double loss, mean_ic, mse;
    vector<vector<double> > d_analytic;
    cross_sectional_objective_gradient(preds, targets, mode, lambda, loss, mean_ic, mse, d_analytic);

    const double eps = 1e-6;
    double max_err = 0.0;
    for (int32_t i = 0; i < n_stocks; i++) {
        for (int32_t j = 0; j < n_dates; j++) {
            double save = preds[i][j];

            preds[i][j] = save + eps;
            double lp = loss_of(preds, targets, mode, lambda);

            preds[i][j] = save - eps;
            double lm = loss_of(preds, targets, mode, lambda);

            preds[i][j] = save;

            double fd = (lp - lm) / (2.0 * eps);
            double err = std::fabs(fd - d_analytic[i][j]);
            if (err > max_err) {
                max_err = err;
            }
        }
    }
    return max_err;
}

int main() {
    std::mt19937 rng(12345);

    const double TOL = 1e-6;

    // Test data is ~N(0,1) so per-date std ~ 1. Raise the variance-floor target well
    // above that so the anti-collapse penalty (and its gradient) is ACTIVE and gets
    // exercised by the finite-difference check for lambda > 0.
    IC_VAR_FLOOR = 2.0;

    struct {
        const char* name;
        IcMode mode;
    } modes[] = {{"pearson", IcMode::PEARSON}, {"spearman", IcMode::SPEARMAN}};

    // A spread of universe sizes / horizons, including small N where ties/edge
    // effects are most likely to bite.
    int32_t shapes[][2] = {{3, 1}, {5, 3}, {10, 4}, {25, 6}, {50, 8}, {2, 5}};
    int32_t n_shapes = (int32_t) (sizeof(shapes) / sizeof(shapes[0]));

    // lambda = 0 -> pure -IC (original); lambda > 0 -> anti-collapse -IC + lambda*MSE.
    double lambdas[] = {0.0, 0.5, 1.0};
    int32_t n_lambdas = (int32_t) (sizeof(lambdas) / sizeof(lambdas[0]));

    bool failed = false;
    for (auto& m : modes) {
        for (int32_t l = 0; l < n_lambdas; l++) {
            double worst = 0.0;
            for (int32_t s = 0; s < n_shapes; s++) {
                for (int32_t rep = 0; rep < 30; rep++) {
                    double err = check_instance(shapes[s][0], shapes[s][1], m.mode, lambdas[l], rng);
                    if (err > worst) {
                        worst = err;
                    }
                }
            }
            printf("ic_mode=%-9s lambda=%.1f  max |analytic - finite-diff| = %.3e  (tol %.0e)  -> %s\n", m.name,
                   lambdas[l], worst, TOL, worst <= TOL ? "PASS" : "FAIL");
            if (worst > TOL) {
                failed = true;
            }
        }
    }

    if (failed) {
        printf("GRADIENT CHECK FAILED\n");
        return 1;
    }
    printf("ALL IC GRADIENT CHECKS PASSED\n");
    return 0;
}
