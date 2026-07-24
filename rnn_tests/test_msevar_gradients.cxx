// Finite-difference gradient check for the dispersion-penalized MSE objective.
//
// This is the correctness GATE for cross_sectional_msevar_gradient() in
// rnn/ic_loss.cxx (arm C of the returns loss campaign): it verifies that the
// analytic gradient matches a central finite-difference of
//   L = MSE + lambda * mean_j Var_i(pred_ij)
// over a grid of random universes/dates and lambdas, that lambda = 0 reproduces
// the plain pooled-MSE gradient 2*r/(N*T) exactly (the batch-MSE control arm),
// and that the returned loss decomposition (mse_part, var_part) is consistent
// with cross_sectional_msevar_objective(). Exits non-zero on any failure.
//
// Build: linked directly against rnn/ic_loss.cxx (no other deps).
// Run:   ./rnn_tests/test_msevar_gradients

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using std::vector;

#include "rnn/ic_loss.hxx"

// Max abs difference between analytic and central finite-difference gradient over
// all (stock, date) entries, for one random problem instance. Also cross-checks the
// loss value/decomposition against cross_sectional_msevar_objective.
static double check_instance(int32_t n_stocks, int32_t n_dates, double lambda, std::mt19937& rng, bool& consistent) {
    std::normal_distribution<double> gauss(0.0, 1.0);

    vector<vector<double> > preds(n_stocks, vector<double>(n_dates));
    vector<vector<double> > targets(n_stocks, vector<double>(n_dates));
    for (int32_t i = 0; i < n_stocks; i++) {
        for (int32_t j = 0; j < n_dates; j++) {
            preds[i][j] = gauss(rng);
            targets[i][j] = gauss(rng);
        }
    }

    double loss, mse_part, var_part;
    vector<vector<double> > d_analytic;
    cross_sectional_msevar_gradient(preds, targets, lambda, loss, mse_part, var_part, d_analytic);

    // decomposition consistency: loss == objective() and loss == mse + lambda*var
    double obj = cross_sectional_msevar_objective(preds, targets, lambda);
    if (std::fabs(loss - obj) > 1e-12 || std::fabs(loss - (mse_part + lambda * var_part)) > 1e-12) {
        consistent = false;
    }

    const double eps = 1e-6;
    double max_err = 0.0;
    for (int32_t i = 0; i < n_stocks; i++) {
        for (int32_t j = 0; j < n_dates; j++) {
            double save = preds[i][j];

            preds[i][j] = save + eps;
            double lp = cross_sectional_msevar_objective(preds, targets, lambda);

            preds[i][j] = save - eps;
            double lm = cross_sectional_msevar_objective(preds, targets, lambda);

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

// lambda = 0 must reproduce the plain pooled-MSE gradient 2*r/(N*T) EXACTLY
// (bitwise up to floating-point associativity; tolerance is effectively zero).
// This is the algebraic guarantee behind the batch-MSE control arm.
static double check_lambda_zero_is_mse(int32_t n_stocks, int32_t n_dates, std::mt19937& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);

    vector<vector<double> > preds(n_stocks, vector<double>(n_dates));
    vector<vector<double> > targets(n_stocks, vector<double>(n_dates));
    for (int32_t i = 0; i < n_stocks; i++) {
        for (int32_t j = 0; j < n_dates; j++) {
            preds[i][j] = gauss(rng);
            targets[i][j] = gauss(rng);
        }
    }

    double loss, mse_part, var_part;
    vector<vector<double> > d;
    cross_sectional_msevar_gradient(preds, targets, 0.0, loss, mse_part, var_part, d);

    double inv_nt = 1.0 / ((double) n_stocks * (double) n_dates);
    double max_err = 0.0;
    for (int32_t i = 0; i < n_stocks; i++) {
        for (int32_t j = 0; j < n_dates; j++) {
            double expected = 2.0 * (preds[i][j] - targets[i][j]) * inv_nt;
            double err = std::fabs(d[i][j] - expected);
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

    // A spread of universe sizes / horizons, including small N where edge effects bite.
    int32_t shapes[][2] = {{3, 1}, {5, 3}, {10, 4}, {25, 6}, {50, 8}, {2, 5}};
    int32_t n_shapes = (int32_t) (sizeof(shapes) / sizeof(shapes[0]));

    // lambda = 0 -> pure batch MSE (the control arm); 1.0 is the campaign dose.
    double lambdas[] = {0.0, 0.3, 1.0, 5.0};
    int32_t n_lambdas = (int32_t) (sizeof(lambdas) / sizeof(lambdas[0]));

    bool failed = false;
    bool consistent = true;

    for (int32_t l = 0; l < n_lambdas; l++) {
        double worst = 0.0;
        for (int32_t s = 0; s < n_shapes; s++) {
            for (int32_t rep = 0; rep < 30; rep++) {
                double err = check_instance(shapes[s][0], shapes[s][1], lambdas[l], rng, consistent);
                if (err > worst) {
                    worst = err;
                }
            }
        }
        printf("msevar lambda=%.1f  max |analytic - finite-diff| = %.3e  (tol %.0e)  -> %s\n", lambdas[l], worst, TOL,
               worst <= TOL ? "PASS" : "FAIL");
        if (worst > TOL) {
            failed = true;
        }
    }

    if (!consistent) {
        printf("LOSS DECOMPOSITION INCONSISTENT (loss != objective or != mse + lambda*var)\n");
        failed = true;
    }

    double worst_l0 = 0.0;
    for (int32_t s = 0; s < n_shapes; s++) {
        for (int32_t rep = 0; rep < 30; rep++) {
            double err = check_lambda_zero_is_mse(shapes[s][0], shapes[s][1], rng);
            if (err > worst_l0) {
                worst_l0 = err;
            }
        }
    }
    printf("msevar lambda=0 == pooled-MSE gradient: max err = %.3e (tol 1e-15)  -> %s\n", worst_l0,
           worst_l0 <= 1e-15 ? "PASS" : "FAIL");
    if (worst_l0 > 1e-15) {
        failed = true;
    }

    if (failed) {
        printf("MSEVAR GRADIENT CHECK FAILED\n");
        return 1;
    }
    printf("ALL MSEVAR GRADIENT CHECKS PASSED\n");
    return 0;
}
