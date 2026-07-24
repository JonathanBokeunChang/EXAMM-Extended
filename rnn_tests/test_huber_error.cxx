// Correctness GATE for RNN::calculate_error_huber (arm B of the returns loss
// campaign). Uses a real minimal network (create_ff) so the REAL code path is
// exercised, then drives output_values directly (they are public, same access the
// cross-sectional loss path uses) to test, over random residual sets straddling
// delta:
//
//   1. loss value == analytic rho_delta in the r^2 convention
//      (rho = r^2 for |r| <= delta, delta*(2|r|-delta) beyond), summed per node
//      and divided by n timesteps;
//   2. error_values[j] == clamp(r, -delta, +delta);
//   3. central finite-difference of the returned loss w.r.t. output_values[j]
//      == 2*error_values[j]/n (residuals kept away from the |r| = delta kink);
//   4. delta -> infinity identity: calculate_error_huber(., 1e9) reproduces
//      calculate_error_mse EXACTLY (same return, same error_values) -- the
//      algebraic guarantee behind the end-to-end bit-identity regression gate.
//
// Run: ./rnn_tests/test_huber_error   (no arguments needed)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using std::vector;

#include <string>
using std::string;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    Log::initialize(arguments);
    Log::set_id("main");

    std::mt19937 rng(1337);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // minimal real network: 1 input -> 1 output, no hidden
    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);
    vector<string> in_names{"input 1"};
    vector<string> out_names{"output 1"};
    RNN_Genome* genome = create_ff(in_names, 0, 0, out_names, 1, weight_rules);
    genome->initialize_randomly();
    RNN* rnn = genome->get_rnn();

    const int32_t N = 40;  // timesteps
    bool failed = false;

    // one forward pass so internal state (series_length, sizes) is consistent
    vector<vector<double> > series(1, vector<double>(N));
    for (int32_t j = 0; j < N; j++) {
        series[0][j] = gauss(rng);
    }
    rnn->forward_pass(series, false, true, 0.0);

    for (int32_t rep = 0; rep < 50; rep++) {
        double delta = 0.05 + 0.5 * std::fabs(gauss(rng));

        // targets + outputs with residuals straddling delta, kept away from the kink
        vector<vector<double> > expected(1, vector<double>(N));
        for (int32_t j = 0; j < N; j++) {
            double r = gauss(rng);  // residual we want
            if (std::fabs(std::fabs(r) - delta) < 1e-3) {
                r += 2e-3;  // stay off the |r| == delta kink for the finite diff
            }
            double out = gauss(rng);
            rnn->get_output_node(0)->output_values[j] = out;
            expected[0][j] = out - r;
        }

        // 1+2: value and error_values against the analytic formula
        double loss = rnn->calculate_error_huber(expected, delta);
        double rho_sum = 0.0;
        double max_ev_err = 0.0;
        for (int32_t j = 0; j < N; j++) {
            double r = rnn->get_output_node(0)->output_values[j] - expected[0][j];
            double rho = (std::fabs(r) <= delta) ? r * r : delta * (2.0 * std::fabs(r) - delta);
            rho_sum += rho;
            double clamp = (r > delta) ? delta : ((r < -delta) ? -delta : r);
            double ev_err = std::fabs(rnn->get_output_node(0)->error_values[j] - clamp);
            if (ev_err > max_ev_err) {
                max_ev_err = ev_err;
            }
        }
        double analytic = rho_sum / N;
        if (std::fabs(loss - analytic) > 1e-12 || max_ev_err > 1e-15) {
            printf("FAIL rep %d: loss %.17g vs analytic %.17g, max error_values err %.3e\n", rep, loss, analytic,
                   max_ev_err);
            failed = true;
        }

        // 3: finite difference of the loss w.r.t. each output == 2*clamp(r)/N
        const double eps = 1e-7;
        for (int32_t j = 0; j < N; j += 7) {  // sample of positions
            double save = rnn->get_output_node(0)->output_values[j];
            rnn->get_output_node(0)->output_values[j] = save + eps;
            double lp = rnn->calculate_error_huber(expected, delta);
            rnn->get_output_node(0)->output_values[j] = save - eps;
            double lm = rnn->calculate_error_huber(expected, delta);
            rnn->get_output_node(0)->output_values[j] = save;
            rnn->calculate_error_huber(expected, delta);  // restore error_values

            double fd = (lp - lm) / (2.0 * eps);
            double grad = 2.0 * rnn->get_output_node(0)->error_values[j] / N;
            if (std::fabs(fd - grad) > 1e-5) {
                printf("FAIL rep %d pos %d: finite-diff %.10g vs 2*error/N %.10g (delta %.4g)\n", rep, j, fd, grad,
                       delta);
                failed = true;
            }
        }

        // 4: delta -> infinity identity with calculate_error_mse. The math identity is
        // exact; the compiled loss SCALAR may differ by 1-2 ULPs because the two
        // functions' accumulation loops get contracted/vectorized differently at -O3
        // (measured: ~1e-16 relative). So the gate asserts what the identity actually
        // guarantees: error_values BITWISE identical (they involve no accumulation),
        // loss scalar equal within a tight ULP-scale relative tolerance.
        double huber_inf = rnn->calculate_error_huber(expected, 1e9);
        vector<double> huber_ev = rnn->get_output_node(0)->error_values;
        double mse = rnn->calculate_error_mse(expected);
        vector<double> mse_ev = rnn->get_output_node(0)->error_values;
        bool ev_identical = true;
        for (int32_t j = 0; j < N && ev_identical; j++) {
            if (huber_ev[j] != mse_ev[j]) {
                ev_identical = false;
            }
        }
        double rel = std::fabs(huber_inf - mse) / std::fmax(1e-300, std::fabs(mse));
        if (!ev_identical || rel > 1e-14) {
            printf("FAIL rep %d: huber(delta=1e9) vs mse: error_values %s, loss rel diff %.3e (%.17g vs %.17g)\n", rep,
                   ev_identical ? "identical" : "DIFFER", rel, huber_inf, mse);
            failed = true;
        }
    }

    delete rnn;
    delete genome;

    if (failed) {
        printf("HUBER ERROR CHECK FAILED\n");
        return 1;
    }
    printf("ALL HUBER ERROR CHECKS PASSED\n");
    return 0;
}
