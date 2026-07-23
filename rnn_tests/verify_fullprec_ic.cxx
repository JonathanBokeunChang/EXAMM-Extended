// Full-precision reproduction check: load a saved genome, rebuild the pooled
// validation set exactly as evaluate_rnn does (genome's own normalize bounds),
// and compute the cross-sectional IC via RNN_Genome::get_ic at full double
// precision -- i.e. WITHOUT the 6-significant-figure prediction CSV round-trip.
//
// If this reproduces the genome's recorded best_validation_mse (= -IC), then the
// genome is faithful and any discrepancy seen through evaluate_rnn's CSV was a
// precision artifact (a near-constant, noise-floor prediction whose ~1e-8
// cross-sectional signal does not survive CSV serialization).
//
// Usage: verify_fullprec_ic --genome_file g.bin --testing_filenames a_val.csv b_val.csv ... --time_offset 1

#include <cmath>
#include <string>
#include <vector>
using std::string;
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/ic_loss.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    Log::initialize(arguments);
    Log::set_id("main");

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome* genome = new RNN_Genome(genome_filename);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    TimeSeriesSets* tss = TimeSeriesSets::generate_test(
        testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names()
    );
    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        tss->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        tss->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    }

    vector<vector<vector<double> > > inputs, outputs;
    tss->export_test_series(time_offset, inputs, outputs);

    vector<double> best = genome->get_best_parameters();

    // Full-precision cross-sectional IC (the training-time metric).
    double ic = genome->get_ic(best, inputs, outputs);

    // Also report the cross-sectional spread of the RAW (full-precision) predictions
    // per date, to show whether the genome is at the numerical noise floor.
    vector<vector<double> > preds(inputs.size());
    vector<vector<double> > targets(inputs.size());
    {
        RNN* rnn = genome->get_rnn();
        rnn->set_weights(best);
        for (int32_t i = 0; i < (int32_t) inputs.size(); i++) {
            rnn->forward_pass(inputs[i], false, false, 0.0);
            preds[i] = rnn->get_output_node(0)->output_values;
            targets[i] = outputs[i][0];
        }
        delete rnn;
    }
    int32_t n_stocks = (int32_t) preds.size();
    int32_t n_dates = n_stocks > 0 ? (int32_t) preds[0].size() : 0;
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

    // IC information ratio (mean/std of the per-date ICs * sqrt(n)) -- cross-check
    // against eval_ensemble_ic.py's reported "IC info ratio".
    double mean_ic, std_ic;
    int32_t n_used;
    spearman_ic_stats(preds, targets, mean_ic, std_ic, n_used);
    double icir = icir_from(mean_ic, std_ic, n_used);

    printf("stocks=%d dates=%d\n", n_stocks, n_dates);
    printf("full-precision cross-sectional IC (get_ic) : %+.6f\n", ic);
    printf("  daily-IC std                             : %.6f\n", std_ic);
    printf("  IC information ratio (ICIR)              : %+.4f\n", icir);
    printf("recorded best_validation_mse (= -IC/-ICIR) : %+.6f\n", genome->get_best_validation_mse());
    printf("cross-sectional MSE (pred vs target)       : %.6f\n", cross_sectional_mse(preds, targets));
    printf("avg full-precision prediction spread/date  : %.3e%s\n", n_dates ? spread_sum / n_dates : 0.0,
           (n_dates && spread_sum / n_dates < 1e-6) ? "   <-- COLLAPSED (near-constant output)" : "");

    delete genome;
    delete tss;
    return 0;
}
