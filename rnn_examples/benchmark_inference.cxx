// Time an evolved genome's FORWARD PASS ONLY, so EXAMM's inference cost can be compared against
// the other models in the study on identical hardware.
//
//   ./benchmark_inference --genome_file g.bin --testing_filenames X_test.csv \
//       --time_offset 1 --repeats 200
//
// WHY THIS EXISTS. The paper's only EXAMM inference figure is 9.59 us measured in C++ on a
// Raspberry Pi Zero, while every other model's is single-thread CPU PyTorch on a laptop. Those
// differ in BOTH hardware and framework, so placing them in one column would compare neither.
// This measures EXAMM on the same laptop the PyTorch numbers come from.
//
// WHAT IS TIMED, AND WHAT IS NOT. Only rnn->forward_pass(). Genome load, CSV parsing,
// normalisation and prediction writing are all outside the timed region -- the PyTorch figures
// time model(x) alone, and including I/O here would compare a whole program against a single call.
//
// PER-PREDICTION IS THE COMPARABLE UNIT, AND THE TWO FAMILIES REACH IT DIFFERENTLY. A transformer
// consumes a 96-step window and emits ONE prediction, so one window is one prediction. EXAMM runs
// recurrently over the whole series and emits a prediction at EVERY step, so a T-step pass yields
// T predictions and the per-prediction cost is (pass time / T). That asymmetry is a real property
// of the architectures for streaming inference, not an accounting trick -- but it is why this
// prints both the whole-series time and the per-prediction figure, so a reader can see which is
// being quoted.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    Log::initialize(arguments);
    Log::set_id("main");

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);
    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);
    int32_t repeats = 200;
    get_argument(arguments, "--repeats", false, repeats);

    RNN_Genome* genome = new RNN_Genome(genome_filename);
    TimeSeriesSets* tss = TimeSeriesSets::generate_test(testing_filenames, genome->get_input_parameter_names(),
                                                        genome->get_output_parameter_names());
    string norm = genome->get_normalize_type();
    if (norm.compare("min_max") == 0) {
        tss->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (norm.compare("avg_std_dev") == 0) {
        tss->normalize_avg_std_dev(genome->get_normalize_avgs(), genome->get_normalize_std_devs(),
                                   genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (norm.compare("instance") == 0) {
        map<string, double> avgs = genome->get_normalize_avgs();
        int32_t w = (avgs.count("__instance_window__") > 0) ? (int32_t) avgs["__instance_window__"] : 96;
        tss->normalize_instance(w);
    }

    vector<vector<vector<double> > > inputs, outputs;
    tss->export_test_series(time_offset, inputs, outputs);

    RNN* rnn = genome->get_rnn();
    vector<double> params = genome->get_best_parameters();
    if (params.empty()) params = genome->get_initial_parameters();
    rnn->set_weights(params);

    const vector<vector<double> >& series = inputs[0];
    int32_t T = (int32_t) series[0].size();

    // Warm up: first pass touches cold caches and any lazy allocation inside the node buffers.
    for (int32_t i = 0; i < 20; i++) rnn->forward_pass(series, false, false, 0.0);

    auto t0 = chrono::high_resolution_clock::now();
    for (int32_t i = 0; i < repeats; i++) rnn->forward_pass(series, false, false, 0.0);
    auto t1 = chrono::high_resolution_clock::now();

    double total_us = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    double per_pass = total_us / repeats;
    double per_pred = per_pass / T;

    cout << fixed << setprecision(3);
    cout << "weights          : " << genome->get_number_weights() << endl;
    cout << "series length    : " << T << " steps (" << T << " predictions per pass)" << endl;
    cout << "repeats          : " << repeats << endl;
    cout << "per whole series : " << per_pass << " us" << endl;
    cout << "per prediction   : " << per_pred << " us" << endl;

    delete rnn;
    delete genome;
    delete tss;
    return 0;
}
