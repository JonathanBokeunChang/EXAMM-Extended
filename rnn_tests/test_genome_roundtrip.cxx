// Genome round-trip invariant test.
//
// INVARIANT: a genome's predictions (computed from get_best_parameters() +
// set_weights, exactly as evaluate_rnn does) MUST be elementwise-identical after
// each of these round-trips, in isolation:
//   inv_copy  : genome->copy()
//   inv_file  : write_to_file -> RNN_Genome(filename)
//   inv_array : write_to_array -> RNN_Genome(char*, length)   (the MPI transfer path)
//
// If any invariant is violated, a saved/transferred genome does not reproduce the
// network that was trained -- which silently invalidates fitness and every saved
// global-best genome. This test is the permanent guard against that: it exercises
// EVERY node type (incl. gated cells + recurrent edges, >16 nodes so the sort path
// is non-trivial) and can also load an existing .bin (e.g. an Anvil genome).
//
// Build: linked against the NN library + gradient_test.cxx (for the RNG helpers).
// Run:   ./rnn_tests/test_genome_roundtrip                 # synthetic all-node-type
//        ./rnn_tests/test_genome_roundtrip <genome.bin>    # + test a real genome

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
using std::string;
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "gradient_test.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"

// Forward-pass a genome using its stored best_parameters and flatten all output
// nodes' values across time -- the exact path evaluate_rnn uses to score a genome.
static vector<double> predict(RNN_Genome* g, const vector<vector<double> >& input) {
    vector<double> params = g->get_best_parameters();
    RNN* rnn = g->get_rnn();
    rnn->set_weights(params);
    rnn->forward_pass(input, false, false, 0.0);

    vector<double> out;
    int32_t n_out = rnn->get_number_output_nodes();
    int32_t series_len = (int32_t) input[0].size();
    for (int32_t t = 0; t < series_len; t++) {
        for (int32_t i = 0; i < n_out; i++) {
            out.push_back(rnn->get_output_node(i)->output_values[t]);
        }
    }
    delete rnn;
    return out;
}

static double max_abs_diff(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) {
        return 1e300;  // size mismatch is a hard failure
    }
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::fabs(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

// Runs the three round-trips on `genome` and prints/accumulates results.
// `genome` is consumed (deleted). Returns true if ALL invariants held exactly.
static bool check_genome(const string& name, RNN_Genome* genome, const vector<vector<double> >& input) {
    // Assign randomized best_parameters so no weight is coincidentally at a default
    // that would mask a lost weight.
    int32_t n = genome->get_number_weights();
    vector<double> params;
    generate_random_vector(n, params);
    genome->set_best_parameters(params);

    vector<double> p0 = predict(genome, input);

    // inv_copy
    RNN_Genome* g_copy = genome->copy();
    double d_copy = max_abs_diff(p0, predict(g_copy, input));
    delete g_copy;

    // inv_file
    string tmp = "/tmp/roundtrip_genome.bin";
    genome->write_to_file(tmp);
    RNN_Genome* g_file = new RNN_Genome(tmp);
    double d_file = max_abs_diff(p0, predict(g_file, input));
    delete g_file;

    // inv_array (MPI path)
    char* arr = nullptr;
    int32_t len = 0;
    genome->write_to_array(&arr, len);
    RNN_Genome* g_arr = new RNN_Genome(arr, len);
    double d_arr = max_abs_diff(p0, predict(g_arr, input));
    delete g_arr;
    free(arr);

    const double TOL = 0.0;  // must be BIT-exact: same params, same structure
    bool ok = (d_copy <= TOL) && (d_file <= TOL) && (d_arr <= TOL);
    printf("%-28s n_weights=%-5d  copy=%.3e  file=%.3e  array=%.3e  -> %s\n", name.c_str(), n, d_copy, d_file, d_arr,
           ok ? "PASS" : "FAIL");
    delete genome;
    return ok;
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    Log::initialize(arguments);
    Log::set_id("main");
    initialize_generator();

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    // Input/output series: 4 inputs, 2 outputs, length 12.
    const int32_t n_in = 4, n_out = 2, series_len = 12, max_rd = 3;
    vector<string> in_names, out_names;
    for (int32_t i = 0; i < n_in; i++) {
        in_names.push_back("input " + std::to_string(i));
    }
    for (int32_t i = 0; i < n_out; i++) {
        out_names.push_back("output " + std::to_string(i));
    }
    vector<vector<double> > input(n_in);
    for (int32_t i = 0; i < n_in; i++) {
        generate_random_vector(series_len, input[i]);
    }

    // 3 hidden layers x 5 nodes -> 4 + 15 + 2 = 21 nodes (>16) + recurrent edges.
    const int32_t HL = 3, HN = 5;
    bool all_ok = true;

    all_ok &= check_genome("simple(FF)", create_ff(in_names, HL, HN, out_names, max_rd, weight_rules), input);
    all_ok &= check_genome("jordan", create_jordan(in_names, HL, HN, out_names, max_rd, weight_rules), input);
    all_ok &= check_genome("elman", create_elman(in_names, HL, HN, out_names, max_rd, weight_rules), input);
    all_ok &= check_genome("ugrnn", create_ugrnn(in_names, HL, HN, out_names, max_rd, weight_rules), input);
    all_ok &= check_genome("mgu", create_mgu(in_names, HL, HN, out_names, max_rd, weight_rules), input);
    all_ok &= check_genome("gru", create_gru(in_names, HL, HN, out_names, max_rd, weight_rules), input);
    all_ok &= check_genome("delta", create_delta(in_names, HL, HN, out_names, max_rd, weight_rules), input);
    all_ok &= check_genome("lstm", create_lstm(in_names, HL, HN, out_names, max_rd, weight_rules), input);

    // Optional: test a real genome file given on the command line. It carries its
    // own best_parameters and input names, so use those.
    for (int32_t a = 1; a < argc; a++) {
        string arg = argv[a];
        if (arg.size() > 4 && arg.substr(arg.size() - 4) == ".bin") {
            RNN_Genome* loaded = new RNN_Genome(arg);
            int32_t n_loaded_in = loaded->get_number_inputs();
            vector<vector<double> > li(n_loaded_in);
            for (int32_t i = 0; i < n_loaded_in; i++) {
                generate_random_vector(series_len, li[i]);
            }
            // keep its OWN best_parameters (don't randomize a real genome)
            vector<double> p0 = predict(loaded, li);
            RNN_Genome* c = loaded->copy();
            double dc = max_abs_diff(p0, predict(c, li));
            delete c;
            loaded->write_to_file("/tmp/roundtrip_loaded.bin");
            RNN_Genome* f = new RNN_Genome(string("/tmp/roundtrip_loaded.bin"));
            double df = max_abs_diff(p0, predict(f, li));
            delete f;
            char* arr = nullptr;
            int32_t len = 0;
            loaded->write_to_array(&arr, len);
            RNN_Genome* ar = new RNN_Genome(arr, len);
            double da = max_abs_diff(p0, predict(ar, li));
            delete ar;
            free(arr);
            bool ok = (dc == 0.0) && (df == 0.0) && (da == 0.0);
            printf("%-28s copy=%.3e  file=%.3e  array=%.3e  -> %s\n", ("LOADED:" + arg).c_str(), dc, df, da,
                   ok ? "PASS" : "FAIL");
            all_ok &= ok;
            delete loaded;
        }
    }

    if (!all_ok) {
        printf("\nGENOME ROUND-TRIP INVARIANT VIOLATED\n");
        return 1;
    }
    printf("\nALL GENOME ROUND-TRIP INVARIANTS HELD\n");
    return 0;
}
