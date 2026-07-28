/**
 * Emit a fixed-architecture RNN genome to a .bin, for use as EXAMM's SEED via --genome_bin.
 *
 * WHY THIS EXISTS
 * ---------------
 * EXAMM normally starts from a minimal genome -- create_ff(inputs, 0, 0, outputs, 0) in
 * get_seed_genome (rnn/generate_nn.cxx:233) -- and grows by single-node mutations. Measured on the
 * CSI300 sequence venue, that growth rate is ~1 hidden unit per ~900 genomes (the growth-biased
 * arm reached 10 hidden units / 192 weights in 9,265 genomes even with a 3:1 add/disable bias).
 * Reaching GRU-2x32-level capacity that way needs O(10^5) genomes and GRU-2x64-level O(10^6),
 * which is not reachable in any budget we have -- and selection actively fights growth, since a
 * larger genome is undertrained at fixed bp_iterations and loses to a smaller one.
 *
 * So instead of asking "can evolution CLIMB to scale?" (a compute wall), this tool lets us ask
 * "GIVEN scale, can evolution improve on the hand-designed architecture?" -- seed the population at
 * GRU 2xN and let EXAMM prune and rewire. That is the question that actually discriminates: EXAMM
 * currently sits ON the GRU capacity curve (four configs, residuals within 15bp), so the finding
 * either way is real -- it stays on the curve at scale, or it bends above it.
 *
 * train_rnn.cxx already calls the same create_* generators, but cannot serve this purpose: it
 * hardcodes the hidden width to number_inputs (train_rnn.cxx:120), so with 6 input fields you get
 * 2x6 and never 2x64, and it never calls write_to_file, so it cannot emit a seed at all.
 *
 * THE RECURRENT-DEPTH TRAP
 * ------------------------
 * create_nn adds a recurrent edge at EVERY depth 1..max_recurrent_depth for EVERY connection
 * (generate_nn.cxx:142). Edge count is therefore O(connections x depth), not O(connections):
 * a 2x64 net at depth 45 generates ~200,000 recurrent edges and a genome nobody can train.
 * Depth 1 is the right default -- it matches what a standard GRU/LSTM baseline does, which is the
 * architecture we are trying to seed. Raising it is a deliberate multi-scale choice, and this tool
 * refuses implausibly large genomes unless --allow_huge is passed.
 *
 * Usage:
 *   ./build/rnn_examples/make_seed_genome \
 *       --rnn_type gru --num_hidden_layers 2 --num_hidden_nodes 64 --max_recurrent_depth 1 \
 *       --input_parameter_names RET OPEN_C HIGH_C LOW_C VWAP_C VOLR \
 *       --output_parameter_names LABEL_CSRANK \
 *       --genome_bin_out seeds/gru_2x64.bin
 *
 * Then seed EXAMM with it (transfer_learning_version is REQUIRED by get_seed_genome:221):
 *   examm_mpi --genome_bin seeds/gru_2x64.bin --transfer_learning_version v1 ...
 */
#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"

// Anchored to TRAINABILITY, not aesthetics. Measured on the CSI300 seq venue: ~48 CPU-s per genome
// at 192 weights with bp_iterations=30, scaling roughly linearly in weight count. At 50,000 weights
// that is ~3.5 CPU-hours per genome, so a 12h x 31-worker job affords only ~100 genomes -- fewer
// than one island population, i.e. no search at all. Above this a seed is not a slow experiment,
// it is a non-experiment, and the usual cause is max_recurrent_depth (a 2x64 net at depth 45 is
// 210,183 weights vs 10,247 at depth 1).
#define HUGE_WEIGHT_THRESHOLD 50000

vector<string> arguments;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string rnn_type;
    get_argument(arguments, "--rnn_type", true, rnn_type);

    int32_t number_hidden_layers;
    get_argument(arguments, "--num_hidden_layers", true, number_hidden_layers);

    int32_t number_hidden_nodes;
    get_argument(arguments, "--num_hidden_nodes", true, number_hidden_nodes);

    // Default 1, NOT the 45 used by the mutation-driven arms: see the recurrent-depth trap above.
    int32_t max_recurrent_depth = 1;
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);

    vector<string> input_parameter_names;
    get_argument_vector(arguments, "--input_parameter_names", true, input_parameter_names);

    vector<string> output_parameter_names;
    get_argument_vector(arguments, "--output_parameter_names", true, output_parameter_names);

    string genome_bin_out;
    get_argument(arguments, "--genome_bin_out", true, genome_bin_out);

    bool allow_huge = argument_exists(arguments, "--allow_huge");

    if (number_hidden_layers < 1 || number_hidden_nodes < 1) {
        Log::fatal("ERROR: --num_hidden_layers and --num_hidden_nodes must both be >= 1\n");
        exit(1);
    }
    if (max_recurrent_depth < 1) {
        Log::fatal("ERROR: --max_recurrent_depth must be >= 1\n");
        exit(1);
    }

    WeightRules* weight_rules = new WeightRules(arguments);

    RNN_Genome* genome = NULL;
    if (rnn_type == "gru") {
        genome = create_gru(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else if (rnn_type == "lstm") {
        genome = create_lstm(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else if (rnn_type == "ugrnn") {
        genome = create_ugrnn(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else if (rnn_type == "mgu") {
        genome = create_mgu(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else if (rnn_type == "delta") {
        genome = create_delta(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else if (rnn_type == "elman") {
        genome = create_elman(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else if (rnn_type == "jordan") {
        genome = create_jordan(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else if (rnn_type == "ff") {
        genome = create_ff(
            input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names,
            max_recurrent_depth, weight_rules
        );
    } else {
        Log::fatal("ERROR: unknown --rnn_type '%s'\n", rnn_type.c_str());
        Log::fatal("Possibilities are: gru, lstm, ugrnn, mgu, delta, elman, jordan, ff\n");
        exit(1);
    }

    genome->initialize_randomly();

    // ---- normalization bounds: REQUIRED for the pretrain path ---------------------------------
    // A genome emitted without bounds has normalize_type "" -- and finetune_rnn.cxx:78 then applies
    // NO normalization, silently training on RAW features. EXAMM meanwhile overwrites the bounds
    // from its own TimeSeriesSets (generate_nn.cxx:215) and trains on NORMALIZED data, so those
    // pretrained weights would land in the wrong input space and be meaningless -- with nothing
    // failing loudly. Storing the same bounds EXAMM will compute makes its overwrite a no-op and
    // keeps pretrain / evolve / evaluate in one space.
    if (argument_exists(arguments, "--training_filenames")) {
        TimeSeriesSets* tss = TimeSeriesSets::generate_from_arguments(arguments);
        genome->set_normalize_bounds(
            tss->get_normalize_type(), tss->get_normalize_mins(), tss->get_normalize_maxs(),
            tss->get_normalize_avgs(), tss->get_normalize_std_devs()
        );
        Log::info("stored normalize bounds from training data, type '%s'\n", tss->get_normalize_type().c_str());
    } else {
        Log::warning("NO --training_filenames given: this genome carries NO normalization bounds.\n");
        Log::warning("It is fine for sizing an architecture, but MUST NOT be passed to finetune_rnn --\n");
        Log::warning("that would train it on raw, unnormalized features. Re-emit with\n");
        Log::warning("--training_filenames <...> --normalize avg_std_dev before pretraining.\n");
    }

    int32_t nodes = genome->get_enabled_node_count();
    int32_t edges = genome->get_enabled_edge_count();
    int32_t rec_edges = genome->get_enabled_recurrent_edge_count();
    int32_t weights = genome->get_number_weights();

    Log::info(
        "SEED GENOME: type %s, %dx%d hidden, rec depth %d -> nodes: %d, edges: %d, rec: %d, weights: %d\n",
        rnn_type.c_str(), number_hidden_layers, number_hidden_nodes, max_recurrent_depth, nodes, edges, rec_edges,
        weights
    );

    if (weights > HUGE_WEIGHT_THRESHOLD && !allow_huge) {
        Log::fatal(
            "ERROR: this genome has %d weights, above the %d guard. Recurrent edges scale as\n"
            "connections x max_recurrent_depth (generate_nn.cxx:142), so a large --max_recurrent_depth\n"
            "is the usual cause -- depth %d is in use here. Lower it (1 matches a standard GRU/LSTM),\n"
            "or pass --allow_huge if this size is genuinely intended.\n",
            weights, HUGE_WEIGHT_THRESHOLD, max_recurrent_depth
        );
        exit(1);
    }

    genome->write_to_file(genome_bin_out);
    Log::info("wrote seed genome to %s\n", genome_bin_out.c_str());
    Log::info(
        "seed EXAMM with:  --genome_bin %s --transfer_learning_version v1\n"
        "(transfer_learning_version is REQUIRED -- get_seed_genome reads it with required=true)\n",
        genome_bin_out.c_str()
    );

    Log::release_id("main");

    return 0;
}
