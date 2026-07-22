/*
 * finetune_rnn -- continue training an already-evolved genome to convergence.
 *
 * WHY THIS EXISTS
 * ---------------
 * During neuroevolution every candidate genome is trained for exactly `bp_iterations`
 * epochs (10 in our campaigns) and the global best is selected on validation MSE. It is
 * never trained further. The fixed-architecture baselines it is compared against (LSTM/GRU)
 * train up to 60 epochs with validation-based early stopping, i.e. they CONVERGE.
 *
 * That is a genuine asymmetry in favour of the baselines: the evolved model may simply be
 * undertrained. This tool removes it by continuing backprop on the selected genome using the
 * same protocol as the baseline -- train on the training split, keep the parameters with the
 * lowest validation MSE. Precedent: Lyu et al. (EvoStar 2025) fine-tuned the best evolved RNN
 * for 100 additional epochs at a reduced learning rate.
 *
 * BOUNDED DOWNSIDE (the property that makes this safe)
 * ----------------------------------------------------
 * RNN_Genome::backpropagate_stochastic evaluates the STARTING weights first and seeds
 * best_validation_mse / best_parameters with them, then only overwrites when a later epoch
 * strictly improves validation MSE, and finally calls set_weights(best_parameters). So the
 * evolved weights are entered as the epoch-0 checkpoint: if no epoch beats them, the genome
 * is returned unchanged. Fine-tuning can never worsen validation MSE. We assert that.
 *
 * NORMALISATION
 * -------------
 * Train and validation are loaded SEPARATELY via generate_test(), which performs no
 * normalisation of its own, and are then normalised with the bounds STORED IN THE GENOME --
 * exactly the path evaluate_rnn uses at test time. This guarantees fine-tuning happens in the
 * identical normalised space the genome will later be scored in. Do NOT pass --normalize:
 * that would make the loader normalise with freshly computed bounds and then this code would
 * normalise a second time.
 *
 * USAGE
 *   finetune_rnn --genome_file <in.bin> --output_genome <out.bin> \
 *       --training_filenames <train csvs...> --validation_filenames <val csvs...> \
 *       --time_offset 0 --finetune_epochs 100 --learning_rate 0.0005 \
 *       --output_directory <dir> --std_message_level INFO --file_message_level NONE
 */
#include <cstdlib>

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_update.hxx"

vector<string> arguments;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > validation_inputs;
vector<vector<vector<double> > > validation_outputs;

/**
 * Apply the genome's stored normalisation bounds to a freshly loaded set.
 * Mirrors evaluate_rnn.cxx so training and scoring share one normalised space.
 */
static void apply_genome_normalization(TimeSeriesSets* tss, RNN_Genome* genome) {
    string normalize_type = genome->get_normalize_type();

    if (normalize_type.compare("min_max") == 0) {
        tss->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        tss->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    } else if (normalize_type.compare("none") != 0 && normalize_type.length() > 0) {
        Log::fatal("unknown normalize type stored in genome: '%s'\n", normalize_type.c_str());
        exit(1);
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);

    string output_genome_filename;
    get_argument(arguments, "--output_genome", true, output_genome_filename);

    vector<string> training_filenames;
    get_argument_vector(arguments, "--training_filenames", true, training_filenames);

    vector<string> validation_filenames;
    get_argument_vector(arguments, "--validation_filenames", true, validation_filenames);

    int32_t time_offset = 0;
    get_argument(arguments, "--time_offset", true, time_offset);

    int32_t finetune_epochs = 100;
    get_argument(arguments, "--finetune_epochs", false, finetune_epochs);

    // Guard against the double-normalisation footgun described in the header comment.
    if (argument_exists(arguments, "--normalize")) {
        Log::fatal("do NOT pass --normalize to finetune_rnn: the data is normalised with the\n");
        Log::fatal("bounds stored in the genome, and --normalize would normalise it twice.\n");
        exit(1);
    }

    RNN_Genome* genome = new RNN_Genome(genome_filename);

    const vector<string> input_parameter_names = genome->get_input_parameter_names();
    const vector<string> output_parameter_names = genome->get_output_parameter_names();

    // ---- load TRAIN and VALIDATION separately, each in the genome's normalised space -----
    TimeSeriesSets* training_sets =
        TimeSeriesSets::generate_test(training_filenames, input_parameter_names, output_parameter_names);
    apply_genome_normalization(training_sets, genome);
    training_sets->export_test_series(time_offset, training_inputs, training_outputs);

    TimeSeriesSets* validation_sets =
        TimeSeriesSets::generate_test(validation_filenames, input_parameter_names, output_parameter_names);
    apply_genome_normalization(validation_sets, genome);
    validation_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    Log::info(
        "loaded %d training series and %d validation series, normalize type '%s'\n",
        (int32_t) training_inputs.size(), (int32_t) validation_inputs.size(),
        genome->get_normalize_type().c_str()
    );

    if (training_inputs.size() == 0 || validation_inputs.size() == 0) {
        Log::fatal("no training or validation series were exported -- check the filename lists\n");
        exit(1);
    }

    // ---- the genome must already carry trained weights ----------------------------------
    vector<double> evolved_parameters = genome->get_best_parameters();
    int32_t n_weights = genome->get_number_weights();

    if (evolved_parameters.size() == 0) {
        Log::fatal("genome '%s' has no best_parameters -- nothing to fine-tune\n", genome_filename.c_str());
        exit(1);
    }
    if ((int32_t) evolved_parameters.size() != n_weights) {
        Log::fatal(
            "genome weight-count mismatch: best_parameters has %d entries but the genome has %d weights\n",
            (int32_t) evolved_parameters.size(), n_weights
        );
        exit(1);
    }

    double validation_mse_before = genome->get_mse(evolved_parameters, validation_inputs, validation_outputs);
    Log::info("BEFORE fine-tune: validation MSE %.10lf (%d weights)\n", validation_mse_before, n_weights);

    // ---- continue from the EVOLVED weights ----------------------------------------------
    // backpropagate_stochastic starts from initial_parameters, NOT best_parameters. Without
    // this line it would restart from the genome's original random initialisation and discard
    // everything evolution found.
    genome->set_initial_parameters(evolved_parameters);
    genome->set_bp_iterations(finetune_epochs);

    WeightUpdate* weight_update_method = new WeightUpdate(arguments);

    Log::info("fine-tuning for %d epochs\n", finetune_epochs);
    genome->backpropagate_stochastic(
        training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
    );

    // ---- verify the bounded-downside property, then persist -----------------------------
    vector<double> tuned_parameters = genome->get_best_parameters();
    double validation_mse_after = genome->get_mse(tuned_parameters, validation_inputs, validation_outputs);

    Log::info(
        "AFTER  fine-tune: validation MSE %.10lf  (delta %+.10lf)\n", validation_mse_after,
        validation_mse_after - validation_mse_before
    );

    // Tolerance absorbs float re-accumulation only; a real regression means the checkpointing
    // contract in backpropagate_stochastic was violated and the result must not be trusted.
    if (validation_mse_after > validation_mse_before + 1e-9) {
        Log::fatal(
            "fine-tuning REGRESSED validation MSE (%.10lf -> %.10lf). The evolved weights should\n",
            validation_mse_before, validation_mse_after
        );
        Log::fatal("have been retained as the epoch-0 checkpoint; refusing to write the genome.\n");
        exit(1);
    }

    genome->write_to_file(output_genome_filename);
    Log::info("wrote fine-tuned genome to %s\n", output_genome_filename.c_str());

    delete weight_update_method;
    delete training_sets;
    delete validation_sets;
    delete genome;

    Log::release_id("main");
    return 0;
}
