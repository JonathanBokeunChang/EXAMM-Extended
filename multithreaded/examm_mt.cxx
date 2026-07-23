#include <chrono>
#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "examm/examm.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/ic_loss.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

mutex examm_mutex;

vector<string> arguments;

EXAMM* examm;

WeightUpdate* weight_update_method;

// Training objective: "mse" (default, pointwise) or "ic" (cross-sectional
// Information Coefficient ranking loss, rnn/ic_loss.*). When "ic", ic_mode selects
// the differentiable surrogate (pearson | spearman) and ic_var_lambda is the weight
// on the anti-collapse MSE term (L = -IC + lambda*MSE; lambda=0 = pure IC).
string loss_function = "mse";
IcMode ic_mode = IcMode::PEARSON;
double ic_var_lambda = 1.0;
bool ic_select_icir = false;  // fitness: mean IC (false) or IC information ratio (true)

bool finished = false;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > validation_inputs;
vector<vector<vector<double> > > validation_outputs;

void examm_thread(int32_t id) {
    while (true) {
        examm_mutex.lock();
        Log::set_id("main");
        RNN_Genome* genome = examm->generate_genome();
        examm_mutex.unlock();

        if (genome == NULL) {
            break;  // generate_individual returns NULL when the search is done
        }

        string log_id = "genome_" + to_string(genome->get_generation_id()) + "_thread_" + to_string(id);
        Log::set_id(log_id);
        if (loss_function == "ic") {
            genome->backpropagate_cross_sectional(
                training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method, ic_mode,
                ic_var_lambda, ic_select_icir
            );
        } else {
            // genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
            genome->backpropagate_stochastic(
                training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
            );
        }
        Log::release_id(log_id);

        examm_mutex.lock();
        Log::set_id("main");
        examm->insert_genome(genome);
        examm_mutex.unlock();

        delete genome;
    }
}

void get_individual_inputs(string str, vector<string>& tokens) {
    string word = "";
    for (auto x : str) {
        if (x == ',') {
            tokens.push_back(word);
            word = "";
        } else {
            word = word + x;
        }
    }
    tokens.push_back(word);
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    int32_t number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    // Loss selection. Defaults preserve the historical MSE behavior exactly.
    get_argument(arguments, "--loss", false, loss_function);
    if (loss_function == "ic") {
        string ic_mode_str = "pearson";
        get_argument(arguments, "--ic_mode", false, ic_mode_str);
        ic_mode = ic_mode_from_string(ic_mode_str);
        double tau;
        if (get_argument(arguments, "--ic_softrank_tau", false, tau)) {
            IC_SOFTRANK_TAU = tau;
        }
        get_argument(arguments, "--ic_var_lambda", false, ic_var_lambda);
        double var_floor;
        if (get_argument(arguments, "--ic_var_floor", false, var_floor)) {
            IC_VAR_FLOOR = var_floor;
        }
        double spread_floor;
        if (get_argument(arguments, "--ic_spread_floor", false, spread_floor)) {
            IC_SPREAD_FLOOR = spread_floor;
        }
        string ic_fitness = "ic";
        get_argument(arguments, "--ic_fitness", false, ic_fitness);
        ic_select_icir = (ic_fitness == "icir");
        if (ic_fitness != "ic" && ic_fitness != "icir") {
            Log::fatal("unknown --ic_fitness '%s' (expected 'ic' or 'icir')\n", ic_fitness.c_str());
            exit(1);
        }
        Log::info(
            "TRAINING OBJECTIVE: cross-sectional IC (ic_mode=%s, softrank_tau=%g, ic_var_lambda=%g, ic_var_floor=%g, "
            "spread_guard=%g, fitness=%s)\n",
            ic_mode_to_string(ic_mode), IC_SOFTRANK_TAU, ic_var_lambda, IC_VAR_FLOOR, IC_SPREAD_FLOOR, ic_fitness.c_str()
        );
    } else if (loss_function == "mse") {
        Log::info("TRAINING OBJECTIVE: MSE (default)\n");
    } else {
        Log::fatal("unknown --loss '%s' (expected 'mse' or 'ic')\n", loss_function.c_str());
        exit(1);
    }

    TimeSeriesSets* time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    get_train_validation_data(
        arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs
    );

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome* seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);

    examm = generate_examm_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);

    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back(thread(examm_thread, i));
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    Log::info("completed!\n");
    Log::release_id("main");

    return 0;
}
