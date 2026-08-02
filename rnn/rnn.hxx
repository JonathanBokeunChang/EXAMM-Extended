#ifndef EXAMM_RNN_GENOME_HXX
#define EXAMM_RNN_GENOME_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "rnn_edge.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_recurrent_edge.hxx"
#include "time_series/time_series.hxx"
// #include "word_series/word_series.hxx"

// Pointwise training-loss variant for the stochastic backprop path. MSE is the
// historical default; HUBER clips the per-timestep residual at +/-delta (robust
// to fat-tailed targets, Gu-Kelly-Xiu 2020). The cross-sectional objectives
// (IC, MSE+dispersion-penalty) are dispatched separately via
// RNN_Genome::backpropagate_cross_sectional and do not use this enum.
// TEMPORAL_IC: per-stock arm (rnn/temporal_ic_loss.*) -- differentiable Pearson
// correlation between one stock's own predicted- and realized-return sequence OVER
// TIME (n_series==1 only; NOT the cross-sectional --loss ic, which pools many stocks
// per date via RNN_Genome::backpropagate_cross_sectional).
enum class LossVariant { MSE, HUBER, TEMPORAL_IC };

// Selection-only fitness mode: which metric decides "best genome" for the search
// (the value written into best_validation_mse). MSE preserves today's default exactly.
// BACKTEST_SPREAD/BACKTEST_SHARPE compute a top-K/bottom-K trading-style metric from
// the genome's own validation predictions (RNN_Genome::compute_backtest_fitness) --
// the TRAINING gradient is unaffected by this; only which genome counts as "best"
// changes (mirrors the existing --loss ic precedent: train on a surrogate, select on
// the true target metric).
enum class FitnessMode { MSE, BACKTEST_SPREAD, BACKTEST_SHARPE };

// Bundles the per-series SGD training path's tunables (RNN_Genome::backpropagate_stochastic,
// RNN::get_analytic_gradient). A single struct instead of accumulating positional params --
// growing that list independently per loss/fitness idea is an easy-to-transpose correctness
// risk. Default-constructed reproduces today's plain-MSE-loss/MSE-fitness behavior exactly.
struct StochasticTrainingOptions {
    LossVariant loss_variant = LossVariant::MSE;
    double huber_delta = 0.0;
    // temporal-IC arm (rnn/temporal_ic_loss.*): weight on the anti-collapse variance floor.
    double temporal_ic_var_lambda = 1.0;
    // pairwise-ranking arm (rnn/pairwise_rank_loss.*): sampled pairs per bp epoch (0 = unset).
    int32_t pairwise_rank_num_pairs = 0;
    FitnessMode fitness_mode = FitnessMode::MSE;
    // backtest-fitness arm: top/bottom-K validation days (0 = unset -> computed as
    // max(5, T_val/10) at call time).
    int32_t backtest_top_k = 0;
};

// Pre-registered Huber delta (NO tuning): 1.345 * std of ALL pooled NORMALIZED
// training-target values (the classical 95%-Gaussian-efficiency constant, Huber
// 1964; Gu-Kelly-Xiu 2020 use Huber for cross-sectional returns). The naked 1.345
// would never clip under avg_std_dev normalization (residual scale ~0.04), so the
// scale must come from the data. Deterministic per dataset -> identical across
// ranks/threads and seeds. Fatals on empty or zero-variance targets.
// training_outputs layout: [series][output_parameter][time].
double huber_delta_from_targets(const std::vector<std::vector<std::vector<double> > >& training_outputs);

class RNN {
   private:
    int32_t series_length;

    vector<RNN_Node_Interface*> input_nodes;
    vector<RNN_Node_Interface*> output_nodes;

    vector<RNN_Node_Interface*> nodes;
    vector<RNN_Edge*> edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

   public:
    RNN(vector<RNN_Node_Interface*>& _nodes, vector<RNN_Edge*>& _edges, const vector<string>& input_parameter_names,
        const vector<string>& output_parameter_names);
    RNN(vector<RNN_Node_Interface*>& _nodes, vector<RNN_Edge*>& _edges, vector<RNN_Recurrent_Edge*>& _recurrent_edges,
        const vector<string>& input_parameter_names, const vector<string>& output_parameter_names);
    ~RNN();

    void fix_parameter_orders(
        const vector<string>& input_parameter_names, const vector<string>& output_parameter_names
    );
    void validate_parameters(const vector<string>& input_parameter_names, const vector<string>& output_parameter_names);

    int32_t get_number_nodes();
    int32_t get_number_edges();

    RNN_Node_Interface* get_node(int32_t i);
    RNN_Edge* get_edge(int32_t i);

    int32_t get_number_recurrent_edges();
    RNN_Recurrent_Edge* get_recurrent_edge(int32_t i);

    // Output-node access for cross-sectional / ranking losses that need to read
    // each network's predictions and write per-timestep output deltas directly
    // (see rnn/ic_loss.*, RNN_Genome::backpropagate_cross_sectional).
    int32_t get_number_output_nodes();
    RNN_Node_Interface* get_output_node(int32_t i);

    void forward_pass(
        const vector<vector<double> >& series_data, bool using_dropout, bool training, double dropout_probability
    );
    void backward_pass(double error, bool using_dropout, bool training, double dropout_probability);

    double calculate_error_softmax(const vector<vector<double> >& expected_outputs);
    double calculate_error_mse(const vector<vector<double> >& expected_outputs);
    double calculate_error_mae(const vector<vector<double> >& expected_outputs);
    // Huber in the r^2 convention (rho = r^2 for |r|<=delta, delta*(2|r|-delta)
    // beyond), so delta -> infinity reduces EXACTLY to calculate_error_mse (same
    // return value, same error_values) -- the bit-identity gate relies on this.
    // error_values[j] = clamp(r, -delta, +delta); the *2.0 lives in the backward
    // scalar, mirroring the MSE path.
    double calculate_error_huber(const vector<vector<double> >& expected_outputs, double delta);
    // Temporal (per-stock) Pearson-IC loss (rnn/temporal_ic_loss.*): requires exactly
    // ONE output node/series (enforced by the caller, see get_analytic_gradient).
    // error_values[j] = d(loss)/d(pred_j), ALREADY fully scaled (Pearson gradient +
    // variance-floor gradient) -- unlike MSE/Huber, the caller must NOT re-scale this
    // by loss*(1/n)*2.0; backward_pass is called with scalar 1.0 (see rnn.cxx).
    double calculate_error_temporal_ic(const vector<vector<double> >& expected_outputs, double var_lambda);

    double prediction_softmax(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool using_dropout,
        bool training, double dropout_probability
    );
    double prediction_mse(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool using_dropout,
        bool training, double dropout_probability
    );
    double prediction_mae(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool using_dropout,
        bool training, double dropout_probability
    );

    vector<double> get_predictions(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool usng_dropout,
        double dropout_probability
    );

    // sequence_length > 0 evaluates the series in independent, contiguous blocks of that
    // many steps instead of one unbroken pass, resetting the hidden state between blocks.
    // This exists so a genome trained with --train_sequence_length N can be SCORED in the
    // regime it was trained in: forward_pass() resets every node and edge on entry, so a
    // block here is byte-for-byte the situation a training slice presented. Evaluating such
    // a genome unsliced instead lets hidden state accumulate over the whole test year --
    // hundreds of steps longer than anything it saw -- which confounds "this knob is bad"
    // with "we scored it out of regime." Default 0 preserves the original single-pass
    // behaviour exactly for every existing caller.
    void write_predictions(
        string output_filename, const vector<string>& input_parameter_names,
        const vector<string>& output_parameter_names, const vector<vector<double> >& series_data,
        const vector<vector<double> >& expected_outputs, TimeSeriesSets* time_series_sets, bool using_dropout,
        double dropout_probability, int32_t sequence_length = 0
    );

    void initialize_randomly();
    void get_weights(vector<double>& parameters);
    void set_weights(const vector<double>& parameters);

    int32_t get_number_weights();

    // options.loss_variant selects the pointwise training loss (default MSE keeps every
    // existing caller unchanged). With HUBER, `mse` returns the Huber loss (r^2 convention)
    // and the backward scalar mirrors the MSE path's loss*(1/n)*2.0 form -- the legacy
    // loss-scaled step is part of the matched configuration, not an accident (see plan:
    // raw-MSE won WITH this quirk). TEMPORAL_IC uses a different backward scalar (1.0) --
    // see rnn.cxx.
    void get_analytic_gradient(
        const vector<double>& test_parameters, const vector<vector<double> >& inputs,
        const vector<vector<double> >& outputs, double& mse, vector<double>& analytic_gradient, bool using_dropout,
        bool training, double dropout_probability,
        const StochasticTrainingOptions& options = StochasticTrainingOptions()
    );
    void get_empirical_gradient(
        const vector<double>& test_parameters, const vector<vector<double> >& inputs,
        const vector<vector<double> >& outputs, double& mae, vector<double>& empirical_gradient, bool using_dropout,
        bool training, double dropout_probability
    );

    // RNN* copy();

    friend void get_mse(
        RNN* genome, const vector<vector<double> >& expected, double& mse, vector<vector<double> >& deltas
    );
    friend void get_mae(
        RNN* genome, const vector<vector<double> >& expected, double& mae, vector<vector<double> >& deltas
    );
};

#endif
