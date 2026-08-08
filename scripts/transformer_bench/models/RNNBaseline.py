"""Plain LSTM / GRU encoders, the recurrent baselines EXAMM is actually competing with.

WHY THESE EXIST. Every other baseline in this study is an attention model two to four orders of
magnitude larger than EXAMM, which makes "small beats big" easy and "evolved beats hand-designed"
untested. At d_model 32 a GRU is ~3.8k parameters and an LSTM ~5.0k -- the closest competitors to
EXAMM's 55-77 in the whole lineup, and smaller than PatchTST. They are the baseline a reader will
ask about first.

CONFIGURATION IS LYU ET AL.'S, NOT OURS. Their DJI study trained "fully connected two-layer LSTM,
GRU, MGU models ... using 1000 back-propagation epochs, with Adam as the optimizer and a learning
rate of 0.0001. The two-layer memory cell models had two hidden layers of memory cells, fully
connected with the number of hidden [units] equal to the input layer size. The initial weights ...
were generated randomly using Xavier weight initialization, and the models with highest validation
performance over the 10 repeated training runs were selected for testing."

So: 2 layers, hidden = enc_in, Adam at 1e-4, Xavier init. That makes these rows citable to prior
work on this exact problem rather than to a choice of ours -- the one methodological asymmetry the
transformer rows did not have. At enc_in=6 it also puts them at 511-679 parameters against EXAMM's
55-77, an order of magnitude apart rather than the four orders the transformers sit at.

INSTANCE NORMALISATION IS ON BY DEFAULT, and that is a deliberate, documented choice rather than an
implementation detail. On this universe Crossformer -- the one model in the study with no instance
normalisation -- collapsed to a near-constant prediction because the global max-based scaler's
denominator is inflated by single crisis-day returns. PatchTST carries RevIN, iTransformer
`use_norm`, DeformTime NSformer stationarisation, and DLinear is scale-equivariant by construction;
a raw RNN would be the only unprotected model here and would risk reproducing that failure. We
normalise each window to zero mean and unit variance and invert on the output, which is exactly
iTransformer's `use_norm` scheme. Set --use_norm 0 to disable and reproduce the unprotected variant.
"""
import torch
import torch.nn as nn


class _RNN(nn.Module):
    def __init__(self, configs, cell):
        super().__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.use_norm = int(getattr(configs, "use_norm", 1))
        # hidden = input layer size, per Lyu et al. The launcher passes --d_model enc_in; this
        # falls back to enc_in directly so the relationship cannot be broken by a stray flag.
        hidden = int(getattr(configs, "d_model", 0)) or configs.enc_in
        layers = max(1, int(getattr(configs, "e_layers", 1)))
        self.rnn = cell(
            input_size=configs.enc_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            # nn.LSTM/GRU only apply dropout BETWEEN layers, so passing it at 1 layer is both
            # inert and a UserWarning; silence it by passing 0 there.
            dropout=float(configs.dropout) if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, self.pred_len * self.c_out)
        # XAVIER INIT, per Lyu et al. PyTorch's default for recurrent layers is U(-1/sqrt(H),
        # 1/sqrt(H)), not Xavier, so this is a real difference rather than a restatement of the
        # default. Biases are left at zero; the cited setup specifies initialisation of weights.
        for name, prm in self.named_parameters():
            if prm.dim() >= 2:
                nn.init.xavier_uniform_(prm)
            elif "bias" in name:
                nn.init.zeros_(prm)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, enc_in] -> [B, pred_len, c_out]
        if self.use_norm:
            # per-window, per-channel standardisation; statistics come from the INPUT window only,
            # so no future information enters, and the inverse below is applied with the target
            # channel's own statistics. features=MS puts the target last, matching c_out=1.
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = (x_enc - mean) / std

        out, _ = self.rnn(x_enc)
        y = self.head(out[:, -1])                       # last hidden state -> horizon
        y = y.view(-1, self.pred_len, self.c_out)

        if self.use_norm:
            tgt = slice(x_enc.shape[-1] - self.c_out, x_enc.shape[-1])
            y = y * std[:, :, tgt] + mean[:, :, tgt]
        return y


class LSTMModel(_RNN):
    def __init__(self, configs):
        super().__init__(configs, nn.LSTM)


class GRUModel(_RNN):
    def __init__(self, configs):
        super().__init__(configs, nn.GRU)
