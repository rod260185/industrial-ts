# industrial-ts

`industrial-ts` is a PyTorch toolkit for industrial time-series modeling with a shared API for recurrent, ODE-based, and PatchTST models.

The package is designed for industrial datasets where the same model may combine:

- reconstruction and imputation of observed channels;
- future-horizon prediction;
- probabilistic decoding with Gaussian/VAE-style heads;
- diffusion-style denoising;
- event detection over the prediction horizon.

Current model wrappers:

- `ITS_GRU`
- `ITS_LSTM`
- `ITS_ODEJump`
- `ITS_PatchTST`

All wrappers inherit from `BaseIndustrialTSModel` and expose the same main methods:

- `train_model(...)`
- `test_model(...)`
- `predict(...)`
- `save_weights(...)`
- `load_weights(...)`

## License, Citation, and Data Policy

`industrial-ts` is licensed under the Apache License, Version 2.0.

Copyright 2026 Rodrigo Petrus Domingues.

This library is associated with the forthcoming article:

> Safety-Oriented ODEJump Adaptation for Future-Time Anomaly Prediction in Oil and Gas Compressor Systems

Rodrigo Petrus Domingues is the principal author of the article and the initial copyright holder of this software release.

If you use this software in academic work, please cite the repository and, once available, the associated article. Citation metadata is provided in `CITATION.cff`.

Unless explicitly stated otherwise, this repository does not include real industrial operating data, or confidential information. Any datasets added in the future should include their own license file. Synthetic or demonstrative datasets may use CC BY 4.0 or CC0, depending on the intended reuse model.

## Table of Contents

- [License, Citation, and Data Policy](#license-citation-and-data-policy)
- [Installation and Imports](#installation-and-imports)
- [Data Format](#data-format)
- [Core Concepts](#core-concepts)
- [Available Models](#available-models)
- [Encoders](#encoders)
- [Decoders](#decoders)
- [Cost Functions and Heads](#cost-functions-and-heads)
- [Quick Example](#quick-example)
- [Training](#training)
- [Testing](#testing)
- [Prediction](#prediction)
- [Configuration Recipes](#configuration-recipes)
- [Events with `limit_events`](#events-with-limitevents)
- [Irregularity and Missingness](#irregularity-and-missingness)
- [Metrics and Logs](#metrics-and-logs)
- [Saving and Loading](#saving-and-loading)
- [Practical Tips](#practical-tips)
- [Project Structure](#project-structure)
- [Quick Reference](#quick-reference)

## Installation and Imports

For local development, install the project in editable mode:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[examples]"
pip install -e ".[viz]"
pip install -e ".[dev]"
```

In notebooks, if the package has not been installed yet, add the parent directory to `sys.path`:

```python
import sys
from pathlib import Path

repo_parent = Path.cwd().resolve().parent
if str(repo_parent) not in sys.path:
    sys.path.append(str(repo_parent))
```

Typical imports:

```python
from industrial_ts.models import (
    ITS_GRU,
    ITS_LSTM,
    ITS_ODEJump,
    ITS_PatchTST,
)

from industrial_ts.confs import (
    TSDecoderType,
    EncoderType,
    MergeMethod,
    PredictionType,
)

from industrial_ts.cost_functions import (
    MSECostFunction,
    MSECostFunctionDecoder,
    LogLikelihoodCostFunction,
    LogLikelihoodCostFunctionDecoder,
    NoisePredictionCostFunction,
    MaskBCECostFunction,
    EventsCostFunction,
    RecallPenaltyCostFunction,
    PrecisionPenaltyCostFunction,
    NF1MicroCostFunction,
    NF1MacroCostFunction,
    NELBOCostFunction,
    NELBOCostFunctionDecoder,
    NLLCostFunction,
    NLLCostFunctionDecoder,
)
```

## Data Format

Training, testing, and prediction methods expect a pandas `DataFrame`.

The core column groups are:

- `input_cols`: numeric time-series channels used as model input.
- `time_col`: timestamp column. Use `"index"` to use the DataFrame index.
- `cost_cols`: subset of `input_cols` used as prediction targets for decoder heads.
- `status_cols`: event/status columns used by softmax event heads when `limit_events=None`.
- `context_cols`: optional static/context features projected once per window.

Example:

```python
input_cols = list(df.columns[:11])
time_col = "index"
cost_cols = input_cols
```

Important expectations:

- The data must be sortable by `time_col` or by index.
- `input_cols`, `cost_cols`, and `context_cols` must be numeric.
- Missing values in `input_cols` are converted to zeros and tracked through masks.
- `cost_cols` must be present in `input_cols`.
- Decoder losses require `head_window_size >= 1`.

## Core Concepts

### `window_size`

Number of timesteps used as encoder context.

```python
window_size = 40
```

Input windows have shape:

```text
batch_size x window_size x num_features
```

### `head_window_size`

Number of future timesteps decoded by prediction, simulation, and event heads.

```python
head_window_size = 20
```

Use `head_window_size=0` only for heads that do not need a future horizon, such as direct reconstruction.

### `cost_functions`

`cost_functions` is a dictionary. The key is the metric/loss name used in logs, and the value is a `CostFunction` instance.

```python
cost_functions = {
    "mse": MSECostFunctionDecoder(training_ratio=0.4, test_ratio=0.5),
    "events": EventsCostFunction(training_ratio=0.4, test_ratio=0.5),
    "nll": LogLikelihoodCostFunctionDecoder(training_ratio=0.1, test_ratio=1e-8),
}
```

Each cost function declares the model heads it requires through its `heads` class attribute. The base model builds the required heads automatically.

### `cost_cols`

`cost_cols` is a single list of target columns, not a list per loss.

```python
cost_cols = input_cols
```

Decoder losses use `cost_cols` to build `head_cost` and `mask_head_cost` over the future horizon.

### `training_ratio` and `test_ratio`

Each cost function has its own contribution weight for training and evaluation:

```python
MSECostFunctionDecoder(training_ratio=0.4, test_ratio=0.5)
```

During training, the model adds:

```text
training_ratio * normalized_loss
```

During validation/testing, the reported score uses `test_ratio`.

### `loss_params`

`loss_params` is passed as keyword arguments to every configured cost function.

Common values:

```python
loss_params = {
    "event_weight": 4.0,
    "sigma_temp": 1.0,
    "kl_scale": 1e-3,
}
```

Unused keys are ignored by cost functions that accept `**kwargs`.

### `ts_decoder_params`

`ts_decoder_params` is passed to the temporal decoder at runtime.

For the Transformer decoder, useful keys include:

```python
ts_decoder_params = {
    "decoder_teacher_forcing": True,
    "decoder_teacher_forcing_start": 1.0,
    "decoder_teacher_forcing_end": 0.25,
    "decoder_teacher_forcing_decay_epochs": 20,
}
```

### `activation`

`activation` controls the nonlinear blocks used by the default MLP encoder and heads. It can be a single `nn.Module`, an `nn.Module` class, a zero-argument factory returning an `nn.Module`, or a dictionary keyed by block name:

```python
activation={
    "default": nn.GELU,
    "x_head": nn.SiLU,
    "vae": nn.GELU,
}
```

Supported keys include `encoder`, `static_encoder`, `x_head`, `noise_head`, `lambda_time_embedding`, `lambda_head`, `events_head`, `vae`, `vae_latent`, `vae_decoder`, and `vae_sigma_head`. Custom activations must preserve the tensor shape expected by the surrounding linear layers; for example, an `x_head` activation receives `hidden_dim * 2` features and must return `hidden_dim * 2` features.

## Available Models

### `ITS_GRU`

GRU temporal encoder.

Typical use:

- fast recurrent baseline;
- regular or moderately irregular data;
- forecasting and event detection;
- bidirectional offline diagnostics.

Example:

```python
model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_encoder_layers=1,
    bidirectional=False,
    ts_decoder_type=TSDecoderType.GRU,
)
```

### `ITS_LSTM`

LSTM temporal encoder.

By default, `ITS_LSTM` uses `TSDecoderType.LSTM`.

Example:

```python
model = ITS_LSTM(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_encoder_layers=2,
    bidirectional=True,
    merge_method=MergeMethod.LINEAR,
)
```

### `ITS_ODEJump`

ODE-Jump temporal encoder.

The encoder integrates latent dynamics between observation times and applies jumps at observed timesteps.

Typical use:

- irregular sampling;
- missing observations;
- continuous-time latent dynamics;
- event prediction under irregular intervals.

Example:

```python
model = ITS_ODEJump(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_encoder_layers=1,
    layernorm_mode="gru",
    ts_decoder_type=TSDecoderType.TRANSFORMER,
)
```

For bidirectional ODE-Jump encoders with decoder heads, the decoder initial state is built with an explicit final-time adapter instead of reusing recurrent fusers as one-step sequence mergers. Use `MergeMethod.LINEAR` or `MergeMethod.FUSER`; `MergeMethod.GRU_FUSER` and `MergeMethod.LSTM_FUSER` are rejected when decoder heads are present. The optional `decoder_init_method` accepts `"linear"` or `"gate"`.

### `ITS_PatchTST`

PatchTST-style encoder wrapper.

The model sets `encoder_type=EncoderType.PATCHTST` internally and uses a patch-based Transformer encoder before the shared IndustrialTS heads and decoders.

Example:

```python
model = ITS_PatchTST(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    encoder_patch_len=8,
    encoder_patch_stride=4,
    encoder_layers=2,
    encoder_nhead=4,
    encoder_dropout=0.05,
    ts_decoder_type=TSDecoderType.TRANSFORMER,
)
```

The encoder keeps PatchTST's main inductive bias: each channel is patched as an independent univariate series before Transformer mixing. When timestamps are available, it uses patch-level time embeddings; positional embeddings remain as a fallback.

## Encoders

The base encoder is controlled by:

```python
encoder_type=EncoderType.MLP
```

Supported values:

| Encoder | Enum | Description |
|---|---|---|
| MLP | `EncoderType.MLP` | Dense per-timestep projection from `[value, mask]` into `hidden_dim`. |
| PatchTST | `EncoderType.PATCHTST` | Channel-independent patch encoder with Transformer layers. |

`ITS_PatchTST` sets `EncoderType.PATCHTST` automatically. Other model wrappers can also receive `encoder_type=EncoderType.PATCHTST` when this is compatible with the intended architecture.

Bidirectional encoders can merge forward/backward states with:

| Merge method | Enum |
|---|---|
| Linear projection | `MergeMethod.LINEAR` |
| Generic fuser | `MergeMethod.FUSER` |
| GRU fuser | `MergeMethod.GRU_FUSER` |
| LSTM fuser | `MergeMethod.LSTM_FUSER` |

## Decoders

The temporal decoder is selected with `ts_decoder_type`.

Available decoder types:

| Decoder | Enum | Description |
|---|---|---|
| GRU | `TSDecoderType.GRU` | Autoregressive GRU decoder. |
| LSTM | `TSDecoderType.LSTM` | Autoregressive LSTM decoder. |
| Transformer | `TSDecoderType.TRANSFORMER` | Time-aware Transformer decoder with optional teacher forcing. |
| ODE-Jump | `TSDecoderType.ODE_JUMP` | Continuous-time decoder for future horizons. |

Base model wrappers support:

```python
TSDecoderType.GRU
TSDecoderType.LSTM
TSDecoderType.ODE_JUMP
TSDecoderType.TRANSFORMER
```

### Transformer Decoder

```python
model = ITS_ODEJump(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_decoder_type=TSDecoderType.TRANSFORMER,
    ts_decoder_layers=2,
    ts_decoder_nhead=4,
    ts_decoder_dropout=0.05,
    ts_decoder_max_length=512,
)
```

Use it when decoded timesteps should attend to each other and when future timestamps are informative.

### ODE-Jump Decoder

```python
model = ITS_ODEJump(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_decoder_type=TSDecoderType.ODE_JUMP,
)
```

Use it when the future horizon itself should be evolved as continuous-time dynamics.

## Cost Functions and Heads

Cost functions determine which heads are built.

| Cost function | Required heads | Typical use |
|---|---|---|
| `MSECostFunction` | `x` | Reconstruction/imputation on the input window. |
| `MSECostFunctionDecoder` | `x`, `decoder` | Future-horizon MSE prediction. |
| `LogLikelihoodCostFunction` | `x`, `lambda` | Reconstruction with learned Gaussian precision. |
| `LogLikelihoodCostFunctionDecoder` | `x`, `lambda`, `decoder` | Future-horizon Gaussian negative log-likelihood. |
| `NELBOCostFunction` | `vae_x` | VAE-style reconstruction loss with KL term. |
| `NELBOCostFunctionDecoder` | `decoder`, `vae_x` | VAE-style future-horizon simulation. |
| `NLLCostFunction` | `vae_x` | Gaussian NLL reconstruction without KL. |
| `NLLCostFunctionDecoder` | `decoder`, `vae_x` | Gaussian NLL prediction without KL. |
| `NoisePredictionCostFunction` | `noise` | Diffusion-style noise prediction. Must be combined with an `x` head. |
| `MaskBCECostFunction` | `miss` | Binary mask/observability prediction. |
| `EventsCostFunction` | `x`, `decoder`, `events` | Event classification over the decoded horizon. |
| `NF1CostFunction` | `x`, `decoder`, `events` | Negative F1 event objective, micro-averaged by default. |
| `NF1MicroCostFunction` | `x`, `decoder`, `events` | Negative micro-F1 event objective. |
| `NF1MacroCostFunction` | `x`, `decoder`, `events` | Negative macro-F1 event objective. |
| `NRecallCostFunction` | `x`, `decoder`, `events` | Negative recall event objective, micro-averaged by default. |
| `NSpecificityCostFunction` | `x`, `decoder`, `events` | Negative specificity event objective, micro-averaged by default. |
| `NPrecisionCostFunction` | `x`, `decoder`, `events` | Negative precision event objective, micro-averaged by default. |
| `RecallPenaltyCostFunction` | `x`, `decoder`, `events` | Soft recall target penalty for events. |
| `PrecisionPenaltyCostFunction` | `x`, `decoder`, `events` | Soft precision target penalty for events. |

Example mixed objective:

```python
cost_functions = {
    "mse": MSECostFunctionDecoder(training_ratio=0.4, test_ratio=0.5),
    "events": EventsCostFunction(training_ratio=0.4, test_ratio=0.5),
    "nll": LogLikelihoodCostFunctionDecoder(training_ratio=0.1, test_ratio=1e-8),
}
```

Example with event penalties:

```python
cost_functions = {
    "events": EventsCostFunction(training_ratio=0.7, test_ratio=0.7),
    "recall_penalty": RecallPenaltyCostFunction(
        target_recall=0.85,
        training_ratio=0.2,
        test_ratio=0.0,
    ),
    "precision_penalty": PrecisionPenaltyCostFunction(
        target_precision=0.75,
        training_ratio=0.1,
        test_ratio=0.0,
    ),
}
```

## Quick Example

```python
import torch
from sklearn.preprocessing import RobustScaler

from industrial_ts.models import ITS_GRU
from industrial_ts.confs import TSDecoderType
from industrial_ts.cost_functions import MSECostFunctionDecoder

input_cols = list(df.columns[:11])
cost_cols = input_cols

cost_functions = {
    "mse": MSECostFunctionDecoder(),
}

model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    dtype="float32",
    ts_decoder_type=TSDecoderType.GRU,
)

model.train_model(
    data=df,
    window_size=40,
    window_stride=5,
    optimizer_class=torch.optim.AdamW,
    optimizer_base_params={
        "lr": 1e-4,
        "weight_decay": 2e-3,
    },
    scheduler_params={
        "warmup_steps": 0.12,
        "warmup_min_steps": 10,
        "min_lr_factor": 0.05,
    },
    test_ratio=0.2,
    epochs=80,
    patience=15,
    scaler=RobustScaler,
    log_metrics=True,
    head_window_size=20,
    batch_size=64,
    best_model_path="./models/gru_mse.pt",
    log_path="./training_logs/gru_mse.json",
)
```

## Training

Main method:

```python
model.train_model(
    data=df,
    window_size=40,
    optimizer_class=torch.optim.AdamW,
    optimizer_base_params={"lr": 1e-4, "weight_decay": 2e-3},
    optimizer_specific_params=None,
    optimizer_additional_kwargs={},
    optimizer_scaler=None,
    window_stride=1,
    validation_ratio=0.0,
    test_ratio=0.2,
    cross_validation=False,
    cv_n_splits=5,
    cv_validation_size=None,
    cv_train_size=None,
    cv_gap=None,
    cv_strategy="expanding",
    batch_size=32,
    epochs=100,
    patience=10,
    best_model_path="best_model.pt",
    scaler=None,
    log_metrics=False,
    log_path="training_log.json",
    scheduler_params=None,
    head_window_size=0,
    force_irregular_timestep=False,
    force_irregular_timestep_max_drop=0.3,
    force_irregular_timestep_test=False,
    rebuild_timeseries=False,
    rebuild_timeseries_test=False,
    test_repetition=13,
    rebuild_timeseries_max_drop=0.3,
    rebuild_timeseries_max_drop_step=0.3,
    loss_params={},
    ts_decoder_params={},
)
```

`train_model(...)` saves the best weights to `best_model_path`.

### Time-Series Cross-Validation

Set `cross_validation=True` to run ordered time-series validation folds instead of the default train/validation/test split.

```python
cv_result = model.train_model(
    data=df,
    window_size=40,
    window_stride=5,
    optimizer_class=torch.optim.AdamW,
    optimizer_base_params={"lr": 1e-4, "weight_decay": 2e-3},
    validation_ratio=0.15,
    test_ratio=0.2,
    cross_validation=True,
    cv_n_splits=5,
    cv_validation_size=0.15,
    cv_gap=None,
    cv_strategy="expanding",
    epochs=80,
    patience=15,
    scaler=RobustScaler,
    log_metrics=True,
    log_path="./validation_logs/model_cv.json",
    best_model_path="./models/model_cv.pt",
    head_window_size=20,
    batch_size=64,
)
```

When `cross_validation=True`, `test_ratio` is not used for fold construction. Each fold has only a training segment and a future validation segment; the log stores `"test_ratio": null` under `params`. Use `test_model(...)` afterward for a final holdout evaluation.

Fold construction is chronological:

- `cv_strategy="expanding"` trains each fold from the beginning of the windowed dataset up to the fold boundary.
- `cv_strategy="rolling"` trains each fold on a fixed-size trailing window and requires `cv_train_size`.
- `cv_validation_size` accepts either an integer number of windows or a ratio in `(0, 1)`.
- If `cv_validation_size=None`, `validation_ratio` is used only to choose the fold validation size. It does not create a separate validation split.
- If `cv_gap=None`, the purge gap is `window_size + head_window_size - 1`, which prevents overlap leakage between the last training window and the validation horizon.

Each fold starts from the model's initial weights, fits the scaler only on that fold's training windows, and saves fold weights as:

```text
./models/model_cv_fold01.pt
./models/model_cv_fold02.pt
...
```

The JSON log contains a top-level cross-validation payload:

```json
{
  "mode": "time_series_cross_validation",
  "params": {
    "cv_n_splits": 5,
    "cv_validation_size": 0.15,
    "cv_train_size": null,
    "cv_gap": null,
    "cv_strategy": "expanding",
    "test_ratio": null
  },
  "summary": {
    "best_epochs": [12, 15, 11],
    "best_epoch_mean": 12.6666666667,
    "best_epoch_median": 12.0,
    "validation_scores": [0.21, 0.19, 0.23],
    "validation_score_mean": 0.21,
    "validation_score_median": 0.21,
    "best_validation_loss_summary": {},
    "best_validation_metrics_summary": {},
    "selection_criterion": "validation_loss.avg"
  },
  "folds": []
}
```

`validation_scores` are the selected fold scores computed from validation losses with the configured cost-function `test_ratio` weights. The summary also reports mean and median values for each numeric validation loss and metric available at the best epoch of each fold.

### AMP and `optimizer_scaler`

When using CUDA float16 automatic mixed precision, pass a scaler:

```python
optimizer_scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),
)
```

In practice, the scaler multiplies the loss before backpropagation, unscales gradients before clipping/optimizer step, and skips the step if non-finite gradients are detected.

For bfloat16, gradient scaling is usually unnecessary because bfloat16 has the same exponent range as float32, although it has lower mantissa precision.

### Optimizer Parameter Groups

`optimizer_specific_params` matches parameter names by substring.

Example:

```python
optimizer_specific_params = {
    "bias": {
        "lr": 1e-4,
        "weight_decay": 0.0,
    },
    "lambda_head": {
        "lr": 6e-5,
        "weight_decay": 1e-3,
    },
    "ts_encoder": {
        "lr": 8e-5,
        "weight_decay": 1e-3,
    },
    "ts_decoder": {
        "lr": 1e-4,
        "weight_decay": 2e-3,
    },
    "decoder": {
        "lr": 1e-4,
        "weight_decay": 2e-3,
    },
}
```

If a configured group does not match any parameter, the model raises `ValueError`.

### Scheduler Parameters

The built-in scheduler is linear warmup followed by cosine decay.

```python
scheduler_params = {
    "warmup_steps": 0.12,
    "warmup_min_steps": 10,
    "min_lr_factor": 0.05,
}
```

`warmup_steps` can be an absolute step count or a fraction of total training steps.

Conservative settings for PatchTST or ODE-heavy configurations:

```python
optimizer_base_params = {
    "lr": 8e-5,
    "weight_decay": 1e-3,
}

optimizer_specific_params = {
    "bias": {
        "lr": 8e-5,
        "weight_decay": 0.0,
    },
    "lambda_head": {
        "lr": 5e-5,
        "weight_decay": 5e-4,
    },
    "ts_encoder": {
        "lr": 6e-5,
        "weight_decay": 5e-4,
    },
    "ts_decoder": {
        "lr": 8e-5,
        "weight_decay": 1e-3,
    },
    "decoder": {
        "lr": 8e-5,
        "weight_decay": 1e-3,
    },
}

scheduler_params = {
    "warmup_steps": 0.18,
    "warmup_min_steps": 20,
    "min_lr_factor": 0.08,
}
```

### Scaling

Use a scikit-learn scaler class:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler
```

`RobustScaler` is a practical default for industrial signals because it is less sensitive to outliers.

## Testing

Evaluate the model with:

```python
result = model.test_model(
    data=df,
    window_size=40,
    window_stride=5,
    test_ratio=1.0,
    scaler=RobustScaler,
    head_window_size=20,
    batch_size=64,
    log_test_metrics=True,
    log_path="./training_logs/test_metrics.json",
)
```

The return value includes:

```python
{
    "train_idx": ...,
    "val_idx": ...,
    "test_idx": ...,
    "val_loss": ...,
    "val_score": ...,
    "val_metrics": ...,
    "test_loss": ...,
    "test_score": ...,
    "test_metrics": ...,
}
```

By default, `test_model(...)` evaluates the full dataset as the test split with `test_ratio=1.0`.

## Prediction

Main method:

```python
pred = model.predict(
    prediction_type=PredictionType.PREDICT,
    data=df,
    window_size=40,
    window_stride=5,
    scaler=RobustScaler,
    head_window_size=20,
    batch_size=64,
)
```

Supported prediction types:

| Prediction type | Required heads | Main output keys |
|---|---|---|
| `PredictionType.RECONSTRUCTION` | `x` | `x_hat`, `x_reconstruction` |
| `PredictionType.PREDICT` | `x`, `decoder` | `x_hat`, `lambda_hat` when available |
| `PredictionType.SIMULATE` | `vae_x`, `decoder` | `x_sim`, `vae_x`, `vae_mu`, `vae_logvar`, `vae_std_obs` |
| `PredictionType.DENOISE` | `noise` | `h_denoised`, `noise_hat`, `x_denoised` when an x decoder exists |
| `PredictionType.PREDICT_EVENT` | `x`, `decoder`, `events` | `events_hat`, `events_pred` |

You can also pass prediction type names as strings:

```python
pred = model.predict(
    prediction_type="predict",
    data=df,
    window_size=40,
    head_window_size=20,
)
```

### Forecast / Future Prediction

```python
forecast = model.predict(
    prediction_type=PredictionType.PREDICT,
    data=df,
    window_size=40,
    window_stride=5,
    scaler=RobustScaler,
    head_window_size=20,
)
```

### Reconstruction

```python
reconstruction = model.predict(
    prediction_type=PredictionType.RECONSTRUCTION,
    data=df,
    window_size=40,
    window_stride=5,
    scaler=RobustScaler,
)
```

### Event Prediction

```python
events = model.predict(
    prediction_type=PredictionType.PREDICT_EVENT,
    data=df,
    window_size=40,
    window_stride=5,
    scaler=RobustScaler,
    head_window_size=20,
)
```

`events_hat` is already a probability tensor:

- sigmoid output when `limit_events` is set;
- softmax output when using `status_cols` without `limit_events`.

## Configuration Recipes

### GRU Forecasting Baseline

```python
cost_functions = {
    "mse": MSECostFunctionDecoder(),
}

model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=input_cols,
    hidden_dim=64,
    ts_decoder_type=TSDecoderType.GRU,
)
```

### Bidirectional GRU with Fuser

```python
model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    bidirectional=True,
    merge_method=MergeMethod.GRU_FUSER,
    ts_decoder_type=TSDecoderType.GRU,
)
```

Use bidirectional models for offline tasks where the complete input window is available.

### LSTM with LSTM Decoder

```python
model = ITS_LSTM(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_encoder_layers=2,
    ts_decoder_type=TSDecoderType.LSTM,
)
```

### ODE-Jump with Transformer Decoder

```python
model = ITS_ODEJump(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    layernorm_mode="gru",
    ts_decoder_type=TSDecoderType.TRANSFORMER,
    ts_decoder_layers=2,
    ts_decoder_nhead=4,
    ts_decoder_dropout=0.05,
)
```

Recommended for irregular timestamps and multi-step prediction.

### PatchTST

```python
model = ITS_PatchTST(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    encoder_patch_len=8,
    encoder_patch_stride=4,
    encoder_layers=2,
    encoder_nhead=4,
    encoder_dropout=0.05,
    ts_decoder_type=TSDecoderType.TRANSFORMER,
)
```

Recommended optimizer settings:

```python
optimizer_base_params = {
    "lr": 8e-5,
    "weight_decay": 1e-3,
}

optimizer_specific_params = {
    "bias": {
        "lr": 8e-5,
        "weight_decay": 0.0,
    },
    "lambda_head": {
        "lr": 5e-5,
        "weight_decay": 5e-4,
    },
    "ts_encoder": {
        "lr": 6e-5,
        "weight_decay": 5e-4,
    },
    "ts_decoder": {
        "lr": 8e-5,
        "weight_decay": 1e-3,
    },
    "decoder": {
        "lr": 8e-5,
        "weight_decay": 1e-3,
    },
}

scheduler_params = {
    "warmup_steps": 0.18,
    "warmup_min_steps": 20,
    "min_lr_factor": 0.08,
}
```

### VAE / Simulation

```python
cost_functions = {
    "nelbo": NELBOCostFunctionDecoder(training_ratio=1.0, test_ratio=1.0),
}

model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_decoder_type=TSDecoderType.GRU,
)

sim = model.predict(
    prediction_type=PredictionType.SIMULATE,
    data=df,
    window_size=40,
    head_window_size=20,
    params={"sigma_temp": 1.0},
)
```

### Denoising

`NoisePredictionCostFunction` must be combined with another cost function that creates the `x` head.

```python
cost_functions = {
    "mse": MSECostFunction(training_ratio=0.5, test_ratio=0.5),
    "noise": NoisePredictionCostFunction(training_ratio=0.5, test_ratio=0.5),
}
```

## Events with `limit_events`

There are two event-target modes.

### Threshold-Derived Events

When `limit_events` is set, event labels are derived from `head_cost`:

```python
events = (head_cost <= lower_limit) | (head_cost >= upper_limit)
```

Configuration:

```python
cost_functions = {
    "events": EventsCostFunction(training_ratio=1.0, test_ratio=1.0),
}

model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    cost_functions=cost_functions,
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_decoder_type=TSDecoderType.GRU,
    limit_events=(-3.0, 3.0),
)
```

In this mode:

- the event head output dimension is `len(cost_cols)`;
- the event head uses sigmoid;
- events are treated as multilabel targets.

### Status-Column Events

When `limit_events=None`, the model expects `status_cols` and builds a softmax event head.

```python
model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    status_cols=["event_start", "event_stop"],
    cost_functions={
        "events": EventsCostFunction(),
    },
    cost_cols=cost_cols,
    hidden_dim=64,
    ts_decoder_type=TSDecoderType.GRU,
)
```

In this mode:

- the event head output dimension is `len(status_cols)`;
- the event head uses softmax;
- targets are mutually exclusive classes per future timestep.

### Event Weighting

For imbalanced event labels, pass `event_weight` in `loss_params`:

```python
loss_params = {
    "event_weight": 4.0,
}
```

For multilabel events, a scalar weight applies to all positive labels. A list or tensor can provide one weight per event output.

## Irregularity and Missingness

The training and prediction APIs can simulate irregular timesteps:

```python
force_irregular_timestep=True
force_irregular_timestep_max_drop=0.5
```

This randomly drops timesteps from generated windows.

The APIs can also rebuild time series with extra missingness:

```python
rebuild_timeseries=True
rebuild_timeseries_max_drop=0.3
rebuild_timeseries_max_drop_step=0.3
```

Useful model choices for irregular data:

- `ITS_ODEJump`
- `ITS_PatchTST` with timestamp-aware patch embeddings
- `TSDecoderType.TRANSFORMER`
- `TSDecoderType.ODE_JUMP`

## Metrics and Logs

Enable training logs with:

```python
log_metrics=True
log_path="./training_logs/model.json"
```

Enable test logs with:

```python
log_test_metrics=True
log_path="./training_logs/test_metrics.json"
```

A typical event metrics subtree contains:

```json
{
  "events": {
    "loss": 0.54,
    "accuracy": 0.91,
    "auc": 0.84,
    "events": {
      "event_0": {
        "timestep": {
          "tp": 12,
          "tn": 931,
          "fp": 7,
          "fn": 9,
          "true": {
            "precision": 0.63,
            "recall": 0.57,
            "f1": 0.60
          },
          "false": {
            "precision": 0.99,
            "recall": 0.99,
            "f1": 0.99
          }
        },
        "global": {
          "tp": 8,
          "tn": 40,
          "fp": 3,
          "fn": 4
        }
      }
    }
  }
}
```

For event heads:

- `precision` is the fraction of predicted positives that were correct.
- `recall` is the fraction of real positives that were recovered.
- `f1` is the harmonic mean of precision and recall.
- `auc` is computed when both classes are present and scores are available.

## Saving and Loading

Training saves the best weights automatically:

```python
best_model_path="./models/my_model.pt"
```

Manual save:

```python
model.save_weights("./models/my_model.pt")
```

Manual load:

```python
model.load_weights("./models/my_model.pt")
```

The Python model object must be created with the same architecture before loading weights.

## Practical Tips

- Start with `ITS_GRU` and `MSECostFunctionDecoder` for a fast baseline.
- Use `ITS_LSTM` when recurrent memory is more important than speed.
- Use `ITS_ODEJump` when irregular timestamps are central to the problem.
- Use `ITS_PatchTST` when channel-local patch structure matters.
- Use `RobustScaler` for industrial signals with outliers.
- Keep `hidden_dim=64` for first experiments.
- Reduce learning rate before increasing model size.
- Use lower learning rates for `ts_encoder`, `ts_decoder`, and `lambda_head`.
- Use `limit_events` and `event_weight` for imbalanced threshold events.
- Use float16 AMP with `GradScaler` on CUDA; bfloat16 usually does not need scaling.

## Project Structure

Simplified structure:

```text
industrial_ts/
├── confs.py
├── cost_functions/
│   ├── __init__.py
│   ├── events_cost_function.py
│   ├── log_likelihood_cost_function.py
│   ├── mask_bce_cost_function.py
│   ├── mse_cost_function.py
│   ├── nmetric_cost_function.py
│   ├── noise_prediction_cost_function.py
│   ├── precision_penalty_cost_function.py
│   ├── recall_penalty_cost_function.py
│   └── vae_nll_cost_function.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── gru.py
│   ├── lstm.py
│   ├── ode_jump.py
│   ├── patchtst.py
│   ├── decoders/
│   │   ├── gru.py
│   │   ├── lstm.py
│   │   ├── ode_jump.py
│   │   └── transformer.py
│   ├── encoders/
│   │   ├── _common.py
│   │   ├── gru.py
│   │   ├── lstm.py
│   │   ├── mlp.py
│   │   ├── ode_jump.py
│   │   └── patchtst.py
│   └── fusers/
│       └── __init__.py
├── dataloader_examples/
│   ├── cognite_compressor.py
│   ├── use_examples.ipynb
│   ├── cv_logs/
│   └── training_logs/
└── pyproject.toml
```

## Quick Reference

Model:

```python
model = ITS_GRU(
    input_cols=input_cols,
    time_col="index",
    cost_functions={"mse": MSECostFunctionDecoder()},
    cost_cols=input_cols,
    hidden_dim=64,
    ts_decoder_type=TSDecoderType.GRU,
)
```

Train:

```python
model.train_model(
    data=df,
    window_size=40,
    window_stride=5,
    optimizer_class=torch.optim.AdamW,
    optimizer_base_params={"lr": 1e-4, "weight_decay": 2e-3},
    test_ratio=0.2,
    epochs=80,
    patience=15,
    scaler=RobustScaler,
    head_window_size=20,
    batch_size=64,
)
```

Test:

```python
result = model.test_model(
    data=df,
    window_size=40,
    window_stride=5,
    scaler=RobustScaler,
    head_window_size=20,
)
```

Predict:

```python
forecast = model.predict(
    prediction_type=PredictionType.PREDICT,
    data=df,
    window_size=40,
    window_stride=5,
    scaler=RobustScaler,
    head_window_size=20,
)
```
