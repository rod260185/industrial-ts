from __future__ import annotations

from abc import ABC
from contextlib import nullcontext
import copy
import datetime
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any,Callable
from pandas import DataFrame
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset,TensorDataset

from .encoders.patchtst import PatchTSTEncoder as _PatchTSTEncoder
from .decoders.transformer import resolve_transformer_nhead,TransformerTimeDecoder as _TransformerTimeDecoder
from .decoders.ode_jump import ODEJumpDecoder as _ODEJumpDecoder
from .decoders.lstm import LSTMDecoder as _LSTMDecoder
from .decoders.gru import GRUDecoder as _GRUDecoder, FutureGRUDecoder as _FutureGRUDecoder

from ..collections import tensors_dict
from ..cost_functions import CostFunction
from ..confs import TSDecoderType,PredictionType,EncoderType

class _BaseEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.in_channels * 2, self.hidden_dim * 2),
            activation,
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )     

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.encoder(x)
   

class BaseIndustrialTSModel(nn.Module,ABC):
    """
    Abstract base class for PyTorch models in the toolkit.
    It defines the minimum contract and reusable default methods.
    """

    def __init__(
            self, 
            input_cols: list, 
            time_col: str,
            cost_functions: dict[str,CostFunction],
            status_cols: list = None, 
            context_cols: list = None,
            cost_cols: list = None,
            timestamp_scale: float = 3600.0,
            hidden_dim: int = 32,
            diffusion_num_steps: int = 1000,
            status_pred_window: float = 1.0,
            lambda_time_embedding_dim: int = 16,
            encoder_type: EncoderType = EncoderType.MLP,
            ts_decoder_type: TSDecoderType = TSDecoderType.GRU,
            activation: dict[str, nn.Module | type[nn.Module] | Callable[[], nn.Module]] | nn.Module | type[nn.Module] | Callable[[], nn.Module] | None = None,
            limit_events: tuple[float, float] | None = None,
            events_time_embedding_dim: int = 16,
            feature_limits: dict[str, tuple[float, float]] | tuple[float, float] = None,
            seed: int | None = None,
            dtype: torch.dtype | str = torch.float32,
            **kwargs
            ) -> None:
        if seed is not None:
            self.set_seed(seed)
        super().__init__()
        #Must me implemented by subclass
        self.ts_encoder = None
        self.ts_decoder = None
        ################################
        # Model parameters that can be overridden at training / test time (e.g. for ablations) are set in `_init_model_params` and should be initialized to None here.
        self.ts_decoder_params = None
        self.loss_params = None
        self.force_irregular_timestep = None
        self.force_irregular_timestep_max_drop = None
        self.force_irregular_timestep_test = None
        self.rebuild_timeseries = None
        self.rebuild_timeseries_test = None
        self.rebuild_timeseries_max_drop = None
        self.rebuild_timeseries_max_drop_step = None
        self.test_repetition = None
        ################################
        self.activation = activation
        self.input_cols = input_cols
        self.time_col = time_col
        self.kwargs = kwargs
        self.status_cols = status_cols if status_cols is not None else []
        self.context_cols = context_cols if context_cols is not None else []
        self.cost_cols = cost_cols
        if cost_cols is not None:
            self.cost_cols_idx = [input_cols.index(cost_col) for cost_col in cost_cols]
        self.scaler = None
        self.cost_scaler = None
        self.cost_functions = cost_functions
        self.timestamp_scale = timestamp_scale
        self.limit_events = limit_events
        self.events_time_embedding_dim = int(events_time_embedding_dim)
        self.dtype = self._resolve_dtype(dtype)
        self.feature_limits = feature_limits

        if len(self.cost_functions) == 0:
            raise ValueError("`cost_functions` must not be empty.")

        self.hidden_dim = int(hidden_dim)
        self.timestamp_scale = float(timestamp_scale)
        self.num_steps = int(diffusion_num_steps)
        self.status_pred_window = float(status_pred_window)
        self.lambda_time_embedding_dim = lambda_time_embedding_dim
        if not isinstance(encoder_type, EncoderType):
            raise TypeError(f"encoder_type must be an EncoderType, got {type(encoder_type).__name__}.")
        self.encoder_type = encoder_type
        if not isinstance(ts_decoder_type, TSDecoderType):
            raise TypeError(f"ts_decoder_type must be a TSDecoderType, got {type(ts_decoder_type).__name__}.")
        self.ts_decoder_type = ts_decoder_type
        self.supported_decoders = {TSDecoderType.GRU, TSDecoderType.LSTM, TSDecoderType.ODE_JUMP, TSDecoderType.TRANSFORMER}
        self.in_channels = len(self.input_cols)
        self.static_dim = len(self.context_cols)
        self.status_dim = len(self.status_cols)
        self.output_dim = len(self.cost_cols) if self.cost_cols is not None else self.in_channels
        self._get_heads()
        decoder_requires_cost_cols = any(
            "decoder" in cf.heads and "events" not in cf.heads
            for cf in self.cost_functions.values()
        )
        if decoder_requires_cost_cols and (self.cost_cols is None or len(self.cost_cols) == 0):
            raise ValueError("Decoder cost functions require `cost_cols` to define decoder input and output columns.")
        self._make_encoder()
        self._make_static_encoder()        
        self._make_heads()

    def train_model(
            self, 
            data: DataFrame, 
            window_size: int,
            optimizer_class: type[torch.optim.Optimizer],
            optimizer_base_params: dict,
            optimizer_specific_params: dict[str, dict] = None,
            optimizer_additional_kwargs: dict = {},
            optimizer_scaler: Any = None,
            window_stride: int = 1,
            validation_ratio: float = 0., 
            test_ratio: float = 0.2,
            cross_validation: bool = False,
            cv_n_splits: int = 5,
            cv_validation_size: int | float | None = None,
            cv_train_size: int | float | None = None,
            cv_gap: int | None = None,
            cv_strategy: str = "expanding",
            batch_size: int = 32,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            epochs: int = 100,
            patience: int = 10,
            verbose: bool = True,
            best_model_path: str | Path = "best_model.pt",
            scaler: Any = None,
            log_metrics: bool = False,
            log_path: str | Path = "training_log.json",
            scheduler_params: dict = None,
            head_window_size: int = 0,
            force_irregular_timestep: bool = False,
            force_irregular_timestep_max_drop: float = 0.3,
            force_irregular_timestep_test: bool = False,
            rebuild_timeseries: bool = False,
            rebuild_timeseries_test: bool = False,
            test_repetition: int = 13,
            rebuild_timeseries_max_drop: float = 0.3,
            rebuild_timeseries_max_drop_step: float = 0.3,
            loss_params: dict = {},
            ts_decoder_params: dict = {},
            ) -> dict[str, Any] | None:
        self._init_model_params(
            force_irregular_timestep=force_irregular_timestep,
            force_irregular_timestep_max_drop=force_irregular_timestep_max_drop,
            force_irregular_timestep_test=force_irregular_timestep_test,
            rebuild_timeseries=rebuild_timeseries,
            rebuild_timeseries_test=rebuild_timeseries_test,
            test_repetition=test_repetition,
            rebuild_timeseries_max_drop=rebuild_timeseries_max_drop,
            rebuild_timeseries_max_drop_step=rebuild_timeseries_max_drop_step,
            ts_decoder_params=ts_decoder_params,
            loss_params=loss_params,
        )
        if cross_validation:
            return self._train_model_time_series_cv(
                data=data,
                window_size=window_size,
                optimizer_class=optimizer_class,
                optimizer_base_params=optimizer_base_params,
                optimizer_specific_params=optimizer_specific_params,
                optimizer_additional_kwargs=optimizer_additional_kwargs,
                optimizer_scaler=optimizer_scaler,
                window_stride=window_stride,
                validation_ratio=validation_ratio,
                batch_size=batch_size,
                device=device,
                epochs=epochs,
                patience=patience,
                verbose=verbose,
                best_model_path=best_model_path,
                scaler=scaler,
                log_metrics=log_metrics,
                log_path=log_path,
                scheduler_params=scheduler_params,
                head_window_size=head_window_size,
                cv_n_splits=cv_n_splits,
                cv_validation_size=cv_validation_size,
                cv_train_size=cv_train_size,
                cv_gap=cv_gap,
                cv_strategy=cv_strategy,
            )

        def _apply_loss_to_metrics(loss_dict: dict[str, float], metrics_dict: dict[str, dict[str, float]]) -> None:
            for k, v in loss_dict.items():
                if k in metrics_dict:
                    metrics_dict[k]['loss'] = v
                else:
                    metrics_dict[k] = {'loss': v}
        device = self._move_model_to_device(device)
        optimizer = self._make_optimizer(optimizer_class, optimizer_base_params, optimizer_specific_params, optimizer_additional_kwargs)
        dataset, train_idx, val_idx, test_idx, train_loader, val_loader, test_loader = self._prepare_dataset_splits(
            data=data,
            window_size=window_size,
            window_stride=window_stride,
            head_window_size=head_window_size,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            batch_size=batch_size,
            scaler=scaler,
            allow_full_test_ratio=False,
        )
        best_score = float("inf"); best_epoch = 0; wait = patience
        steps_per_epoch = max(len(train_loader), 1)
        self.total_steps = max(epochs * steps_per_epoch, 1)
        scheduler = None
        if scheduler_params is not None:
            scheduler = self._make_scheduler(optimizer, scheduler_params)
        if scheduler is not None:
            has_scheduler = True
        else:
            has_scheduler = False
        if log_metrics:
            log = []
        else:
            log = None
        for ep in range(1, epochs + 1):
            epoch_start = datetime.datetime.now()
            self._set_current_epoch(ep)
            self.train()
            train_loss = {k:[0.0,0.0,self.cost_functions[k].training_ratio] for k in self.cost_functions.keys() if self.cost_functions[k].training_ratio > 0}
            train_metrics_epoch = []
            val_metrics_epoch = []
            test_metrics_epoch = []
            for batch in train_loader:
                batch = tensors_dict(dict(zip(dataset.keys, batch)))
                batch = batch.move_to_device(device)
                optimizer.zero_grad()
                loss,loss_dict,train_metrics_batch = self._training_step(batch,log_metrics)
                self._check_finite_loss(loss)
                train_metrics_epoch.extend(train_metrics_batch)
                if optimizer_scaler is not None:
                    optimizer_scaler.scale(loss).backward()
                    optimizer_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    prev_scale = optimizer_scaler.get_scale()
                    optimizer_scaler.step(optimizer)
                    optimizer_scaler.update()
                    new_scale = optimizer_scaler.get_scale()
                    stepped = (new_scale >= prev_scale)
                    if has_scheduler and stepped:
                        scheduler.step()     # safe now (optimizer.step() happened on this batch)
                else:
                    loss.backward()
                    optimizer.step()
                    if has_scheduler:
                        scheduler.step()
                for k, loss_result in loss_dict.items():
                    train_loss[k][0] += loss_result[0]
                    train_loss[k][1] += loss_result[1]

            if val_loader is not None:
                val_loss, val_metrics_epoch = self._evaluate_split_loader(val_loader, dataset, device, log_metrics)

            test_loss, test_metrics_epoch = self._evaluate_split_loader(test_loader, dataset, device, log_metrics)
            
            if verbose or log_metrics:
                train_loss_means = self._log_means(train_loss)
                val_loss_means = self._log_means(val_loss) if val_loader is not None else None
                test_loss_means = self._log_means(test_loss)
            if verbose:
                self._log_training_progress(
                    ep,
                    train_loss_means,
                    val_loss_means,
                    test_loss_means,
                    epoch_start=epoch_start,
                )
            if log_metrics:
                train_metrics_dict = self._process_train_metrics(train_metrics_epoch)
                val_metrics_dict = self._process_val_metrics(val_metrics_epoch) if val_loader is not None else None
                test_metrics_dict = self._process_val_metrics(test_metrics_epoch)
                _apply_loss_to_metrics(train_loss_means, train_metrics_dict)
                if val_metrics_dict is not None:
                    _apply_loss_to_metrics(val_loss_means, val_metrics_dict)
                _apply_loss_to_metrics(test_loss_means, test_metrics_dict)
                log += [
                    {
                        "epoch": ep,
                        "epoch_time": (datetime.datetime.now() - epoch_start).total_seconds(),
                        "train_metrics": train_metrics_dict,
                        "val_metrics": val_metrics_dict if val_loader is not None else None,
                        "test_metrics": test_metrics_dict,
                    }
                ]

            if val_loader is not None:
                score = self._get_score_from_loss_dict(val_loss)
                if score < best_score:
                    best_score = score
                    best_epoch = ep
                    wait = patience
                    self.save_weights(best_model_path)
                else:
                    wait -= 1
                    if wait == 0:
                        if verbose:
                            print(f"Early stopping at epoch {ep} (best epoch was {best_epoch} with val_avg {best_score:.6f})")
                        break
            else:
                score = self._get_score_from_loss_dict(test_loss)
                if score < best_score:
                    best_score = score
                    best_epoch = ep
                    wait = patience
                    self.save_weights(best_model_path)
                else:
                    wait -= 1
                    if wait == 0:
                        if verbose:
                            print(f"Early stopping at epoch {ep} (best epoch was {best_epoch} with test_avg {best_score:.6f})")
                        break
        if log_metrics and log:
            self._log_metrics(log, log_path)
        if verbose and best_epoch > 0:
            score_split = "val" if val_loader is not None else "test"
            print(f"Best epoch {best_epoch:04d} | {score_split}_avg:{best_score:.6f}")

    @staticmethod
    def _apply_loss_to_metrics(loss_dict: dict[str, float] | None, metrics_dict: dict[str, dict[str, float]] | None) -> None:
        if loss_dict is None or metrics_dict is None:
            return
        for k, v in loss_dict.items():
            if k in metrics_dict:
                metrics_dict[k]["loss"] = v
            else:
                metrics_dict[k] = {"loss": v}

    @staticmethod
    def _split_log_summary(idx: list[int]) -> dict[str, int | None]:
        return {
            "start": idx[0] if idx else None,
            "end": idx[-1] if idx else None,
            "count": len(idx),
        }

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return math.isfinite(float(value))
        return False

    @classmethod
    def _flatten_numeric_metrics(
        cls,
        metrics: Any,
        prefix: tuple[str, ...] = (),
    ) -> dict[tuple[str, ...], float]:
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            flat: dict[tuple[str, ...], float] = {}
            for key, value in metrics.items():
                flat.update(cls._flatten_numeric_metrics(value, prefix + (str(key),)))
            return flat
        if cls._is_finite_number(metrics) and prefix:
            return {prefix: float(metrics)}
        return {}

    @staticmethod
    def _set_nested_summary(out: dict[str, Any], path: tuple[str, ...], value: dict[str, float | int]) -> None:
        node = out
        for part in path[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[path[-1]] = value

    @classmethod
    def _summarize_numeric_metric_trees(cls, metric_trees: list[Any]) -> dict[str, Any]:
        path_values: dict[tuple[str, ...], list[float]] = {}
        for tree in metric_trees:
            for path, value in cls._flatten_numeric_metrics(tree).items():
                path_values.setdefault(path, []).append(value)

        summary: dict[str, Any] = {}
        for path, values in sorted(path_values.items()):
            cls._set_nested_summary(
                summary,
                path,
                {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "count": len(values),
                },
            )
        return summary

    @staticmethod
    def _fold_model_path(best_model_path: str | Path, fold_number: int) -> Path:
        path = Path(best_model_path)
        suffix = path.suffix
        return path.with_name(f"{path.stem}_fold{fold_number:02d}{suffix}")

    @staticmethod
    def _resolve_cv_size(
        value: int | float | None,
        n: int,
        *,
        default: int | None,
        name: str,
    ) -> int | None:
        if value is None:
            return default
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an int, float ratio, or None.")
        if isinstance(value, (float, np.floating)):
            ratio = float(value)
            if 0.0 < ratio < 1.0:
                return max(1, int(math.floor(n * ratio)))
            if not ratio.is_integer():
                raise ValueError(f"{name} as a float must be a ratio in (0, 1) or an integer-like value.")
            value = int(ratio)
        size = int(value)
        if size < 1:
            raise ValueError(f"{name} must be >= 1.")
        return size

    def _make_time_series_cv_folds(
        self,
        n_windows: int,
        cv_n_splits: int,
        cv_validation_size: int | float | None,
        cv_train_size: int | float | None,
        cv_gap: int | None,
        cv_strategy: str,
        validation_ratio: float,
    ) -> list[dict[str, Any]]:
        if cv_n_splits < 2:
            raise ValueError("cv_n_splits must be >= 2.")
        if n_windows < 2:
            raise ValueError("Cross-validation requires at least two windows.")

        strategy = str(cv_strategy).lower().strip()
        if strategy not in {"expanding", "rolling"}:
            raise ValueError("cv_strategy must be either 'expanding' or 'rolling'.")

        gap = self.window_size + self.head_window_size - 1 if cv_gap is None else int(cv_gap)
        if gap < 0:
            raise ValueError("cv_gap must be >= 0.")

        max_auto_val_size = (n_windows - gap - 1) // cv_n_splits
        if max_auto_val_size < 1:
            raise ValueError(
                "Not enough windows for time-series cross-validation after applying the purge gap. "
                f"n_windows={n_windows}, cv_n_splits={cv_n_splits}, cv_gap={gap}."
            )

        val_size_default = None
        if cv_validation_size is None:
            if validation_ratio > 0.0:
                val_size_default = max(1, int(math.floor(n_windows * validation_ratio)))
            else:
                val_size_default = max(1, n_windows // (cv_n_splits + 1))
            val_size_default = min(val_size_default, max_auto_val_size)
        val_size = self._resolve_cv_size(
            cv_validation_size,
            n_windows,
            default=val_size_default,
            name="cv_validation_size",
        )
        if val_size is None:
            raise ValueError("Could not resolve cv_validation_size.")

        train_size = self._resolve_cv_size(
            cv_train_size,
            n_windows,
            default=None,
            name="cv_train_size",
        )
        if strategy == "rolling" and train_size is None:
            raise ValueError("cv_strategy='rolling' requires cv_train_size.")

        total_validation = cv_n_splits * val_size
        first_val_start = n_windows - total_validation
        if first_val_start <= gap:
            raise ValueError(
                "Not enough windows for time-series cross-validation. "
                f"n_windows={n_windows}, cv_n_splits={cv_n_splits}, "
                f"cv_validation_size={val_size}, cv_gap={gap}."
            )

        folds: list[dict[str, Any]] = []
        for fold_idx in range(cv_n_splits):
            val_start = first_val_start + fold_idx * val_size
            val_end = val_start + val_size
            train_end = val_start - gap
            if strategy == "rolling":
                assert train_size is not None
                train_start = max(0, train_end - train_size)
            else:
                train_start = 0

            if train_end <= train_start:
                raise ValueError(
                    f"Fold {fold_idx + 1} has an empty training split. "
                    f"train_start={train_start}, train_end={train_end}."
                )
            if val_end > n_windows:
                raise ValueError(
                    f"Fold {fold_idx + 1} validation split exceeds dataset size. "
                    f"val_end={val_end}, n_windows={n_windows}."
                )

            folds.append(
                {
                    "fold": fold_idx + 1,
                    "train_idx": list(range(train_start, train_end)),
                    "val_idx": list(range(val_start, val_end)),
                    "gap": gap,
                }
            )
        return folds

    def _prepare_dataset_splits_from_indices(
        self,
        data: DataFrame,
        window_size: int,
        window_stride: int,
        head_window_size: int | None,
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
        batch_size: int,
        scaler: callable | None,
    ) -> tuple[TensorDataset, DataLoader, DataLoader | None, DataLoader | None]:
        if head_window_size is None:
            head_window_size = 0
        self.window_size = window_size
        self.head_window_size = head_window_size
        dataset = self._make_dataset(
            data,
            window_size=window_size,
            window_stride=window_stride,
            head_window_size=head_window_size,
        )
        if len(dataset) == 0:
            raise ValueError("Received empty dataset after _make_dataset.")
        for split_name, idx in (("train_idx", train_idx), ("val_idx", val_idx), ("test_idx", test_idx)):
            if any(i < 0 or i >= len(dataset) for i in idx):
                raise ValueError(f"{split_name} contains indices outside dataset bounds.")
        if len(train_idx) == 0:
            raise ValueError("Training split must not be empty.")

        self._preprocess_dataset(dataset, scaler, fit_idx=train_idx)

        train_loader = self._make_trainloader(dataset, train_idx, batch_size)
        val_loader = self._make_dataloader(dataset, val_idx, batch_size) if len(val_idx) > 0 else None
        test_loader = self._make_dataloader(dataset, test_idx, batch_size) if len(test_idx) > 0 else None
        return dataset, train_loader, val_loader, test_loader

    def _run_training_loop(
        self,
        dataset: TensorDataset,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        optimizer_class: type[torch.optim.Optimizer],
        optimizer_base_params: dict,
        optimizer_specific_params: dict[str, dict] | None,
        optimizer_additional_kwargs: dict,
        optimizer_scaler: Any,
        scheduler_params: dict | None,
        epochs: int,
        patience: int,
        device: torch.device,
        verbose: bool,
        best_model_path: str | Path | None,
        log_metrics: bool,
        *,
        fold_label: str | None = None,
        collect_epoch_log: bool = False,
    ) -> dict[str, Any]:
        device = self._move_model_to_device(device)
        optimizer = self._make_optimizer(
            optimizer_class,
            optimizer_base_params,
            optimizer_specific_params,
            optimizer_additional_kwargs,
        )
        best_score = float("inf")
        best_epoch = 0
        best_record: dict[str, Any] | None = None
        wait = patience
        steps_per_epoch = max(len(train_loader), 1)
        self.total_steps = max(epochs * steps_per_epoch, 1)
        scheduler = self._make_scheduler(optimizer, scheduler_params) if scheduler_params is not None else None
        has_scheduler = scheduler is not None
        epoch_log: list[dict[str, Any]] = []
        score_split = "val" if val_loader is not None else "test"
        if val_loader is None and test_loader is None:
            raise ValueError("Training loop requires either a validation loader or a test loader for model selection.")

        for ep in range(1, epochs + 1):
            epoch_start = datetime.datetime.now()
            self._set_current_epoch(ep)
            self.train()
            train_loss = {
                k: [0.0, 0.0, self.cost_functions[k].training_ratio]
                for k in self.cost_functions.keys()
                if self.cost_functions[k].training_ratio > 0
            }
            train_metrics_epoch = []
            for batch in train_loader:
                batch = tensors_dict(dict(zip(dataset.keys, batch)))
                batch = batch.move_to_device(device)
                optimizer.zero_grad()
                loss, loss_dict, train_metrics_batch = self._training_step(batch, log_metrics)
                self._check_finite_loss(loss)
                train_metrics_epoch.extend(train_metrics_batch)
                if optimizer_scaler is not None:
                    optimizer_scaler.scale(loss).backward()
                    optimizer_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    prev_scale = optimizer_scaler.get_scale()
                    optimizer_scaler.step(optimizer)
                    optimizer_scaler.update()
                    new_scale = optimizer_scaler.get_scale()
                    stepped = new_scale >= prev_scale
                    if has_scheduler and stepped:
                        scheduler.step()
                else:
                    loss.backward()
                    optimizer.step()
                    if has_scheduler:
                        scheduler.step()
                for k, loss_result in loss_dict.items():
                    train_loss[k][0] += loss_result[0]
                    train_loss[k][1] += loss_result[1]

            val_loss = None
            val_metrics_epoch = []
            if val_loader is not None:
                val_loss, val_metrics_epoch = self._evaluate_split_loader(val_loader, dataset, device, log_metrics)

            test_loss = None
            test_metrics_epoch = []
            if test_loader is not None:
                test_loss, test_metrics_epoch = self._evaluate_split_loader(test_loader, dataset, device, log_metrics)

            train_loss_means = self._log_means(train_loss)
            val_loss_means = self._log_means(val_loss) if val_loss is not None else None
            test_loss_means = self._log_means(test_loss) if test_loss is not None else None

            if verbose:
                self._log_training_progress(
                    ep,
                    train_loss_means,
                    val_loss_means,
                    test_loss_means,
                    epoch_start=epoch_start,
                    prefix=fold_label,
                )

            train_metrics_dict = None
            val_metrics_dict = None
            test_metrics_dict = None
            if log_metrics:
                train_metrics_dict = self._process_train_metrics(train_metrics_epoch)
                val_metrics_dict = self._process_val_metrics(val_metrics_epoch) if val_loss is not None else None
                test_metrics_dict = self._process_val_metrics(test_metrics_epoch) if test_loss is not None else None
                self._apply_loss_to_metrics(train_loss_means, train_metrics_dict)
                self._apply_loss_to_metrics(val_loss_means, val_metrics_dict)
                self._apply_loss_to_metrics(test_loss_means, test_metrics_dict)

            epoch_entry = {
                "epoch": ep,
                "epoch_time": (datetime.datetime.now() - epoch_start).total_seconds(),
                "train_loss": train_loss_means,
                "val_loss": val_loss_means,
                "test_loss": test_loss_means,
            }
            if log_metrics:
                epoch_entry.update(
                    {
                        "train_metrics": train_metrics_dict,
                        "val_metrics": val_metrics_dict,
                        "test_metrics": test_metrics_dict,
                    }
                )
            if collect_epoch_log:
                epoch_log.append(epoch_entry)

            selection_loss = val_loss if val_loss is not None else test_loss
            score = self._get_score_from_loss_dict(selection_loss)
            if score < best_score:
                best_score = score
                best_epoch = ep
                best_record = epoch_entry
                wait = patience
                if best_model_path is not None:
                    self.save_weights(best_model_path)
            else:
                wait -= 1
                if wait == 0:
                    if verbose:
                        print(
                            f"Early stopping at epoch {ep} "
                            f"(best epoch was {best_epoch} with {score_split}_avg {best_score:.6f})"
                        )
                    break

        if verbose and best_epoch > 0:
            prefix = f"{fold_label} | " if fold_label else ""
            print(f"{prefix}Best epoch {best_epoch:04d} | {score_split}_avg:{best_score:.6f}")

        return {
            "epochs": epoch_log,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_score_split": score_split,
            "best_record": best_record,
            "best_model_path": str(best_model_path) if best_model_path is not None else None,
        }

    def _train_model_time_series_cv(
        self,
        data: DataFrame,
        window_size: int,
        optimizer_class: type[torch.optim.Optimizer],
        optimizer_base_params: dict,
        optimizer_specific_params: dict[str, dict] | None,
        optimizer_additional_kwargs: dict,
        optimizer_scaler: Any,
        window_stride: int,
        validation_ratio: float,
        batch_size: int,
        device: torch.device,
        epochs: int,
        patience: int,
        verbose: bool,
        best_model_path: str | Path,
        scaler: Any,
        log_metrics: bool,
        log_path: str | Path,
        scheduler_params: dict | None,
        head_window_size: int,
        cv_n_splits: int,
        cv_validation_size: int | float | None,
        cv_train_size: int | float | None,
        cv_gap: int | None,
        cv_strategy: str,
    ) -> dict[str, Any]:
        if head_window_size is None:
            head_window_size = 0
        if head_window_size < 0:
            raise ValueError("head_window_size must be at least 0 to ensure valid windows.")
        if "decoder" in self.heads and head_window_size < 1:
            raise ValueError("Decoder cost functions require head_window_size >= 1.")
        if "events" in self.heads and head_window_size < 1:
            raise ValueError("EventsCostFunction requires head_window_size >= 1.")
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.window_size = window_size
        self.head_window_size = head_window_size
        planning_dataset = self._make_dataset(
            data,
            window_size=window_size,
            window_stride=window_stride,
            head_window_size=head_window_size,
        )
        n_windows = len(planning_dataset)
        folds = self._make_time_series_cv_folds(
            n_windows=n_windows,
            cv_n_splits=cv_n_splits,
            cv_validation_size=cv_validation_size,
            cv_train_size=cv_train_size,
            cv_gap=cv_gap,
            cv_strategy=cv_strategy,
            validation_ratio=validation_ratio,
        )
        initial_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        scaler_state = copy.deepcopy(optimizer_scaler.state_dict()) if optimizer_scaler is not None and hasattr(optimizer_scaler, "state_dict") else None

        fold_payloads: list[dict[str, Any]] = []
        for fold in folds:
            fold_number = int(fold["fold"])
            self.load_state_dict(initial_state, strict=True)
            if optimizer_scaler is not None and scaler_state is not None and hasattr(optimizer_scaler, "load_state_dict"):
                optimizer_scaler.load_state_dict(copy.deepcopy(scaler_state))

            train_idx = fold["train_idx"]
            val_idx = fold["val_idx"]
            dataset, train_loader, val_loader, _ = self._prepare_dataset_splits_from_indices(
                data=data,
                window_size=window_size,
                window_stride=window_stride,
                head_window_size=head_window_size,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=[],
                batch_size=batch_size,
                scaler=scaler,
            )
            fold_model_path = self._fold_model_path(best_model_path, fold_number)
            fold_label = f"Fold {fold_number}/{cv_n_splits}"
            if verbose:
                print(
                    f"{fold_label} | "
                    f"train:{len(train_idx)} windows "
                    f"val:{len(val_idx)} windows "
                    f"gap:{fold['gap']}"
                )

            fold_result = self._run_training_loop(
                dataset=dataset,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,
                optimizer_class=optimizer_class,
                optimizer_base_params=optimizer_base_params,
                optimizer_specific_params=optimizer_specific_params,
                optimizer_additional_kwargs=optimizer_additional_kwargs,
                optimizer_scaler=optimizer_scaler,
                scheduler_params=scheduler_params,
                epochs=epochs,
                patience=patience,
                device=device,
                verbose=verbose,
                best_model_path=fold_model_path,
                log_metrics=log_metrics,
                fold_label=fold_label,
                collect_epoch_log=True,
            )
            fold_payloads.append(
                {
                    "fold": fold_number,
                    "split": {
                        "train": self._split_log_summary(train_idx),
                        "validation": self._split_log_summary(val_idx),
                        "test": None,
                        "gap": int(fold["gap"]),
                    },
                    **fold_result,
                }
            )

        best_epochs = [int(fold["best_epoch"]) for fold in fold_payloads if int(fold["best_epoch"]) > 0]
        best_scores = [float(fold["best_score"]) for fold in fold_payloads if math.isfinite(float(fold["best_score"]))]
        best_records = [fold.get("best_record") for fold in fold_payloads if isinstance(fold.get("best_record"), dict)]
        best_validation_losses = [
            record.get("val_loss")
            for record in best_records
            if isinstance(record.get("val_loss"), dict)
        ]
        best_validation_metrics = [
            record.get("val_metrics")
            for record in best_records
            if isinstance(record.get("val_metrics"), dict)
        ]
        summary = {
            "best_epochs": best_epochs,
            "best_epoch_mean": float(np.mean(best_epochs)) if best_epochs else None,
            "best_epoch_median": float(np.median(best_epochs)) if best_epochs else None,
            "validation_scores": best_scores,
            "validation_score_mean": float(np.mean(best_scores)) if best_scores else None,
            "validation_score_median": float(np.median(best_scores)) if best_scores else None,
            "best_validation_loss_summary": self._summarize_numeric_metric_trees(best_validation_losses),
            "best_validation_metrics_summary": self._summarize_numeric_metric_trees(best_validation_metrics),
            "n_splits": cv_n_splits,
            "selection_criterion": "validation_loss.avg",
        }
        payload = {
            "mode": "time_series_cross_validation",
            "params": {
                "cv_n_splits": cv_n_splits,
                "cv_validation_size": cv_validation_size,
                "cv_train_size": cv_train_size,
                "cv_gap": cv_gap,
                "cv_strategy": cv_strategy,
                "window_size": window_size,
                "window_stride": window_stride,
                "head_window_size": head_window_size,
                "validation_ratio_used_for_default_cv_validation_size": validation_ratio,
                "test_ratio": None,
            },
            "summary": summary,
            "folds": fold_payloads,
        }
        self._log_metrics(payload, log_path)
        self.load_state_dict(initial_state, strict=True)
        if verbose:
            print(
                "Cross-validation summary | "
                f"best_epoch_mean:{summary['best_epoch_mean']} "
                f"best_epoch_median:{summary['best_epoch_median']} "
                f"validation_score_mean:{summary['validation_score_mean']} "
                f"validation_score_median:{summary['validation_score_median']}"
            )
        return payload

    def test_model(self,
            data: DataFrame,
            window_size: int,
            window_stride: int = 1,
            validation_ratio: float = 0.0,
            test_ratio: float = 1.0,
            batch_size: int = 32,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            verbose: bool = True,
            log_test_metrics: bool = False,
            log_path: str | Path = "test_metrics.json",
            head_window_size: int | None = None,
            scaler: Any = None,
            force_irregular_timestep: bool = False,
            force_irregular_timestep_max_drop: float = 0.3,
            force_irregular_timestep_test: bool = False,
            rebuild_timeseries: bool = False,
            rebuild_timeseries_test: bool = False,
            test_repetition: int = 13,
            rebuild_timeseries_max_drop: float = 0.3,
            rebuild_timeseries_max_drop_step: float = 0.3,
            loss_params: dict = {},
            ts_decoder_params: dict = {},
            ) -> dict[str, Any]:
        """
        Evaluate the model using the same split/evaluation process as `train_model`.
        By default, the entire dataset is evaluated as test (`test_ratio=1.0`).
        """
        self._init_model_params(
            force_irregular_timestep=force_irregular_timestep,
            force_irregular_timestep_max_drop=force_irregular_timestep_max_drop,
            force_irregular_timestep_test=force_irregular_timestep_test,
            rebuild_timeseries=rebuild_timeseries,
            rebuild_timeseries_test=rebuild_timeseries_test,
            test_repetition=test_repetition,
            rebuild_timeseries_max_drop=rebuild_timeseries_max_drop,
            rebuild_timeseries_max_drop_step=rebuild_timeseries_max_drop_step,
            ts_decoder_params=ts_decoder_params,
            loss_params=loss_params,
        )
        dataset, train_idx, val_idx, test_idx, _, val_loader, test_loader = self._prepare_dataset_splits(
            data=data,
            window_size=window_size,
            window_stride=window_stride,
            head_window_size=head_window_size,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            batch_size=batch_size,
            scaler=scaler,
            allow_full_test_ratio=True,
        )
        if test_loader is None:
            raise ValueError(
                "test_model: empty test split after applying validation_ratio/test_ratio. "
                f"validation_ratio={validation_ratio}, test_ratio={test_ratio}."
            )
        device = self._move_model_to_device(device)
        val_loss, val_metrics = self._evaluate_split_loader(val_loader, dataset, device, log_test_metrics)
        test_loss, test_metrics = self._evaluate_split_loader(test_loader, dataset, device, log_test_metrics)
        val_score = self._get_score_from_loss_dict(val_loss) if val_loader is not None else None
        score = self._get_score_from_loss_dict(test_loss)
        val_loss_means = self._log_means(val_loss) if val_loader is not None else None
        test_loss_means = self._log_means(test_loss)

        if verbose:
            self._log_test_progress(
                val_loss=val_loss_means,
                test_loss=test_loss_means,
                val_score=val_score,
                test_score=score,
            )

        processed_val_metrics = self._process_val_metrics(val_metrics) if val_metrics else None
        processed_test_metrics = self._process_val_metrics(test_metrics) if test_metrics else None
        if log_test_metrics and (processed_val_metrics is not None or processed_test_metrics is not None):
            payload: Any
            if processed_val_metrics is None:
                payload = {
                    "mode": "test",
                    "validation_ratio": validation_ratio,
                    "test_ratio": test_ratio,
                    "test_loss": test_loss_means,
                    "test_score": score,
                    "test_metrics": processed_test_metrics,
                }
            else:
                payload = {
                    "mode": "test",
                    "validation_ratio": validation_ratio,
                    "test_ratio": test_ratio,
                    "val_loss": val_loss_means,
                    "val_score": val_score,
                    "val_metrics": processed_val_metrics,
                    "test_loss": test_loss_means,
                    "test_score": score,
                    "test_metrics": processed_test_metrics,
                }
            self._log_metrics(payload, log_path)

        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "val_loss": val_loss if val_loader is not None else None,
            "val_score": val_score,
            "val_metrics": val_metrics if val_loader is not None else None,
            "test_loss": test_loss,
            "test_score": score,
            "test_metrics": test_metrics
        }
    
    def predict(
            self,
            prediction_type: PredictionType | str,
            data: DataFrame,
            window_size: int,
            window_stride: int = 1,
            batch_size: int = 32,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            verbose: bool = True,
            scaler: Any = None,
            scaler_data: DataFrame | None = None,
            scaler_ratio: float | tuple[float, float] | tuple[float, float, float] | list[float] | dict[str, float] | None = 1.0,
            head_window_size: int | None = None,
            force_irregular_timestep: bool = False,
            force_irregular_timestep_max_drop: float = 0.3,
            force_irregular_timestep_test: bool = False,
            rebuild_timeseries: bool = False,
            rebuild_timeseries_test: bool = False,
            test_repetition: int = 13,
            rebuild_timeseries_max_drop: float = 0.3,
            rebuild_timeseries_max_drop_step: float = 0.3,
            loss_params: dict = {},
            ts_decoder_params: dict = {},
            params: dict = {}
            ) -> Any:
        """
        Run inference on the full DataFrame.
        When `scaler` is provided, it may be fit on `scaler_data` using the
        train-equivalent component described by `scaler_ratio`.
        Returns batch outputs concatenated into a single structure.
        """
        self._init_model_params(
            force_irregular_timestep=force_irregular_timestep,
            force_irregular_timestep_max_drop=force_irregular_timestep_max_drop,
            force_irregular_timestep_test=force_irregular_timestep_test,
            rebuild_timeseries=rebuild_timeseries,
            rebuild_timeseries_test=rebuild_timeseries_test,
            test_repetition=test_repetition,
            rebuild_timeseries_max_drop=rebuild_timeseries_max_drop,
            rebuild_timeseries_max_drop_step=rebuild_timeseries_max_drop_step,
            ts_decoder_params=ts_decoder_params,
            loss_params=loss_params,
        )
        prediction_type = self._normalize_prediction_type(prediction_type)
        self._validate_prediction_type(prediction_type)
        prediction_heads = self._prediction_type_heads(prediction_type)
        prediction_method = self._get_prediction_method(prediction_type)

        if head_window_size is None:
            head_window_size = 0
        if head_window_size < 0:
            raise ValueError("head_window_size must be at least 0.")
        if self._prediction_type_requires_head_window(prediction_type) and head_window_size < 1:
            raise ValueError(f"predict requires head_window_size >= 1 for {prediction_type.name}.")
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        original_heads = self.heads
        self.heads = prediction_heads
        try:
            self.window_size = window_size
            self.head_window_size = head_window_size
            dataset = self._make_dataset(
                data,
                window_size=window_size,
                window_stride=window_stride,
                head_window_size=head_window_size,
            )
            full_idx = list(range(len(dataset)))
            if len(full_idx) == 0:
                raise ValueError("predict: empty dataset after _make_dataset.")

            scaler_dataset = None
            scaler_fit_idx = None
            if scaler is not None:
                scaler_source = data if scaler_data is None else scaler_data
                scaler_dataset = self._make_dataset(
                    scaler_source,
                    window_size=window_size,
                    window_stride=window_stride,
                    head_window_size=head_window_size,
                )
                scaler_fit_idx = self._resolve_scaler_fit_idx(scaler_dataset, scaler_ratio)

            self._preprocess_dataset(
                dataset,
                scaler,
                fit_idx=scaler_fit_idx,
                scaler_dataset=scaler_dataset,
            )
            loader = self._make_dataloader(dataset, full_idx, batch_size)
            device = self._move_model_to_device(device)
            self.eval()

            outputs = []
            with torch.no_grad():
                for batch in loader:
                    batch = tensors_dict(dict(zip(dataset.keys, batch)))
                    batch = batch.move_to_device(device)
                    pred = prediction_method(batch, params)
                    outputs.append(self._detach_to_cpu(pred))

            predictions = self._concat_predictions(outputs)
            if verbose:
                scaler_windows = len(scaler_fit_idx) if scaler_fit_idx is not None else 0
                scaler_msg = f" | scaler_fit_windows:{scaler_windows}" if scaler is not None else ""
                print(f"Predict {prediction_type.name}: {len(full_idx)} samples processed.{scaler_msg}")
            return predictions
        finally:
            self.heads = original_heads
    
    def _init_model_params(
            self,
            force_irregular_timestep: bool,
            force_irregular_timestep_max_drop: float,
            force_irregular_timestep_test: bool,
            rebuild_timeseries: bool,
            rebuild_timeseries_test: bool,
            test_repetition: int,
            rebuild_timeseries_max_drop: float,
            rebuild_timeseries_max_drop_step: float,
            ts_decoder_params: dict,
            loss_params: dict,
            ):
        self.ts_decoder_params = ts_decoder_params
        self.loss_params = loss_params
        self.force_irregular_timestep = force_irregular_timestep
        self.force_irregular_timestep_max_drop = force_irregular_timestep_max_drop
        self.force_irregular_timestep_test = force_irregular_timestep_test
        self.rebuild_timeseries = rebuild_timeseries
        self.rebuild_timeseries_test = rebuild_timeseries_test
        self.rebuild_timeseries_max_drop = rebuild_timeseries_max_drop
        self.rebuild_timeseries_max_drop_step = rebuild_timeseries_max_drop_step
        if test_repetition % 2 != 1:
            test_repetition += 1
        self.test_repetition = int(test_repetition)
    
    @staticmethod
    def set_seed(seed:int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True, warn_only=True)

    @staticmethod
    def reset_seed() -> None:
        random.seed(None)
        np.random.seed(None)

        torch.seed()  # reseed CPU RNG from nondeterministic source

        if torch.cuda.is_available():
            torch.cuda.seed_all()  # reseed all CUDA RNGs from nondeterministic source

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        torch.use_deterministic_algorithms(False)


    @staticmethod
    def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
        supported_dtypes = {torch.float32, torch.bfloat16, torch.float16}
        if isinstance(dtype, torch.dtype):
            if dtype not in supported_dtypes:
                valid = ", ".join(str(item).replace("torch.", "") for item in sorted(supported_dtypes, key=str))
                raise ValueError(f"Unsupported dtype={dtype!r}. Valid torch dtypes: {valid}.")
            return dtype
        if not isinstance(dtype, str):
            raise TypeError(f"dtype must be a torch.dtype or string, got {type(dtype).__name__}.")

        normalized = dtype.strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "float": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "brain16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            valid = ", ".join(sorted(aliases))
            raise ValueError(f"Unsupported dtype={dtype!r}. Valid aliases: {valid}.") from exc

    def _validate_dtype_device(self, device: torch.device | str) -> torch.device:
        device = torch.device(device)
        if self.dtype == torch.float16 and device.type == "cpu":
            raise ValueError("float16 is not supported for CPU training/inference; use float32 or bfloat16.")
        if self.dtype == torch.bfloat16 and device.type == "cuda":
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            if not torch.cuda.is_available() or not is_bf16_supported():
                raise ValueError("bfloat16 requested, but this CUDA device does not report bfloat16 support.")
        return device

    def _move_model_to_device(self, device: torch.device | str) -> torch.device:
        device = self._validate_dtype_device(device)
        self.to(device=device)
        return device

    def _autocast_context(self, device: torch.device | str):
        device = self._validate_dtype_device(device)
        if self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast(
            device_type=device.type,
            dtype=self.dtype,
            enabled=True,
        )

    def _to_model_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.dtype) if torch.is_floating_point(tensor) else tensor

    @staticmethod
    def _to_loss_dtype(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() if torch.is_floating_point(tensor) else tensor

    @staticmethod
    def _check_finite_loss(loss: torch.Tensor) -> None:
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError("Non-finite loss detected. Check dtype, learning rate, scaling, and loss weights.")

    def _make_encoder(self):
        
        if self.encoder_type == EncoderType.MLP:
            self.encoder = _BaseEncoder(
                in_channels=self.in_channels,
                hidden_dim=self.hidden_dim,
                activation=self._get_activation("encoder")
            )
        elif self.encoder_type == EncoderType.PATCHTST:
            patch_len = int(self.kwargs.get("encoder_patch_len", self.kwargs.get("patchtst_patch_len", 16)))
            patch_stride = self.kwargs.get("encoder_patch_stride", self.kwargs.get("patchtst_stride"))
            if patch_stride is None:
                patch_stride = max(1, patch_len // 2)
            self.encoder = _PatchTSTEncoder(
                input_size=self.in_channels*2,
                hidden_size=self.hidden_dim,
                patch_len=patch_len,
                stride=int(patch_stride),
                num_layers=self.kwargs.get("encoder_layers", self.kwargs.get("patchtst_layers", 2)),
                nhead=self.kwargs.get("encoder_nhead", self.kwargs.get("patchtst_nhead", 4)),
                dropout=self.kwargs.get("encoder_dropout", self.kwargs.get("patchtst_dropout", 0.0)),
                max_patches=self.kwargs.get("encoder_max_patches", 512),
                dim_feedforward=self.kwargs.get("encoder_dim_feedforward"),
                activation=self.kwargs.get("encoder_activation", "gelu"),
                norm_first=self.kwargs.get("encoder_norm_first", True),
        )

    def _make_static_encoder(self):
        if self.static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(self.static_dim, self.hidden_dim * 2),
                self._get_activation("static_encoder"),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            )
        else:
            self.static_proj = None

    @staticmethod
    def _make_activation_module(spec: nn.Module | type[nn.Module] | Callable[[], nn.Module] | None) -> nn.Module:
        if spec is None:
            return nn.GELU()
        if isinstance(spec, nn.Module):
            return copy.deepcopy(spec)
        if isinstance(spec, type) and issubclass(spec, nn.Module):
            return spec()
        if callable(spec):
            module = spec()
            if isinstance(module, nn.Module):
                return module
        raise TypeError(
            "activation entries must be nn.Module instances, nn.Module classes, "
            "zero-argument factories returning nn.Module, or None."
        )

    def _get_activation(self, name: str, fallback_name: str | None = None) -> nn.Module:
        if isinstance(self.activation, dict):
            spec = self.activation.get(name)
            if spec is None and fallback_name is not None:
                spec = self.activation.get(fallback_name)
            if spec is None:
                spec = self.activation.get("default")
        else:
            spec = self.activation
        return self._make_activation_module(spec)

    def _make_x_head(self):
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            self._get_activation("x_head"),
            nn.Linear(self.hidden_dim * 2, self.output_dim),
        )

        if 'decoder' in self.heads:
            self.ts_decoder = self._instaciate_ts_decoder(self.decoder)

    def _check_supported_decoders(self):
        if self.ts_decoder_type not in self.supported_decoders:
            raise ValueError(f"Decoder type {self.ts_decoder_type!r} is not supported by this model. Supported decoders: {self.supported_decoders}.")

    def _make_noise_head(self):   
        if 'x' not in self.heads:
            raise ValueError('NoisePredictionCostFunction must be used with another cost function that demands x head.')
        self.noise_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            self._get_activation("noise_head"),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )    

    def _make_miss_head(self):
        self.miss_head = nn.Linear(self.hidden_dim, self.in_channels)

    def _make_lambda_head(self):
        input_dim = self.hidden_dim * 2 if 'decoder' in self.heads else self.hidden_dim
        if 'decoder' in self.heads and self.lambda_time_embedding_dim > 0:
            self.lambda_time_embedding = nn.Sequential(
                nn.Linear(1, self.lambda_time_embedding_dim),
                self._get_activation("lambda_time_embedding"),
                nn.Linear(self.lambda_time_embedding_dim, self.lambda_time_embedding_dim),
                nn.LayerNorm(self.lambda_time_embedding_dim),
            )
            input_dim += self.lambda_time_embedding_dim
        else:
            self.lambda_time_embedding = None
        self.lambda_head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim // 4),
            self._get_activation("lambda_head"),
            nn.Linear(self.hidden_dim // 4, self.output_dim),
        )

    def _make_lambda_decoder_input(
        self,
        state: torch.Tensor,
        decoder_state: torch.Tensor,
        batch: tensors_dict,
    ) -> torch.Tensor:
        last_state = state[:, -1, :].unsqueeze(1).expand(-1, decoder_state.size(1), -1)
        lambda_input = torch.cat([last_state, decoder_state], dim=-1)

        if self.lambda_time_embedding is None:
            return lambda_input

        head_timestamps = batch.get("head_timestamps")
        if head_timestamps is None:
            raise ValueError("lambda_time_embedding_dim > 0 requires `head_timestamps` in the batch.")
        origin_ts = batch["timestamps"][:, -1].unsqueeze(1)
        delta_t = (head_timestamps - origin_ts).to(dtype=decoder_state.dtype).unsqueeze(-1)
        time_embedding = self.lambda_time_embedding(delta_t)
        return torch.cat([lambda_input, time_embedding], dim=-1)

    def _make_events_heads(self):
        if self.limit_events is None and self.status_dim <= 0:
            raise ValueError("Events head with Softmax requires `status_cols`.")
        if self.limit_events is not None and not self.cost_cols:
            raise ValueError("Events head with limit_events requires `cost_cols`.")
        events_dim = self.status_dim if self.limit_events is None else len(self.cost_cols)
        self.events_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self._get_activation("events_head"),
            nn.Linear(self.hidden_dim // 2, events_dim),
            nn.Softmax(dim=-1) if self.limit_events is None else nn.Sigmoid(),
        )

    def _instaciate_ts_decoder(self,decoder: Callable) -> nn.Module:
        self._check_supported_decoders()
        if self.ts_decoder_type == TSDecoderType.GRU:
            return _GRUDecoder(
                    input_size=self.output_dim,
                    hidden_size=self.hidden_dim,
                    decoder=decoder
                )
        
        if self.ts_decoder_type == TSDecoderType.LSTM:
            return _LSTMDecoder(
                input_size=self.output_dim,
                hidden_size=self.hidden_dim,
                decoder=decoder,
            )
        
        if self.ts_decoder_type == TSDecoderType.ODE_JUMP:
            return _ODEJumpDecoder(
                input_size=self.output_dim,
                hidden_size=self.hidden_dim,
                decoder=decoder,
                layernorm_mode=self.kwargs.get("ts_decoder_layernorm_mode", self.kwargs.get("layernorm_mode", "none")),
            )

        if self.ts_decoder_type == TSDecoderType.TRANSFORMER:
            decoder_autoregressive = self.kwargs.get("ts_decoder_autoregressive", True)
            nhead = resolve_transformer_nhead(
                self.hidden_dim,
                self.kwargs.get("ts_decoder_nhead", 4),
            )
            return _TransformerTimeDecoder(
                hidden_size=self.hidden_dim,
                input_size=self.output_dim,
                decoder=decoder,
                nhead=nhead,
                num_layers=self.kwargs.get("ts_decoder_layers", 2),
                dropout=self.kwargs.get("ts_decoder_dropout", self.kwargs.get("dropout", 0.0)),
                max_length=self.kwargs.get("ts_decoder_max_length", 512),
                autoregressive=decoder_autoregressive,
                causal=self.kwargs.get('ts_decoder_always_causal', False)
            )           

        if os.environ.get('dev') == 'true':
            if self.ts_decoder_type == TSDecoderType.FUTURE_GRU:
                return _FutureGRUDecoder(
                    input_size=self.output_dim,
                    hidden_size=self.hidden_dim,
                    decoder=decoder
                )             

    def _make_vae_x_heads(self):
        self.vae_latent = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            self._get_activation("vae_latent", "vae"),
            nn.Linear(self.hidden_dim* 4, self.hidden_dim * 2),
        )
        self.vae_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            self._get_activation("vae_decoder", "vae"),
            nn.Linear(self.hidden_dim * 2, self.output_dim),
        )
        self.vae_sigma_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            self._get_activation("vae_sigma_head", "vae"),
            nn.Linear(self.hidden_dim * 2, self.output_dim),
        )

        if 'decoder' in self.heads:
            self.vae_ts_decoder = self._instaciate_ts_decoder(self._forward_vae_x_head_decoder_step)


    def _make_heads(self) -> None:
        # Heads used by different cost functions
        if 'x' in self.heads:
            self._make_x_head()

        if 'noise' in self.heads:
            self._make_noise_head()

        if 'miss' in self.heads:
            self._make_miss_head()
        
        if 'lambda' in self.heads:
            self._make_lambda_head()

        if 'events' in self.heads:
            self._make_events_heads()

        if 'vae_x' in self.heads:
            self._make_vae_x_heads()

    def _forward_x_head(self, state: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> None:
        if 'decoder' in self.heads:
            decoder_out = self._forward_ts_decoder(state, batch, res)
            x_hat, decoder_state, _ = decoder_out
            res['decoder_state'] = decoder_state
            if 'lambda' in self.heads:
                lambda_input = self._make_lambda_decoder_input(state, decoder_state, batch)
                lam_t = F.softplus(self.lambda_head(lambda_input)).clamp(
                    min=1 / (2 * math.pi), max=2 * math.pi
                )
                lam2 = lam_t
                res['lambda_hat'] = lam2
        else:
            x_hat = self.decoder(state)
            if 'lambda' in self.heads:
                lam_t = F.softplus(self.lambda_head(state)).clamp(
                    min=1 / (2 * math.pi), max=2 * math.pi
                )
                lam2 = lam_t
                res['lambda_hat'] = lam2
        x_hat = self._limit_features(x_hat, self.cost_cols or self.input_cols)
        res['x_hat'] = x_hat


    def _forward_ts_encoder(self, h: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> torch.Tensor:
        state = self.ts_encoder(h)
        if self.ts_encoder.last_decoder_state is not None:
            res["decoder_state"] = self.ts_encoder.last_decoder_state
        return state

    def _forward_ts_decoder_inner(self, decoder: nn.Module, h: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> tuple[torch.Tensor,torch.Tensor,tuple[torch.Tensor,...]]:
        decoder_input = self._to_model_dtype(res['x_train_cost'])
        if self.ts_decoder_type == TSDecoderType.GRU:
            return decoder(
                decoder_input,
                h,
                self.head_window_size,
                self.ts_decoder_params,
                initial_state=res.get("decoder_state"),
            )

        if self.ts_decoder_type == TSDecoderType.LSTM:
            return decoder(
                decoder_input,
                h,
                self.head_window_size,
                self.ts_decoder_params,
                initial_state=res.get("decoder_state"),
                cell_state=res.get("cell_state"),
            )     
        
        if self.ts_decoder_type == TSDecoderType.ODE_JUMP:     
            return decoder(
                decoder_input,
                h,
                self.head_window_size,
                self.ts_decoder_params,
                initial_state=res.get("decoder_state"),
                input_timestamps=batch.get("timestamps"),
                head_timestamps=batch.get("head_timestamps"),
            )
        
        if self.ts_decoder_type == TSDecoderType.TRANSFORMER:
            head_targets = batch.get("head_cost")
            if head_targets is not None:
                head_targets = self._to_model_dtype(head_targets)
            return decoder(
                h,
                self.head_window_size,
                self.ts_decoder_params,
                initial_state=res.get("decoder_state"),
                head_targets=head_targets,
                input_timestamps=batch.get("timestamps"),
                head_timestamps=batch.get("head_timestamps")
            )  
        
        if os.environ.get('dev') == 'true':
            if self.ts_decoder_type == TSDecoderType.FUTURE_GRU:
                h_in = res.get('h_in')
                return decoder(
                    decoder_input,
                    h,
                    self.head_window_size,
                    self.ts_decoder_params,
                    initial_state=res.get("decoder_state"),
                    initial_z=h_in[:,-1,:] if h_in is not None else None
                )           
        
    def _forward_ts_decoder(self, h: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> tuple[torch.Tensor,torch.Tensor,tuple[torch.Tensor,...]]:

        return self._forward_ts_decoder_inner(self.ts_decoder,h,batch,res)

    def _forward_events_head(self, state: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> None:
        head_timestamps = batch.get("head_timestamps")
        if head_timestamps is None:
            raise ValueError("Events head requires `head_timestamps` in the batch.")

        decoder_state = res.get("decoder_state")
        if decoder_state is None:
            raise ValueError("Events head requires `decoder_state`; event predictions are decoder-based only.")
        if decoder_state.dim() != 3:
            raise ValueError(
                "Events head requires decoder_state with shape [batch, head_window_size, hidden_dim]. "
                f"Got shape={tuple(decoder_state.shape)}."
            )
        if decoder_state.size(1) != head_timestamps.size(1):
            raise ValueError(
                "decoder_state time dimension must match head_timestamps. "
                f"Got decoder_state={tuple(decoder_state.shape)}, head_timestamps={tuple(head_timestamps.shape)}."
            )

        events_hat = self.events_head(decoder_state)

        if self.limit_events is not None:
            head_cost = batch["head_cost"]
            events = (head_cost <= self.limit_events[0]) | (head_cost >= self.limit_events[1])
            events_mask = batch.get("mask_head_cost")
            if events_mask is None:
                events_mask = torch.ones_like(events, dtype=events_hat.dtype)
            else:
                events_mask = events_mask.to(dtype=events_hat.dtype)
            res["events_multilabel"] = True
        else:
            state_pred = batch.get("state_pred")
            if state_pred is None:
                raise ValueError("Events head with Softmax requires `state_pred` in the batch.")
            origin_ts = batch["timestamps"][:, -1].unsqueeze(1)
            event_times = state_pred[:, -1, :].to(dtype=head_timestamps.dtype)
            previous_timestamps = torch.cat([origin_ts, head_timestamps[:, :-1]], dim=1)
            in_step = (
                (event_times.unsqueeze(1) > previous_timestamps.unsqueeze(-1))
                & (event_times.unsqueeze(1) <= head_timestamps.unsqueeze(-1))
            )
            masked_event_times = event_times.unsqueeze(1).masked_fill(~in_step, float("inf"))
            event_idx = masked_event_times.argmin(dim=-1)
            events_mask = in_step.any(dim=-1, keepdim=True).float()
            events = F.one_hot(event_idx, num_classes=self.status_dim).to(dtype=events_hat.dtype)
            events = events * events_mask
            res["events_multilabel"] = False

        res["events_hat"] = events_hat
        res["events"] = events.to(dtype=events_hat.dtype)
        res["events_mask"] = events_mask.to(dtype=events_hat.dtype)

    @staticmethod
    def apply_vae_distribution(vae_mu: torch.Tensor, vae_logvar: torch.Tensor) -> torch.Tensor:

        return vae_mu + torch.randn_like(vae_mu) * torch.exp(0.5 * vae_logvar)
    
    def _forward_vae_x_decoder(self, state: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> tuple[torch.Tensor,torch.Tensor,tuple[torch.Tensor,...]]:
        return self._forward_ts_decoder_inner(self.vae_ts_decoder,state,batch,res)

    def _forward_vae_x_head(self, state: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> None:
        if 'decoder' in self.heads:
            vae_x,_,(vae_mu,vae_logvar,vae_logvar_obs) = self._forward_vae_x_decoder(state, batch, res)
        else:
            vae_x,vae_mu,vae_logvar,vae_logvar_obs = self._forward_vae_x_head_decoder_step(state)
        res['vae_x'] = vae_x
        res['vae_mu'] = vae_mu
        res['vae_logvar'] = vae_logvar
        res['vae_logvar_obs'] = vae_logvar_obs

    def _forward_vae_x_head_decoder_step(self, state: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        mu_logvar = self.vae_latent(state)
        vae_mu, vae_logvar = torch.chunk(mu_logvar, 2, dim=-1)
        z_vae = self.apply_vae_distribution(vae_mu, vae_logvar)
        vae_x = self.vae_decoder(z_vae)
        vae_logvar_obs = self.vae_sigma_head(z_vae).clamp(min=-5.0, max=5.0)
        return vae_x,vae_mu,vae_logvar,vae_logvar_obs

    def _forward_noise_head(self, state: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> None:
        noise_hat = self.noise_head(state)
        res['noise_hat'] = noise_hat
        res['mask_train_cost'] = torch.zeros_like(res['mask_train_cost'])

    def _forward_noise_diffusion(self, h0: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(h0)
        t = torch.randint(0, self.num_steps, (h0.size(0),), device=h0.device)
        beta = ((t.to(dtype=h0.dtype) + 1.0) / float(self.num_steps)).view(-1, 1, 1)
        h_noisy = torch.sqrt(1.0 - beta) * h0 + torch.sqrt(beta) * noise
        res["noise_step"] = t
        res["noise_beta"] = beta
        res["h_noisy"] = h_noisy
        return noise,h_noisy

    def _forward_impl(
        self,
        batch: tensors_dict
    ) -> tensors_dict:
        
        x = batch['x']
        mask = batch['mask']  
        static_feats = batch.get('static_feats')
        ts = batch['timestamps']
        encoder_final_timestamp = ts[:, -1]
        batch["encoder_final_timestamp"] = encoder_final_timestamp
        res = tensors_dict()
        if mask is None:
            mask = torch.ones_like(x)

        if (self.force_irregular_timestep and self.training) or (not self.training and self.force_irregular_timestep_test):
            keep_idx = self._make_irregular_timestep_keep_idx(x, x.device)
            if keep_idx is not None:
                x = self._select_timesteps(x, keep_idx)
                mask = self._select_timesteps(mask, keep_idx)
                ts = self._select_timesteps(ts, keep_idx)
                batch = batch.copy()
                batch["x"] = x
                batch["mask"] = mask
                batch["timestamps"] = ts
                res["x"] = x
                res["mask"] = mask
                res["timestamps"] = ts
                state_pred = batch.get("state_pred")
                if state_pred is not None:
                    state_pred = self._select_timesteps(state_pred, keep_idx)
                    batch["state_pred"] = state_pred
                    res["state_pred"] = state_pred

        if (self.rebuild_timeseries and self.training) or (not self.training and self.rebuild_timeseries_test):
            mask_train = self._make_random_mask(x,ts,mask,x.device)
        else:
            mask_train = torch.zeros_like(mask)
            mask_train = mask_train.copy_(mask)
        x_train = x * mask_train
        if self.cost_cols is None:
            res['x_cost'] = x
            res['mask_cost'] = mask
            res['mask_train_cost'] = mask_train
            res['x_train_cost'] = x_train
        else:
            res['x_cost'] = x[:,:,self.cost_cols_idx]
            res['mask_cost'] = mask[:,:,self.cost_cols_idx]      
            res['mask_train_cost'] = mask_train[:,:,self.cost_cols_idx]
            res['x_train_cost'] = x_train[:,:,self.cost_cols_idx]

        res['mask_train'] = mask_train
        res['x_train'] = x_train
        

        encoder_input = self._to_model_dtype(torch.cat([x_train, mask_train], dim=-1))
        h0 = self.encoder(encoder_input, ts)

        if 'miss' in self.heads:
            miss_hat = self.miss_head(h0)
            res['miss_hat'] = miss_hat

        if static_feats is not None and self.static_proj is not None:
            h0 = h0 + self.static_proj(self._to_model_dtype(static_feats)).unsqueeze(1)

        if 'noise' in self.heads:
            noise, h_in = self._forward_noise_diffusion(h0, batch, res)
        else:
            h_in = h0
            noise = None
        res['h_in'] = h_in
        res['noise'] = noise

        state = self._forward_ts_encoder(h_in, batch, res)
        res['state'] = state
        if 'noise' in self.heads:
            self._forward_noise_head(state, batch, res)

        if 'x' in self.heads:
            self._forward_x_head(state, batch, res)
        
        if 'events' in self.heads:
            self._forward_events_head(state, batch, res)

        if 'vae_x' in self.heads:
            self._forward_vae_x_head(state, batch, res)

        return res

    def forward(
        self,
        batch: tensors_dict
    ) -> tensors_dict:
        with self._autocast_context(batch["x"].device):
            return self._forward_impl(batch)

    @staticmethod
    def _select_timesteps(tensor: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
        if tensor.dim() < 2:
            raise ValueError("Expected a batched time tensor with at least 2 dimensions.")
        if tensor.size(0) != keep_idx.size(0):
            raise ValueError(
                "keep_idx batch dimension must match tensor batch dimension. "
                f"Got tensor={tuple(tensor.shape)}, keep_idx={tuple(keep_idx.shape)}."
            )
        if tensor.dim() == 2:
            return torch.gather(tensor, 1, keep_idx)

        gather_idx = keep_idx.reshape(keep_idx.size(0), keep_idx.size(1), *([1] * (tensor.dim() - 2)))
        gather_idx = gather_idx.expand(-1, -1, *tensor.shape[2:])
        return torch.gather(tensor, 1, gather_idx)

    def _make_irregular_timestep_keep_idx(self, x: torch.Tensor, device: torch.device) -> torch.Tensor | None:
        max_drop = float(self.force_irregular_timestep_max_drop)
        if max_drop < 0.0 or max_drop > 1.0:
            raise ValueError("force_irregular_timestep_max_drop must be in [0, 1].")
        if max_drop == 0.0 or x.size(1) <= 1:
            return None

        num_steps = max(int(self.num_steps), 1)
        t_drop = torch.randint(0, num_steps, (1,), device=device)
        p_drop = (t_drop.float() / float(max(num_steps - 1, 1))) * max_drop
        drop_count = int(torch.floor(p_drop * x.size(1)).item())
        drop_count = min(max(drop_count, 0), x.size(1) - 1)
        if drop_count == 0:
            return None

        keep_count = x.size(1) - drop_count
        keep_idx = torch.rand(x.size(0), x.size(1), device=device).argsort(dim=1)[:, :keep_count]
        keep_idx, _ = torch.sort(keep_idx)
        return keep_idx

    def _make_random_mask(self,x,ts_batch,m,device):
        max_drop = self.rebuild_timeseries_max_drop
        max_drop_step = self.rebuild_timeseries_max_drop_step
        t_mask = torch.randint(0, self.num_steps, (x.size(0),), device=device)
        t_mask_ts = torch.randint(0, self.num_steps, (x.size(0),), device=device)
        # 2) probabilidade de *extra-missing* cresce com t
        p_drop_t = (t_mask.float() / (self.num_steps - 1)) * max_drop   # (B,)
        p_drop_t = p_drop_t.view(-1, 1, 1)                         # broadcast
        p_drop_ts = (t_mask_ts.float() / (self.num_steps - 1)) * max_drop_step   # (B,)
        p_drop_ts = p_drop_ts.view(-1, 1)                         # broadcast

        rand_mask = (torch.rand_like(m) > p_drop_t).float()
        rand_mask_ts = (torch.rand_like(ts_batch) > p_drop_ts).unsqueeze(-1).float()
        return m * rand_mask * rand_mask_ts
    
    def _get_heads(self) -> None:
        heads = set()
        for cf in self.cost_functions.values():
            for head in cf.heads:
                heads.add(head)
        self.heads = heads

    def compute_loss(
        self,
        batch: tensors_dict,
        outputs: tensors_dict,
        log_metrics: bool,
    ) -> tuple[torch.Tensor, dict[str, list[float]], list[dict[str, Any]]]:
        loss_batch = batch.copy()
        for k, v in outputs.items():
            loss_batch[k] = self._to_loss_dtype(v) if torch.is_tensor(v) else v

        total_loss = 0.0
        loss_dict: dict[str, list[float, float]] = {}
        metrics: dict[str, dict[str, Any]] = {}

        for name, cost_fn in self.cost_functions.items():
            loss_value = cost_fn(loss_batch, self.training, **self.loss_params)

            if cost_fn.valid_cost_function:
                total_loss = total_loss + cost_fn.get_ratio(self.training) * loss_value

                loss_dict[name] = [cost_fn.loss, cost_fn.loss_div]
                metrics[name] = {}
                for metric_name, metric in cost_fn.metrics.items():
                    metrics[name][metric_name] = metric

        return total_loss, loss_dict, [metrics] if log_metrics else []

    def _training_step(self, batch: tensors_dict, log_metrics: bool = True) -> tuple[torch.Tensor, dict[str, list[float]], list[dict[str, Any]]]:
        outputs = self.forward(batch)
        return self.compute_loss(batch, outputs, log_metrics)

    def _validation_step(self, batch: tensors_dict, log_metrics: bool = True) -> tuple[torch.Tensor, dict[str, list[float]], list[dict[str, Any]]]:
        with torch.no_grad():
            
            outputs = self.forward(batch)
            return self.compute_loss(batch, outputs, log_metrics)

    def predict_step(self, batch: tensors_dict) -> tensors_dict:
        with torch.no_grad():
            return self.forward(batch)

    def _get_score_from_loss_dict(self, loss_dict: dict[str, list[float]]) -> float:
        if not loss_dict:
            return float("inf")
        score = 0.0
        for cost_name, (total, weight, ratio) in loss_dict.items():
            score += ratio * float(total) / max(float(weight), 1.0)
        return float(score)
    
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def save_weights(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> None:
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def get_x_cost(self,x):
        return x[:,:,self.cost_cols_idx] if self.cost_cols is not None else x

    def _make_tensor_scaler(self,t_train:torch.Tensor,Scaler: callable) -> None:
        self.scaler = Scaler()
        B, T, C = t_train.shape
        t_2d = t_train.numpy().reshape(B * T, C)
        self.scaler.fit(t_2d)
        if self.cost_cols is not None:
            self.cost_scaler = Scaler()
            t_cost_2d = self.get_x_cost(t_train).numpy().reshape(B * T, len(self.cost_cols))
            self.cost_scaler.fit(t_cost_2d)
    
    def _apply_tensor_scaler(self,t_all:torch.Tensor, scaler: callable) -> torch.Tensor:
        N, T, C = t_all.shape
        t_all = t_all.numpy().reshape(N * T, C)
        t_all_scaled = scaler.transform(t_all).reshape(N, T, C)        
        return torch.tensor(t_all_scaled, dtype=torch.float32)

    def _scale_dataset(self, dataset: TensorDataset) -> None:
        """
        Utility method to scale or normalize inputs when needed.
        Subclasses may override this for custom logic.
        """
        dataset.tensors = (self._apply_tensor_scaler(dataset.tensors[0], self.scaler),) + dataset.tensors[1:]
        
        if self.cost_cols is not None and 'head_cost' in dataset.keys:
            head_cost_idx = dataset.keys.index('head_cost')
            dataset.tensors = dataset.tensors[:head_cost_idx] + (self._apply_tensor_scaler(dataset.tensors[head_cost_idx], self.cost_scaler),) + dataset.tensors[head_cost_idx+1:]
    
    def _segregate_data_time_series(
        self,
        dataset: TensorDataset,
        validation_ratio: float,
        test_ratio: float,
    ) -> tuple[list[int], list[int], list[int]]:
        n = len(dataset)
        validate = validation_ratio > 0.0
        train_frac = max(1.0 - max(validation_ratio, 0.0) - max(test_ratio, 0.0), 0.0)
        val_frac = max(validation_ratio, 0.0)
        test_frac = max(test_ratio, 0.0)
        window_size, head_size = self.window_size, self.head_window_size
        purge_gap = window_size + head_size - 1

        n_splits = 3 if validate else 2
        n_gaps = n_splits - 1
        min_required = n_splits + purge_gap * n_gaps
        if n < min_required:
            raise ValueError(
                "Not enough windows to create leakage-safe chronological splits. "
                f"n_windows={n}, window_size={window_size}, head_window_size={head_size}, "
                f"required_min={min_required}."
            )

        effective_n = n - purge_gap * n_gaps
        n_train, n_val, n_test = self._split_counts(effective_n, train_frac, val_frac, test_frac)

        if validate:
            train_start = 0
            train_end = train_start + n_train
            val_start = train_end + purge_gap
            val_end = val_start + n_val
            test_start = val_end + purge_gap
            test_end = test_start + n_test

            train_idx = torch.arange(train_start, train_end, dtype=torch.long)
            val_idx = torch.arange(val_start, val_end, dtype=torch.long)
            test_idx = torch.arange(test_start, test_end, dtype=torch.long)
            return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

        train_start = 0
        train_end = train_start + n_train
        test_start = train_end + purge_gap
        test_end = test_start + n_test

        train_idx = torch.arange(train_start, train_end, dtype=torch.long)
        test_idx = torch.arange(test_start, test_end, dtype=torch.long)
        return train_idx.tolist(), [], test_idx.tolist()

    def _segregate_data(
        self,
        dataset: TensorDataset,
        validation_ratio: float,
        test_ratio: float,
    ) -> tuple[list[int], list[int], list[int]]:
        n = len(dataset)
        if n == 0:
            return [], [], []

        return self._segregate_data_time_series(
            dataset=dataset,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
    
    def _make_trainloader(
        self,
        dataset: TensorDataset,
        train_idx: list[int],
        batch_size: int,
    ) -> DataLoader:
        return DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def _make_dataloader(
        self,
        dataset: TensorDataset,
        idx: list[int],
        batch_size: int,
    ) -> DataLoader:
        return DataLoader(
            Subset(dataset, idx),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
    
    def _log_metrics(
        self, 
        log: Any,
        log_path: str | Path
    ):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    @staticmethod
    def _log_means(split_loss: dict[str, list[float]] | None) -> dict[str, float]:
        if not split_loss:
            return {}
        means = {}
        mean_all = 0.
        for name, (total, weight, ratio) in split_loss.items():
            metric = float(total) / max(float(weight), 1.0)
            means[name] = metric
            mean_all += ratio * metric
        means['avg'] = mean_all

        return means
    
    def _log_training_progress(
        self, 
        epoch: int, 
        train_loss: dict[str, list[float]], 
        val_loss: dict[str, list[float]] | None, 
        test_loss: dict[str, list[float]] | None,
        epoch_start: datetime.datetime | None = None,
        prefix: str | None = None,
        
        ) -> None:
        """
        Default train/validation/test logger with metric means and split-level mean.
        Expected value format for each metric: [error_sum, weight_sum].
        """


        def _fmt_split(name: str, split_loss: dict[str, list[float]] | None) -> str:
            if split_loss is None:
                means = {}
                mean_all = None
            else:
                means = {k:v for k,v in split_loss.items() if k != 'avg'}
                mean_all = split_loss['avg']
            if mean_all is None:
                return f"{name}=n/a"
            metrics_str = " ".join(f"{k}:{v:.6f}" for k, v in sorted(means.items()))
            return f"{name}[avg:{mean_all:.6f}] {metrics_str}"

        elapsed = ""
        if epoch_start is not None:
            elapsed_seconds = (datetime.datetime.now() - epoch_start).total_seconds()
            elapsed = f" | time:{elapsed_seconds:.2f}s"

        prefix_str = f"{prefix} | " if prefix else ""

        print(
            f"{prefix_str}"
            f"Epoch {epoch:04d} | "
            f"{_fmt_split('train', train_loss)} | "
            f"{_fmt_split('val', val_loss)} | "
            f"{_fmt_split('test', test_loss)}"
            f"{elapsed}"
        )

    def _log_test_progress(
        self,
        val_loss: dict[str, float] | None,
        test_loss: dict[str, float],
        val_score: float | None = None,
        test_score: float | None = None,
    ) -> None:
        def _fmt_split(name: str, split_loss: dict[str, float] | None) -> str:
            if split_loss is None:
                means = {}
                mean_all = None
            else:
                means = {k: v for k, v in split_loss.items() if k != "avg"}
                mean_all = split_loss.get("avg")
            if mean_all is None:
                return f"{name}=n/a"
            metrics_str = " ".join(f"{k}:{v:.6f}" for k, v in sorted(means.items()))
            return f"{name}[avg:{mean_all:.6f}] {metrics_str}"

        score_str = []
        if val_score is not None:
            score_str.append(f"val_score:{val_score:.6f}")
        if test_score is not None:
            score_str.append(f"test_score:{test_score:.6f}")
        score_suffix = f" | {' '.join(score_str)}" if score_str else ""

        print(
            f"Test | "
            f"{_fmt_split('val', val_loss)} | "
            f"{_fmt_split('test', test_loss)}"
            f"{score_suffix}"
        )

    def _detach_to_cpu(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._detach_to_cpu(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._detach_to_cpu(v) for v in value)
        if isinstance(value, list):
            return [self._detach_to_cpu(v) for v in value]
        return value

    def _concat_predictions(self, outputs: list[Any]) -> Any:
        if len(outputs) == 0:
            return []
        first = outputs[0]
        if torch.is_tensor(first):
            return torch.cat(outputs, dim=0)
        if isinstance(first, dict):
            keys = first.keys()
            return {k: self._concat_predictions([out[k] for out in outputs]) for k in keys}
        if isinstance(first, tuple):
            return tuple(self._concat_predictions([out[i] for out in outputs]) for i in range(len(first)))
        if isinstance(first, list):
            transposed = list(zip(*outputs)) if len(first) > 0 else []
            return [self._concat_predictions(list(items)) for items in transposed]
        return list(itertools.chain.from_iterable(
            [out] if not isinstance(out, list) else out for out in outputs
        ))
    
    def _make_optimizer(
        self,
        optimizer_class: type[torch.optim.Optimizer],
        optimizer_base_params: dict,
        optimizer_specific_params: dict[str, dict],
        optimizer_additional_kwargs: dict
    ) -> torch.optim.Optimizer:
        
        if optimizer_specific_params is None:
            return optimizer_class(self.parameters(), **optimizer_base_params, **optimizer_additional_kwargs)
        else:
            grouped_params = {name: [] for name in optimizer_specific_params.keys()}
            base_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                matched = False
                for group_name in optimizer_specific_params.keys():
                    if group_name in name:
                        grouped_params[group_name].append(param)
                        matched = True
                        break
                if not matched:
                    base_params.append(param)

            optimizer_params = []
            if base_params:
                optimizer_params.append({**optimizer_base_params, "params": base_params})

            for group_name, group_config in optimizer_specific_params.items():
                params = grouped_params[group_name]
                if len(params) == 0:
                    raise ValueError(
                        f"Optimizer params specified for '{group_name}' but no parameters matched that name."
                    )
                optimizer_params.append({**group_config, "params": params})

            if len(optimizer_params) == 0:
                raise ValueError("No trainable parameters were collected for the optimizer.")

            return optimizer_class(optimizer_params, **optimizer_additional_kwargs)
                
    def _make_scheduler(self, optimizer: torch.optim.Optimizer, scheduler_params: dict) -> torch.optim.lr_scheduler._LRScheduler:
        warmup_steps_cfg = scheduler_params.get("warmup_steps", 0)
        warmup_min_steps = max(int(scheduler_params.get("warmup_min_steps", 0)), 0)
        if isinstance(warmup_steps_cfg, (float, np.floating)):
            warmup_ratio = float(warmup_steps_cfg)
            if not 0.0 <= warmup_ratio <= 1.0:
                raise ValueError("scheduler_params['warmup_steps'] as percentual must be in [0, 1].")
            warmup_steps = max(warmup_min_steps, int(warmup_ratio * self.total_steps))
        else:
            warmup_steps = max(int(warmup_steps_cfg), 0)
        warmup_steps = min(warmup_steps, self.total_steps)
        min_lr_factor = scheduler_params.get("min_lr_factor", 0.1)
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)  # linear warmup
            # cosine decay até min_lr_factor
            progress = (step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (1 - min_lr_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _set_current_epoch(self, epoch: int) -> None:
        for module in self.modules():
            if hasattr(module, "current_epoch"):
                module.current_epoch = int(epoch)
    
    def _limit_features(self, tensor: torch.Tensor, col_list: list[str]) -> torch.Tensor:
        if self.feature_limits is None:
            return tensor

        if isinstance(self.feature_limits, tuple):
            tensor = torch.clamp(tensor, min=self.feature_limits[0], max=self.feature_limits[1])
        else:
            for col, (min_val, max_val) in self.feature_limits.items():
                idx = col_list.index(col)
                tensor[:, :, idx] = torch.clamp(tensor[:, :, idx], min=min_val, max=max_val)
        
        return tensor
    
    def _preprocess_limit_features(self, dataset: TensorDataset) -> None:
        if self.feature_limits is None:
            return
        
        x_tensor = dataset.tensors[0]
        x_tensor = self._limit_features(x_tensor, self.input_cols)
        dataset.tensors = (x_tensor,) + dataset.tensors[1:]
        if self.cost_cols is not None and 'head_cost' in dataset.keys:
            head_cost_idx = dataset.keys.index('head_cost')
            cost_tensor = dataset.tensors[head_cost_idx]
            cost_tensor = self._limit_features(cost_tensor, self.cost_cols)
            dataset.tensors = dataset.tensors[:head_cost_idx] + (cost_tensor,) + dataset.tensors[head_cost_idx+1:]
        


            
    @staticmethod
    def _merge_metric_tree(aggregated: dict[str, Any], metrics_batch: dict[str, Any]) -> None:
        if not isinstance(metrics_batch, dict):
            raise TypeError(
                "Each metrics batch must be a dict[str, Any], "
                f"got {type(metrics_batch).__name__}."
            )
        for key, value in metrics_batch.items():
            if isinstance(value, dict) and value.get("kind") == "standard_error":
                agg = aggregated.setdefault(
                    key,
                    {"kind": "standard_error", "loss_sum": 0.0, "loss_div": 0.0, "wm2": 0.0, "count": 0.0},
                )
                agg["loss_sum"] += float(value.get("loss_sum", 0.0))
                agg["loss_div"] += float(value.get("loss_div", 0.0))
                agg["wm2"] += float(value.get("wm2", 0.0))
                agg["count"] += float(value.get("count", 0.0))
                continue

            if isinstance(value, dict):
                branch = aggregated.setdefault(key, {})
                BaseIndustrialTSModel._merge_metric_tree(branch, value)
                continue

            if isinstance(value, (list, tuple)) and len(value) == 2:
                if value[1] is None:
                    agg = aggregated.setdefault(key, {"kind": "sum", "total": 0.0})
                    agg["total"] += float(value[0])
                    continue
                agg = aggregated.setdefault(key, {"kind": "weighted_mean", "total": 0.0, "weight": 0.0})
                agg["total"] += float(value[0])
                agg["weight"] += float(value[1])
                continue

            agg = aggregated.setdefault(key, {"kind": "scalar_mean", "total": 0.0, "count": 0.0})
            agg["total"] += float(value)
            agg["count"] += 1.0

    @staticmethod
    def _finalize_metric_tree(aggregated: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in aggregated.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            kind = value.get("kind")
            if kind == "scalar_mean":
                out[key] = float(value["total"] / max(value["count"], 1.0))
                continue

            if kind == "weighted_mean":
                out[key] = float(value["total"] / max(value["weight"], 1.0))
                continue

            if kind == "standard_error":
                if value["loss_div"] <= 0.0 or value["count"] < 2.0:
                    out[key] = float("nan")
                else:
                    mean = value["loss_sum"] / value["loss_div"]
                    variance = max(value["wm2"] / value["loss_div"] - mean * mean, 0.0)
                    out[key] = float(math.sqrt(variance / value["count"]))
                continue

            if kind == "sum":
                out[key] = float(value["total"])
                continue

            out[key] = BaseIndustrialTSModel._finalize_metric_tree(value)

        return out

    def _process_val_metrics(self, val_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if len(val_metrics) == 0:
            return {}
        aggregated: dict[str, Any] = {}
        for metrics_batch in val_metrics:
            self._merge_metric_tree(aggregated, metrics_batch)
        return self._finalize_metric_tree(aggregated)

    def _process_train_metrics(self, train_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        return self._process_val_metrics(train_metrics)

    def _fit_dataset_scaler(
            self,
            dataset: TensorDataset,
            scaler: callable | None,
            fit_idx: list[int] | None = None) -> None:
        if scaler is None:
            return

        t_fit = dataset.tensors[0]
        if fit_idx is not None:
            if len(fit_idx) == 0:
                raise ValueError("Cannot fit scaler without training windows.")
            t_fit = t_fit[fit_idx]
        self._make_tensor_scaler(t_fit, scaler)

    def _apply_dataset_preprocessing(self, dataset: TensorDataset) -> None:
        if self.scaler is not None:
            self._scale_dataset(dataset)
        if self.feature_limits is not None:
            self._preprocess_limit_features(dataset)

    def _parse_scaler_ratio(
            self,
            scaler_ratio: float | tuple[float, float] | tuple[float, float, float] | list[float] | dict[str, float] | None,
        ) -> tuple[float, float, float]:
        if scaler_ratio is None:
            return 1.0, 0.0, 0.0

        train_ratio: float
        validation_ratio: float
        test_ratio: float

        if isinstance(scaler_ratio, (int, float, np.floating)):
            train_ratio = float(scaler_ratio)
            validation_ratio = 0.0
            test_ratio = max(1.0 - train_ratio, 0.0)
        elif isinstance(scaler_ratio, dict):
            if "train_ratio" in scaler_ratio or "train" in scaler_ratio:
                train_ratio = float(scaler_ratio.get("train_ratio", scaler_ratio.get("train", 0.0)))
                validation_ratio = float(scaler_ratio.get("validation_ratio", scaler_ratio.get("validation", scaler_ratio.get("val", 0.0))))
                test_ratio = float(scaler_ratio.get("test_ratio", scaler_ratio.get("test", 0.0)))
                total = train_ratio + validation_ratio + test_ratio
                if total <= 0.0:
                    raise ValueError("scaler_ratio must contain at least one positive split component.")
                train_ratio /= total
                validation_ratio /= total
                test_ratio /= total
            else:
                validation_ratio = float(scaler_ratio.get("validation_ratio", scaler_ratio.get("validation", scaler_ratio.get("val", 0.0))))
                test_ratio = float(scaler_ratio.get("test_ratio", scaler_ratio.get("test", 0.0)))
                train_ratio = max(1.0 - validation_ratio - test_ratio, 0.0)
        elif isinstance(scaler_ratio, (tuple, list)):
            if len(scaler_ratio) == 2:
                validation_ratio = float(scaler_ratio[0])
                test_ratio = float(scaler_ratio[1])
                train_ratio = max(1.0 - validation_ratio - test_ratio, 0.0)
            elif len(scaler_ratio) == 3:
                train_ratio = float(scaler_ratio[0])
                validation_ratio = float(scaler_ratio[1])
                test_ratio = float(scaler_ratio[2])
                total = train_ratio + validation_ratio + test_ratio
                if total <= 0.0:
                    raise ValueError("scaler_ratio must contain at least one positive split component.")
                train_ratio /= total
                validation_ratio /= total
                test_ratio /= total
            else:
                raise ValueError(
                    "scaler_ratio tuple/list must have 2 elements "
                    "(validation_ratio, test_ratio) or 3 elements "
                    "(train_ratio, validation_ratio, test_ratio)."
                )
        else:
            raise TypeError(
                "scaler_ratio must be a float, tuple/list, dict, or None, "
                f"got {type(scaler_ratio).__name__}."
            )

        for name, value in (
            ("train_ratio", train_ratio),
            ("validation_ratio", validation_ratio),
            ("test_ratio", test_ratio),
        ):
            if value < 0.0:
                raise ValueError(f"scaler_ratio produced a negative {name}: {value}.")

        return train_ratio, validation_ratio, test_ratio

    def _resolve_scaler_fit_idx(
            self,
            dataset: TensorDataset,
            scaler_ratio: float | tuple[float, float] | tuple[float, float, float] | list[float] | dict[str, float] | None,
        ) -> list[int]:
        train_ratio, validation_ratio, test_ratio = self._parse_scaler_ratio(scaler_ratio)
        if len(dataset) == 0:
            return []

        if train_ratio >= 1.0 and validation_ratio <= 0.0 and test_ratio <= 0.0:
            return list(range(len(dataset)))

        if train_ratio <= 0.0:
            raise ValueError(
                "scaler_ratio does not reserve any train-equivalent windows for fitting the scaler."
            )

        train_idx, _, _ = self._segregate_data(
            dataset,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        if len(train_idx) == 0:
            raise ValueError(
                "scaler_ratio produced an empty train-equivalent split for scaler fitting."
            )
        return train_idx
    
    def _preprocess_dataset(
            self, 
            dataset: TensorDataset, 
            scaler: callable | None, 
            fit_idx: list[int] | None = None,
            scaler_dataset: TensorDataset | None = None) -> None:
        fit_dataset = scaler_dataset if scaler_dataset is not None else dataset
        self._fit_dataset_scaler(fit_dataset, scaler, fit_idx=fit_idx)
        self._apply_dataset_preprocessing(dataset)

    def _prepare_dataset_splits(
            self,
            data: DataFrame,
            window_size: int,
            window_stride: int,
            head_window_size: int | None,
            validation_ratio: float,
            test_ratio: float,
            batch_size: int,
            scaler: callable | None,
            allow_full_test_ratio: bool = False,
        ) -> tuple[TensorDataset, list[int], list[int], list[int], DataLoader | None, DataLoader | None, DataLoader | None]:
        if head_window_size is None:
            head_window_size = 0
        if head_window_size < 0:
            raise ValueError("head_window_size must be at least 0 to ensure valid windows.")
        if "decoder" in self.heads and head_window_size < 1:
            raise ValueError("Decoder cost functions require head_window_size >= 1.")
        if "events" in self.heads and head_window_size < 1:
            raise ValueError("EventsCostFunction requires head_window_size >= 1.")
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.window_size = window_size
        self.head_window_size = head_window_size
        dataset = self._make_dataset(
            data,
            window_size=window_size,
            window_stride=window_stride,
            head_window_size=head_window_size,
        )
        if len(dataset) == 0:
            raise ValueError("Received empty dataset after _make_dataset.")

        if allow_full_test_ratio and validation_ratio <= 0.0 and test_ratio >= 1.0:
            train_idx = []
            val_idx = []
            test_idx = list(range(len(dataset)))
        else:
            train_idx, val_idx, test_idx = self._segregate_data(dataset, validation_ratio, test_ratio)
        if len(train_idx) > 0:
            fit_idx = train_idx
        else:
            fit_idx = test_idx

        self._preprocess_dataset(dataset, scaler, fit_idx=fit_idx)

        train_loader = None
        if len(train_idx) > 0:
            train_loader = self._make_trainloader(dataset, train_idx, batch_size)
        val_loader = self._make_dataloader(dataset, val_idx, batch_size) if validation_ratio > 0 else None
        test_loader = self._make_dataloader(dataset, test_idx, batch_size)
        return dataset, train_idx, val_idx, test_idx, train_loader, val_loader, test_loader

    def _evaluate_split_loader(
            self,
            split_loader: DataLoader | None,
            dataset: TensorDataset,
            device: torch.device,
            log_metrics: bool,
            ratio_attr: str = "test_ratio",
        ) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
        loss_template = {
            k: [0.0, 0.0, float(getattr(self.cost_functions[k], ratio_attr))]
            for k in self.cost_functions.keys()
            if float(getattr(self.cost_functions[k], ratio_attr)) > 0
        }
        if split_loader is None:
            return loss_template, []

        def _single_pass() -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
            loss = {
                k: [0.0, 0.0, ratio]
                for k, (_, _, ratio) in loss_template.items()
            }
            metrics_batches = []
            for batch in split_loader:
                batch = tensors_dict(dict(zip(dataset.keys, batch)))
                batch = batch.move_to_device(device)
                _, loss_dict, metrics_batch = self._validation_step(batch, log_metrics)
                metrics_batches.extend(metrics_batch)
                for k, loss_result in loss_dict.items():
                    if k not in loss:
                        continue
                    loss[k][0] += loss_result[0]
                    loss[k][1] += loss_result[1]
            return loss, metrics_batches

        self.eval()
        with torch.no_grad():
            if self.rebuild_timeseries_test or self.force_irregular_timestep_test:
                repeated_runs = [_single_pass() for _ in range(self.test_repetition)]
                repeated_scores = [self._get_score_from_loss_dict(loss) for loss, _ in repeated_runs]
                median_idx = repeated_scores.index(np.median(repeated_scores))
                return repeated_runs[median_idx]
            return _single_pass()


    @staticmethod
    def _normalize_prediction_type(prediction_type: PredictionType | str) -> PredictionType:
        if isinstance(prediction_type, PredictionType):
            return prediction_type
        if isinstance(prediction_type, str):
            try:
                return PredictionType[prediction_type.upper()]
            except KeyError as exc:
                valid = ", ".join(pt.name for pt in PredictionType)
                raise ValueError(f"Unknown prediction_type={prediction_type!r}. Valid values: {valid}.") from exc
        raise TypeError(
            "prediction_type must be a PredictionType or string name, "
            f"got {type(prediction_type).__name__}."
        )

    @staticmethod
    def _prediction_type_heads(prediction_type: PredictionType) -> set[str]:
        return set(prediction_type.value)

    @staticmethod
    def _prediction_type_requires_head_window(prediction_type: PredictionType) -> bool:
        return "decoder" in prediction_type.value or prediction_type == PredictionType.PREDICT_EVENT

    def _validate_prediction_type(self, prediction_type: PredictionType) -> None:
        required_heads = self._prediction_type_heads(prediction_type)
        missing_heads = sorted(required_heads - set(self.heads))
        if missing_heads:
            raise ValueError(
                f"Prediction type {prediction_type.name} requires heads {sorted(required_heads)}, "
                f"but the model was initialized with heads {sorted(self.heads)}. "
                f"Missing heads: {missing_heads}."
            )

    @staticmethod
    def _copy_prediction_keys(result: tensors_dict, source: tensors_dict, keys: tuple[str, ...]) -> None:
        for key in keys:
            value = source.get(key)
            if value is not None:
                result[key] = value

    @staticmethod
    def _gaussian_log_pdf(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (logvar + ((x - mean) ** 2) / torch.exp(logvar) + math.log(2 * math.pi))

    def _predict_reconstruction(self, batch: tensors_dict, params: dict) -> tensors_dict:
        outputs = self.predict_step(batch)
        result = tensors_dict()
        x_hat = outputs["x_hat"]
        x_train = outputs["x_train_cost"]
        mask_train = outputs["mask_train_cost"]
        result["x_hat"] = x_hat
        result["x_reconstruction"] = torch.where(mask_train > 0, x_train, x_hat)
        self._copy_prediction_keys(
            result,
            outputs,
            ("x_cost", "mask_cost", "mask_train_cost", "x_train_cost"),
        )
        self._copy_prediction_keys(result, batch, ("timestamps",))
        return result

    def _predict_forecast(self, batch: tensors_dict, params: dict) -> tensors_dict:
        outputs = self.predict_step(batch)
        result = tensors_dict()
        result["x_hat"] = outputs["x_hat"]
        self._copy_prediction_keys(result, batch, ("head_timestamps", "head_cost", "mask_head_cost"))
        self._copy_prediction_keys(result, outputs, ("lambda_hat",))
        return result

    def _predict_simulation(self, batch: tensors_dict, params: dict) -> tensors_dict:
        outputs = self.predict_step(batch)
        sigma_temp = float(params.get("sigma_temp", 1.0))
        if sigma_temp <= 0.0:
            raise ValueError("predict SIMULATE params['sigma_temp'] must be > 0.")
        vae_logvar_obs = (outputs["vae_logvar_obs"] + math.log(sigma_temp)).clamp(min=-5.0, max=5.0)
        result = tensors_dict()
        result["x_sim"] = outputs["vae_x"]
        result["vae_x"] = outputs["vae_x"]
        result["vae_mu"] = outputs["vae_mu"]
        result["vae_logvar"] = outputs["vae_logvar"]
        result["vae_logvar_obs"] = vae_logvar_obs
        result["vae_std_obs"] = torch.exp(0.5 * vae_logvar_obs)
        result["sigma_temp"] = sigma_temp
        if "head_cost" in batch:
            log_pdf = self._gaussian_log_pdf(batch["head_cost"], outputs["vae_x"], vae_logvar_obs)
            result["vae_log_pdf"] = log_pdf
            result["vae_pdf"] = torch.exp(log_pdf)
        self._copy_prediction_keys(result, batch, ("head_timestamps", "head_cost", "mask_head_cost"))
        return result

    def _predict_denoise(self, batch: tensors_dict, params: dict) -> tensors_dict:
        outputs = self.predict_step(batch)
        beta = outputs["noise_beta"]
        alpha = (1.0 - beta).clamp(min=torch.finfo(outputs["h_noisy"].dtype).eps)
        h_denoised = (outputs["h_noisy"] - torch.sqrt(beta) * outputs["noise_hat"]) / torch.sqrt(alpha)
        result = tensors_dict()
        result["h_denoised"] = h_denoised
        result["h_noisy"] = outputs["h_noisy"]
        result["noise_hat"] = outputs["noise_hat"]
        result["noise_beta"] = outputs["noise_beta"]
        result["noise_step"] = outputs["noise_step"]
        if hasattr(self, "decoder") and self.decoder is not None:
            with self._autocast_context(h_denoised.device):
                x_denoised = self.decoder(self._to_model_dtype(h_denoised))
            result["x_denoised"] = self._limit_features(x_denoised, self.cost_cols or self.input_cols)
        self._copy_prediction_keys(result, outputs, ("noise", "state", "mask_cost", "mask_train_cost"))
        self._copy_prediction_keys(result, batch, ("timestamps",))
        return result

    def _predict_events(self, batch: tensors_dict, params: dict) -> tensors_dict:
        outputs = self.predict_step(batch)
        events_hat = outputs["events_hat"]
        result = tensors_dict()
        result["events_hat"] = events_hat
        if outputs.get("events_multilabel", False):
            result["events_pred"] = (events_hat >= 0.5).to(dtype=events_hat.dtype)
        elif events_hat.size(-1) == 1:
            result["events_pred"] = (events_hat >= 0.5).to(dtype=events_hat.dtype)
        else:
            event_idx = events_hat.argmax(dim=-1)
            result["events_pred_idx"] = event_idx
            result["events_pred"] = F.one_hot(event_idx, num_classes=events_hat.size(-1)).to(dtype=events_hat.dtype)
        self._copy_prediction_keys(result, batch, ("head_timestamps",))
        self._copy_prediction_keys(
            result,
            outputs,
            ("events", "events_mask"),
        )
        return result

    def _get_prediction_method(self, prediction_type: PredictionType) -> Callable[[tensors_dict], tensors_dict]:
        if prediction_type == PredictionType.RECONSTRUCTION:
            return self._predict_reconstruction
        if prediction_type == PredictionType.PREDICT:
            return self._predict_forecast
        if prediction_type == PredictionType.SIMULATE:
            return self._predict_simulation
        if prediction_type == PredictionType.DENOISE:
            return self._predict_denoise
        if prediction_type == PredictionType.PREDICT_EVENT:
            return self._predict_events
        raise ValueError(f"Unsupported prediction_type={prediction_type!r}.")




    def _window_tensor(
        self,
        tensor: torch.Tensor,
        starts: np.ndarray,
        window_size: int,
        head_window_size: int = 0,
        return_starts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
        starts = np.asarray(starts, dtype=int)
        if starts.ndim != 1:
            raise ValueError("starts must be a 1D array.")
        if window_size <= 0:
            raise ValueError("window_size must be > 0.")
        if head_window_size < 0:
            raise ValueError("head_window_size must be >= 0.")

        max_start = tensor.shape[0] - window_size - head_window_size
        valid = (starts >= 0) & (starts <= max_start)
        starts_valid = starts[valid]

        if len(starts_valid) == 0:
            raise ValueError(
                "No valid windows after applying window/head constraints. "
                f"tensor_len={tensor.shape[0]}, window_size={window_size}, head_window_size={head_window_size}."
            )

        if len(starts_valid) == 1 and window_size >= tensor.shape[0]:
            out = tensor.unsqueeze(0)
        else:
            out = torch.stack([tensor[s : s + window_size] for s in starts_valid], dim=0)

        if return_starts:
            return out, starts_valid
        return out

    def _to_time_seconds(self, data: DataFrame) -> np.ndarray:
        if self.time_col == "index":
            series = pd.Series(data.index, index=data.index)
        else:
            series = data[self.time_col]

        dt = pd.to_datetime(series, errors="coerce")
        if dt.notna().any():
            valid_dt = dt.notna().to_numpy()
            values = dt.astype("int64").to_numpy(dtype=np.float64) / 1e9
            values = np.where(valid_dt, values, np.nan)
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)

        finite = np.isfinite(values)
        if not finite.any():
            values = np.arange(len(series), dtype=np.float64)
        else:
            first = values[np.flatnonzero(finite)[0]].item()
            values = np.where(finite, values, first)
        return values
    
    def _to_status_numeric(self, data: DataFrame, ts0_seconds: float) -> np.ndarray | None:
        if not self.status_cols:
            return None

        out = np.zeros((len(data), len(self.status_cols)), dtype=np.float32)
        for j, col in enumerate(self.status_cols):
            s = data[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                valid_dt = s.notna().to_numpy()
                vals = s.astype("int64").to_numpy(dtype=np.float64) / 1e9
                vals = np.where(valid_dt, vals, np.nan)
                vals = (vals - ts0_seconds) / self.timestamp_scale
            else:
                # Try datetime parse for object-like status columns.
                if s.dtype == object:
                    dt = pd.to_datetime(s, errors="coerce")
                    if dt.notna().sum() >= max(1, int(0.5 * len(s))):
                        valid_dt = dt.notna().to_numpy()
                        vals = dt.astype("int64").to_numpy(dtype=np.float64) / 1e9
                        vals = np.where(valid_dt, vals, np.nan)
                        vals = (vals - ts0_seconds) / self.timestamp_scale
                    else:
                        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
                else:
                    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            out[:, j] = vals.astype(np.float32)
        return out
    
    def _make_dataset(
        self,
        data: DataFrame,
        window_size: int,
        window_stride: int,
        head_window_size: int | None,
    ) -> TensorDataset:
        if len(data) == 0:
            raise ValueError("_make_dataset: received empty dataframe.")

        if self.time_col != "index":
            data = data.sort_values(self.time_col).reset_index(drop=True)
        else:
            data = data.sort_index().copy()

        x_np = data[self.input_cols].to_numpy(dtype=np.float32)
        mask_np = (~np.isnan(x_np)).astype(np.float32)
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

        ts_seconds = self._to_time_seconds(data)
        ts_np = ((ts_seconds - ts_seconds[0]) / self.timestamp_scale).astype(np.float32)
        ts0 = float(ts_seconds[0])

        x = torch.tensor(x_np, dtype=torch.float32)
        mask = torch.tensor(mask_np, dtype=torch.float32)
        timestamps = torch.tensor(ts_np, dtype=torch.float32)

        static = None
        if self.context_cols:
            static_np = data[self.context_cols].to_numpy(dtype=np.float32)
            static_np = np.nan_to_num(static_np, nan=0.0, posinf=0.0, neginf=0.0)
            static = torch.tensor(static_np, dtype=torch.float32)

        state_pred = None
        status_np = self._to_status_numeric(data, ts0_seconds=ts0)
        if status_np is not None:
            state_pred = torch.tensor(status_np, dtype=torch.float32)

        head_size = int(head_window_size or 0)
        if head_size < 0:
            raise ValueError("head_window_size must be >= 0.")

        if window_size is None or window_size <= 0:
            window_size = len(data) - head_size if head_size > 0 else len(data)
            if window_size <= 0:
                raise ValueError(
                    "window_size is too small for the requested head_window_size "
                    f"(len(data)={len(data)}, head_window_size={head_size})."
                )
        if window_stride <= 0:
            window_stride = 1

        needs_head_cost = "events" in self.heads and self.limit_events is not None
        if head_size > 0 and needs_head_cost and not self.cost_cols:
            raise ValueError(
                "head_window_size > 0 requires `self.cost_cols` with at least one feature "
                "for limit_events event targets."
            )

        if window_size >= len(data):
            starts = np.array([0], dtype=int)
            window_size = len(data)
        else:
            starts = np.arange(0, len(data) - window_size + 1, window_stride, dtype=int)

        x_seq, starts = self._window_tensor(
            x,
            starts,
            window_size,
            head_window_size=head_size,
            return_starts=True,
        )
        ts_seq = self._window_tensor(timestamps, starts, window_size, head_window_size=head_size)
        mask_seq = self._window_tensor(mask, starts, window_size, head_window_size=head_size)

        kwargs: dict[str, torch.Tensor] = {
            "x": x_seq,
            "timestamps": ts_seq,
            "mask": mask_seq,
        }

        if static is not None:
            kwargs["static_feats"] = torch.stack([static[s] for s in starts], dim=0)
        if state_pred is not None:
            kwargs["state_pred"] = self._window_tensor(state_pred, starts, window_size, head_window_size=head_size)
        if head_size > 0:
            head_timestamps = torch.stack(
                [timestamps[s + window_size : s + window_size + head_size] for s in starts],
                dim=0,
            )
            if head_timestamps.shape[1] != head_size:
                raise ValueError(
                    "Incomplete head window detected while building dataset. "
                    f"expected head size={head_size}, got head_timestamps={head_timestamps.shape}."
                )
            series_last_ts = timestamps[-1]
            if torch.any(head_timestamps[:, -1] > series_last_ts):
                raise ValueError(
                    "Head window exceeds last timestamp of the series. "
                    f"series_last_ts={float(series_last_ts.item())}."
                )
            kwargs["head_timestamps"] = head_timestamps
            if self.cost_cols:
                cost_idx = [self.input_cols.index(col) for col in self.cost_cols]
                head_cost = torch.stack(
                    [x[s + window_size : s + window_size + head_size, cost_idx] for s in starts],
                    dim=0,
                )
                mask_head_cost = torch.stack(
                    [mask[s + window_size : s + window_size + head_size, cost_idx] for s in starts],
                    dim=0,
                )
                if head_cost.shape[1] != head_size:
                    raise ValueError(
                        "Incomplete head cost window detected while building dataset. "
                        f"expected head size={head_size}, got head_cost={head_cost.shape}."
                    )
                kwargs["head_cost"] = head_cost
                kwargs['mask_head_cost'] = mask_head_cost

        ds = TensorDataset(*kwargs.values())
        ds.keys = list(kwargs.keys())
        return ds

    @staticmethod
    def _split_counts(
        n: int,
        train_frac: float,
        val_frac: float,
        test_frac: float
    ) -> tuple[int, int, int]:
        validate = val_frac > 0
        if validate:
            raw = np.array([train_frac, val_frac, test_frac], dtype=float)
        else:
            raw = np.array([train_frac + val_frac, 0.0, test_frac], dtype=float)

        if raw.sum() <= 0:
            raw = np.array([1.0, 0.0, 0.0], dtype=float)

        k = raw / max(raw.sum(), 1e-8)
        counts = np.floor(k * n).astype(int)

        while counts.sum() < n:
            gaps = k * n - counts
            counts[int(np.argmax(gaps))] += 1

        while counts.sum() > n:
            counts[int(np.argmax(counts))] -= 1

        counts = np.maximum(counts, 0)
        if validate:
            n_train, n_val, n_test = counts.tolist()
            if n >= 3:
                if n_val == 0:
                    n_val, n_train = 1, max(n_train - 1, 0)
                if n_test == 0:
                    if n_train > n_val:
                        n_test, n_train = 1, max(n_train - 1, 0)
                    else:
                        n_test, n_val = 1, max(n_val - 1, 0)
        else:
            n_train, n_val, n_test = counts.tolist()

        extra = n_train + n_val + n_test - n
        if extra > 0:
            for _ in range(extra):
                if n_train >= max(n_val, n_test) and n_train > 0:
                    n_train -= 1
                elif n_val >= n_test and n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1
        return n_train, n_val, n_test
