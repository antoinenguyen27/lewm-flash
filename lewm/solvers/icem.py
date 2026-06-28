"""Local iCEM variant for planner-changing LeWM experiments."""

from __future__ import annotations

import math
import time
import warnings
from typing import Any

import numpy as np
import torch

from .cem_exact import prepare_init_action


COLORED_NOISE_BETA = {
    "none": 0.0,
    "pink": 1.0,
    "brown": 2.0,
}


class ICEMStepSummaryCallback:
    """Collect compact per-step iCEM convergence summaries."""

    output_key = "icem_step_summary"

    def __init__(self, saturation_epsilon: float = 1e-5) -> None:
        self.saturation_epsilon = float(saturation_epsilon)
        self.action_low: torch.Tensor | None = None
        self.action_high: torch.Tensor | None = None
        self.history: list[dict[str, float | int | None]] = []
        self._records: dict[int, list[dict[str, Any]]] = {}

    def configure(
        self,
        *,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
    ) -> None:
        self.action_low = action_low
        self.action_high = action_high

    def reset(self) -> None:
        self.history = []
        self._records = {}

    def start_batch(self) -> None:
        return

    def __call__(
        self,
        *,
        step: int,
        candidates: torch.Tensor,
        costs: torch.Tensor,
        topk_vals: torch.Tensor,
        topk_inds: torch.Tensor,
        topk_candidates: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        prev_mean: torch.Tensor,
        prev_var: torch.Tensor,
        n_samples: int | None = None,
    ) -> None:
        del topk_inds, topk_candidates, prev_var
        batch_size = int(costs.shape[0])
        step_records = self._records.setdefault(int(step), [])
        record: dict[str, Any] = {
            "batch_size": batch_size,
            "sample_count": int(n_samples if n_samples is not None else costs.shape[1]),
            "best_cost_mean": topk_vals[:, 0].mean().detach(),
            "mean_candidate_cost_mean": costs[:, 0].mean().detach(),
            "topk_mean_cost_mean": topk_vals.mean(dim=1).mean().detach(),
            "topk_std_cost_mean": topk_vals.std(dim=1, unbiased=False).mean().detach(),
            "mean_delta_norm_mean": (mean - prev_mean).flatten(1).norm(dim=1).mean().detach(),
            "var_mean": var.mean().detach(),
            "var_min": var.min().detach(),
            "var_max": var.max().detach(),
        }

        if self.action_low is not None and self.action_high is not None:
            low = self.action_low.to(device=candidates.device, dtype=candidates.dtype)
            high = self.action_high.to(device=candidates.device, dtype=candidates.dtype)
            saturated = (candidates - low).abs().le(self.saturation_epsilon) | (
                candidates - high
            ).abs().le(self.saturation_epsilon)
            record["saturation_count"] = saturated.sum().detach()
            record["saturation_total"] = int(saturated.numel())
        else:
            record["saturation_count"] = None
            record["saturation_total"] = 0

        step_records.append(record)

    @staticmethod
    def _as_float(value: torch.Tensor) -> float:
        return float(value.detach().cpu().item())

    def end_solve(self) -> None:
        rows: list[dict[str, float | int | None]] = []
        mean_keys = (
            "best_cost_mean",
            "mean_candidate_cost_mean",
            "topk_mean_cost_mean",
            "topk_std_cost_mean",
            "mean_delta_norm_mean",
            "var_mean",
        )
        for step in sorted(self._records):
            records = self._records[step]
            total_batch = sum(int(record["batch_size"]) for record in records)
            row: dict[str, float | int | None] = {
                "step": step,
                "sample_count": int(records[0]["sample_count"]),
            }
            for key in mean_keys:
                weighted = sum(
                    record[key] * int(record["batch_size"]) for record in records
                ) / max(total_batch, 1)
                row[key] = self._as_float(weighted)

            row["var_min"] = self._as_float(
                torch.stack([record["var_min"] for record in records]).min()
            )
            row["var_max"] = self._as_float(
                torch.stack([record["var_max"] for record in records]).max()
            )

            saturation_counts = [
                record["saturation_count"]
                for record in records
                if record.get("saturation_count") is not None
            ]
            saturation_total = sum(int(record.get("saturation_total", 0)) for record in records)
            if saturation_counts and saturation_total:
                saturation_count = torch.stack(saturation_counts).sum()
                row["action_saturation_rate"] = self._as_float(
                    saturation_count / float(saturation_total)
                )
            else:
                row["action_saturation_rate"] = None

            rows.append(row)
        self.history = rows


def _normalize_colored_noise(noise: torch.Tensor) -> torch.Tensor:
    std = noise.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return noise / std


def _colored_noise_along_time(
    *,
    batch_size: int,
    num_samples: int,
    channels: int,
    time_steps: int,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    white = torch.randn(
        batch_size,
        num_samples,
        channels,
        time_steps,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if beta <= 0.0 or time_steps <= 1:
        return white

    freqs = torch.fft.rfftfreq(time_steps, device=device).to(dtype)
    freqs[0] = 1.0
    scale = freqs.pow(-beta / 2.0)
    if scale.numel() > 1:
        scale[0] = scale[1]
    colored = torch.fft.irfft(torch.fft.rfft(white, dim=-1) * scale, n=time_steps, dim=-1)
    return _normalize_colored_noise(colored)


def make_colored_noise(
    *,
    batch_size: int,
    num_samples: int,
    horizon: int,
    action_dim: int,
    beta: float,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    action_block: int = 1,
    time_axis: str = "block",
) -> torch.Tensor:
    """Sample action noise with optional temporal coloring.

    ``time_axis="block"`` matches upstream stable-worldmodel iCEM: each flattened
    action dimension is correlated across solver horizon steps. ``time_axis="raw"``
    colors across ``horizon * action_block`` raw actions, then flattens each raw
    action block back to LeWM's solver action dimension.
    """
    device = torch.device(device)
    if time_axis not in {"block", "raw"}:
        raise ValueError(f"colored noise time_axis must be 'block' or 'raw', got {time_axis!r}")
    if action_block <= 0:
        raise ValueError(f"action_block must be positive, got {action_block}")
    if beta <= 0.0:
        return torch.randn(
            batch_size,
            num_samples,
            horizon,
            action_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    if time_axis == "raw" and action_block > 1:
        if action_dim % action_block != 0:
            raise ValueError(
                f"action_dim={action_dim} is not divisible by action_block={action_block}"
            )
        raw_action_dim = action_dim // action_block
        raw_time = horizon * action_block
        raw = _colored_noise_along_time(
            batch_size=batch_size,
            num_samples=num_samples,
            channels=raw_action_dim,
            time_steps=raw_time,
            beta=beta,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        raw = raw.transpose(-1, -2).contiguous()
        return raw.reshape(batch_size, num_samples, horizon, action_dim)

    block = _colored_noise_along_time(
        batch_size=batch_size,
        num_samples=num_samples,
        channels=action_dim,
        time_steps=horizon,
        beta=beta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return block.transpose(-1, -2).contiguous()


class ICEMSolver:
    """Improved CEM solver with isolated within-solve feature flags."""

    defer_cpu_materialization = True

    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        elite_retention: int = 0,
        colored_noise: str = "none",
        colored_noise_time: str = "raw",
        momentum_alpha: float = 0.0,
        sample_decay: float = 1.0,
        sample_count_bucket_size: int = 0,
        return_best_elite: bool = False,
        clip_actions: bool = False,
        callbacks: list[Any] | None = None,
    ) -> None:
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if num_samples < topk:
            raise ValueError(f"num_samples={num_samples} must be >= topk={topk}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if elite_retention < 0:
            raise ValueError(f"elite_retention must be non-negative, got {elite_retention}")
        if colored_noise not in COLORED_NOISE_BETA:
            raise ValueError(
                f"colored_noise must be one of {sorted(COLORED_NOISE_BETA)}, got {colored_noise!r}"
            )
        if not 0.0 <= momentum_alpha < 1.0:
            raise ValueError(f"momentum_alpha must satisfy 0 <= alpha < 1, got {momentum_alpha}")
        if sample_decay < 1.0:
            raise ValueError(f"sample_decay must be >= 1.0, got {sample_decay}")
        if sample_count_bucket_size < 0:
            raise ValueError(
                "sample_count_bucket_size must be non-negative, "
                f"got {sample_count_bucket_size}"
            )

        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = torch.device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)
        self.elite_retention = elite_retention
        self.colored_noise = colored_noise
        self.colored_noise_time = colored_noise_time
        self.momentum_alpha = float(momentum_alpha)
        self.sample_decay = float(sample_decay)
        self.sample_count_bucket_size = int(sample_count_bucket_size)
        self.return_best_elite = bool(return_best_elite)
        self.clip_actions = bool(clip_actions)
        self.callbacks = list(callbacks) if callbacks else []
        self._action_low: torch.Tensor | None = None
        self._action_high: torch.Tensor | None = None
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        if "Box" not in type(action_space).__name__:
            warnings.warn(
                f"Action space is not Box-like, got {type(action_space)}. "
                "ICEMSolver may not work as expected.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if self.clip_actions and hasattr(action_space, "low") and hasattr(action_space, "high"):
            self._action_low = torch.as_tensor(
                action_space.low[0],
                device=self.device,
                dtype=self.dtype,
            ).repeat(self._config.action_block)
            self._action_high = torch.as_tensor(
                action_space.high[0],
                device=self.device,
                dtype=self.dtype,
            ).repeat(self._config.action_block)
        for cb in self.callbacks:
            configure = getattr(cb, "configure", None)
            if callable(configure):
                configure(action_low=self._action_low, action_high=self._action_high)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def action_block(self) -> int:
        return int(getattr(self._config, "action_block", 1))

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.solve(*args, **kwargs)

    def sample_count_for_step(self, step: int) -> int:
        if self.sample_decay == 1.0:
            return self.num_samples
        decayed = int(math.ceil(self.num_samples * (self.sample_decay ** (-step))))
        count = max(decayed, min(self.num_samples, 2 * self.topk))
        if self.sample_count_bucket_size > 0:
            count = min(
                self.num_samples,
                int(math.ceil(count / self.sample_count_bucket_size))
                * self.sample_count_bucket_size,
            )
        return count

    def sample_counts(self) -> list[int]:
        return [self.sample_count_for_step(step) for step in range(self.n_steps)]

    def model_cost_evals_per_solve(self) -> int:
        return sum(self.sample_counts())

    def init_action_distrib(
        self,
        n_envs: int,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim],
            dtype=self.dtype,
        )
        mean = (
            torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype)
            if actions is None
            else actions
        )
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros(
                [n_envs, remaining, self.action_dim],
                dtype=self.dtype,
                device=device,
            )
            mean = torch.cat([mean, new_mean], dim=1).to(device)
        return mean, var

    def inject_retained_elites(
        self,
        candidates: torch.Tensor,
        previous_elites: torch.Tensor | None,
    ) -> torch.Tensor:
        if previous_elites is None or self.elite_retention <= 0:
            return candidates
        n_inject = min(self.elite_retention, previous_elites.shape[1], candidates.shape[1] - 1)
        if n_inject <= 0:
            return candidates
        candidates[:, 1 : 1 + n_inject] = previous_elites[:, :n_inject]
        return candidates

    def _expand_info_for_samples(
        self,
        info_dict: dict[str, Any],
        start_idx: int,
        end_idx: int,
        n_samples: int,
    ) -> dict[str, Any]:
        current_bs = end_idx - start_idx
        expanded_infos = {}
        for key, value in info_dict.items():
            value_batch = value[start_idx:end_idx]
            if torch.is_tensor(value):
                target_dtype = self.dtype if value_batch.is_floating_point() else None
                value_batch = (
                    value_batch.to(device=self.device, dtype=target_dtype)
                    .unsqueeze(1)
                    .expand(current_bs, n_samples, *value_batch.shape[1:])
                )
            elif isinstance(value, np.ndarray):
                value_batch = np.repeat(value_batch[:, None, ...], n_samples, axis=1)
            expanded_infos[key] = value_batch
        return expanded_infos

    def _sample_noise(self, batch_size: int, n_samples: int) -> torch.Tensor:
        return make_colored_noise(
            batch_size=batch_size,
            num_samples=n_samples,
            horizon=self.horizon,
            action_dim=self.action_dim,
            beta=COLORED_NOISE_BETA[self.colored_noise],
            device=self.device,
            dtype=self.dtype,
            generator=self.torch_gen,
            action_block=self.action_block,
            time_axis=self.colored_noise_time,
        )

    @torch.inference_mode()
    def solve(
        self,
        info_dict: dict[str, Any],
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()
        outputs: dict[str, Any] = {
            "costs": [],
            "mean": [],
            "var": [],
        }
        total_envs = len(next(iter(info_dict.values())))

        init_action = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )

        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        for cb in self.callbacks:
            cb.reset()

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            previous_elites = None
            best_elite = None
            final_batch_cost_gpu = None
            cached_cost_inputs: dict[str, torch.Tensor] = {}

            for cb in self.callbacks:
                cb.start_batch()

            for step, n_samples in enumerate(self.sample_counts()):
                expanded_infos = self._expand_info_for_samples(
                    info_dict,
                    start_idx,
                    end_idx,
                    n_samples,
                )
                expanded_infos.update(cached_cost_inputs)
                noise = self._sample_noise(current_bs, n_samples)
                candidates = noise * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean
                candidates = self.inject_retained_elites(candidates, previous_elites)
                if self._action_low is not None and self._action_high is not None:
                    candidates = candidates.clamp(self._action_low, self._action_high)

                costs = self.model.get_cost(expanded_infos, candidates)
                for cache_key in ("emb", "goal_emb"):
                    cached_value = expanded_infos.get(cache_key)
                    if torch.is_tensor(cached_value) and cached_value.ndim == 3:
                        cached_cost_inputs[cache_key] = cached_value
                assert isinstance(costs, torch.Tensor), (
                    f"Expected cost to be a torch.Tensor, got {type(costs)}"
                )
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == n_samples
                ), f"Expected cost shape ({current_bs}, {n_samples}), got {costs.shape}"

                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )
                topk_candidates = candidates[batch_indices, topk_inds]
                previous_elites = topk_candidates
                best_elite = topk_candidates[:, 0]

                prev_mean = batch_mean
                prev_var = batch_var
                elite_mean = topk_candidates.mean(dim=1)
                elite_var = topk_candidates.std(dim=1)
                if self.momentum_alpha:
                    alpha = self.momentum_alpha
                    batch_mean = alpha * batch_mean + (1.0 - alpha) * elite_mean
                    batch_var = alpha * batch_var + (1.0 - alpha) * elite_var
                else:
                    batch_mean = elite_mean
                    batch_var = elite_var

                for cb in self.callbacks:
                    cb(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                        n_samples=n_samples,
                    )

                final_batch_cost_gpu = topk_vals.mean(dim=1)

            assert final_batch_cost_gpu is not None
            if self.return_best_elite:
                assert best_elite is not None
                mean[start_idx:end_idx] = best_elite
            else:
                mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost_gpu.detach().cpu().tolist())

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        if self.callbacks:
            outputs["callbacks"] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs["callbacks"][cb.output_key] = cb.history

        print(f"iCEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs


__all__ = ["ICEMSolver", "ICEMStepSummaryCallback", "make_colored_noise"]
