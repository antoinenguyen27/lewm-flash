"""Exact CEM variant without loop-internal CPU materialization."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import torch


def prepare_init_action(
    model: Any,
    info_dict: dict[str, Any],
    init_action: torch.Tensor | None,
    horizon: int,
    *,
    n_envs: int,
    action_dim: int,
) -> torch.Tensor:
    """Match stable-worldmodel CEM warm-start padding semantics."""
    if init_action is not None:
        assert init_action.shape[0] == n_envs, (
            f"init_action batch size {init_action.shape[0]} != n_envs {n_envs}"
        )
        assert init_action.shape[2] == action_dim, (
            f"init_action action_dim {init_action.shape[2]} != action_dim {action_dim}"
        )

    n_prev = init_action.shape[1] if init_action is not None else 0
    remaining = horizon - n_prev
    if remaining <= 0:
        return init_action

    if not hasattr(model, "get_action"):
        device = init_action.device if init_action is not None else "cpu"
        tail = torch.zeros(n_envs, remaining, action_dim, device=device)
        if init_action is not None:
            return torch.cat([init_action, tail], dim=1)
        return tail

    with torch.no_grad():
        tail = model.get_action(
            info_dict,
            horizon=remaining,
            prefix_actions=init_action,
        )
    if init_action is not None:
        return torch.cat([init_action.to(tail.device), tail], dim=1)
    return tail


class CEMExactSolver:
    """CEM solver matching upstream semantics while deferring CPU logging sync."""

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
        warm_start: bool = False,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = torch.device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)
        self.warm_start = bool(warm_start)
        self.callbacks = list(callbacks) if callbacks else []
        self._warm_start_mean: torch.Tensor | None = None
        self._warm_start_var: torch.Tensor | None = None
        self._warm_start_signature: tuple[Any, ...] | None = None
        self._last_warm_start_info: dict[str, Any] = {
            "enabled": self.warm_start,
            "used": False,
            "reset": False,
            "reset_reason": None,
        }
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
        self.reset_warm_start_state("configure")
        if "Box" not in type(action_space).__name__:
            warnings.warn(
                f"Action space is not Box-like, got {type(action_space)}. "
                "CEMExactSolver may not work as expected.",
                RuntimeWarning,
                stacklevel=2,
            )

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
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.solve(*args, **kwargs)

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

    @property
    def receding_horizon(self) -> int:
        return int(getattr(self._config, "receding_horizon", self.horizon))

    @property
    def last_warm_start_info(self) -> dict[str, Any]:
        return dict(self._last_warm_start_info)

    def reset_warm_start_state(self, reason: str = "manual") -> None:
        self._warm_start_mean = None
        self._warm_start_var = None
        self._warm_start_signature = None
        self._last_warm_start_info = {
            "enabled": self.warm_start,
            "used": False,
            "reset": True,
            "reset_reason": reason,
        }

    def _distribution_signature(self, mean: torch.Tensor, var: torch.Tensor) -> tuple[Any, ...]:
        return (
            int(mean.shape[0]),
            int(mean.shape[1]),
            int(mean.shape[2]),
            int(var.shape[0]),
            int(var.shape[1]),
            int(var.shape[2]),
            int(self.batch_size),
            self.receding_horizon,
            torch.device(mean.device),
            mean.dtype,
            torch.device(var.device),
            var.dtype,
        )

    def _state_matches_distribution(self, mean: torch.Tensor, var: torch.Tensor) -> bool:
        expected = self._distribution_signature(mean, var)
        return (
            self._warm_start_signature == expected
            and self._warm_start_mean is not None
            and self._warm_start_var is not None
            and tuple(self._warm_start_mean.shape) == tuple(mean.shape)
            and tuple(self._warm_start_var.shape) == tuple(var.shape)
        )

    def _shift_warm_start_distribution(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize CEM distribution from the shifted previous solve state."""
        info = {
            "enabled": self.warm_start,
            "used": False,
            "reset": False,
            "reset_reason": None,
            "receding_horizon": self.receding_horizon,
        }
        if not self.warm_start:
            self._last_warm_start_info = info
            return mean, var

        shift = self.receding_horizon
        if shift <= 0 or shift >= self.horizon:
            info["reset"] = shift >= self.horizon
            info["reset_reason"] = "no_unexecuted_tail" if shift >= self.horizon else "invalid_shift"
            self._last_warm_start_info = info
            return mean, var

        if not self._state_matches_distribution(mean, var):
            had_state = self._warm_start_mean is not None or self._warm_start_var is not None
            self._warm_start_mean = None
            self._warm_start_var = None
            self._warm_start_signature = None
            info["reset"] = had_state
            info["reset_reason"] = "signature_mismatch" if had_state else "no_previous_state"
            self._last_warm_start_info = info
            return mean, var

        assert self._warm_start_mean is not None
        assert self._warm_start_var is not None
        shifted_mean = torch.zeros_like(mean)
        shifted_var = self.var_scale * torch.ones_like(var)
        shifted_mean[:, :-shift] = self._warm_start_mean[:, shift:].to(
            device=mean.device,
            dtype=mean.dtype,
        )
        shifted_var[:, :-shift] = self._warm_start_var[:, shift:].to(
            device=var.device,
            dtype=var.dtype,
        )
        info["used"] = True
        self._last_warm_start_info = info
        return shifted_mean, shifted_var

    def _remember_warm_start_distribution(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
    ) -> None:
        if not self.warm_start:
            return
        self._warm_start_signature = self._distribution_signature(mean, var)
        self._warm_start_mean = mean.detach().clone()
        self._warm_start_var = var.detach().clone()

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
        mean, var = self._shift_warm_start_distribution(mean, var)

        for cb in self.callbacks:
            cb.reset()

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                if torch.is_tensor(value):
                    target_dtype = self.dtype if value_batch.is_floating_point() else None
                    value_batch = (
                        value_batch.to(device=self.device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(current_bs, self.num_samples, *value_batch.shape[1:])
                    )
                elif isinstance(value, np.ndarray):
                    value_batch = np.repeat(value_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[key] = value_batch

            final_batch_cost_gpu = None
            for cb in self.callbacks:
                cb.start_batch()

            for step in range(self.n_steps):
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                costs = self.model.get_cost(expanded_infos, candidates)
                assert isinstance(costs, torch.Tensor), (
                    f"Expected cost to be a torch.Tensor, got {type(costs)}"
                )
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == self.num_samples
                ), f"Expected cost shape ({current_bs}, {self.num_samples}), got {costs.shape}"

                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )
                topk_candidates = candidates[batch_indices, topk_inds]
                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

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
                    )

                final_batch_cost_gpu = topk_vals.mean(dim=1)

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            assert final_batch_cost_gpu is not None
            outputs["costs"].extend(final_batch_cost_gpu.detach().cpu().tolist())

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        outputs["warm_start"] = self.last_warm_start_info
        self._remember_warm_start_distribution(mean, var)

        if self.callbacks:
            outputs["callbacks"] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs["callbacks"][cb.output_key] = cb.history

        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs
