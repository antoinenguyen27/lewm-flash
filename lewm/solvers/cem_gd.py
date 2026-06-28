"""CEM/iCEM seed search with eager action-gradient refinement."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .cem_exact import prepare_init_action
from .icem import COLORED_NOISE_BETA, make_colored_noise


@dataclass
class SeedSearchResult:
    actions: torch.Tensor
    costs: torch.Tensor
    seed_solver: str
    model_cost_evals: int


@dataclass
class DummyRefinementResult:
    before: float
    after: float
    accepted: bool
    summary: dict[str, Any]


class CEMGDSolver:
    """Run a CEM/iCEM proposal, then refine evaluated elites with action GD."""

    defer_cpu_materialization = True

    def __init__(
        self,
        model: Any | None = None,
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
        gd_seed_solver: str = "cem",
        gd_seed_source: str = "best_elite",
        gd_refine_topk: int = 1,
        gd_steps: int = 1,
        gd_lr: float = 0.01,
        gd_compile_cost: bool = False,
        gd_backtracking: bool = True,
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
        if gd_seed_solver not in {"cem", "icem"}:
            raise ValueError(f"gd_seed_solver must be 'cem' or 'icem', got {gd_seed_solver!r}")
        if gd_seed_source not in {"best_elite", "mean"}:
            raise ValueError(
                f"gd_seed_source must be 'best_elite' or 'mean', got {gd_seed_source!r}"
            )
        if gd_refine_topk <= 0:
            raise ValueError(f"gd_refine_topk must be positive, got {gd_refine_topk}")
        if gd_refine_topk > topk:
            raise ValueError(f"gd_refine_topk={gd_refine_topk} must be <= topk={topk}")
        if gd_steps < 0:
            raise ValueError(f"gd_steps must be non-negative, got {gd_steps}")
        if gd_lr <= 0.0:
            raise ValueError(f"gd_lr must be positive, got {gd_lr}")

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
        self.gd_seed_solver = gd_seed_solver
        self.gd_seed_source = gd_seed_source
        self.gd_refine_topk = int(gd_refine_topk)
        self.gd_steps = int(gd_steps)
        self.gd_lr = float(gd_lr)
        self.gd_compile_cost = bool(gd_compile_cost)
        self.gd_backtracking = bool(gd_backtracking)
        self.callbacks = list(callbacks) if callbacks else []
        self._action_low: torch.Tensor | None = None
        self._action_high: torch.Tensor | None = None
        try:
            self._dtype = next(model.parameters()).dtype if model is not None else torch.float32
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
                "CEMGDSolver may not work as expected.",
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
        if self.gd_seed_solver == "cem" or self.sample_decay == 1.0:
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
        beta = COLORED_NOISE_BETA[self.colored_noise] if self.gd_seed_solver == "icem" else 0.0
        return make_colored_noise(
            batch_size=batch_size,
            num_samples=n_samples,
            horizon=self.horizon,
            action_dim=self.action_dim,
            beta=beta,
            device=self.device,
            dtype=self.dtype,
            generator=self.torch_gen,
            action_block=self.action_block,
            time_axis=self.colored_noise_time,
        )

    def inject_retained_elites(
        self,
        candidates: torch.Tensor,
        previous_elites: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.gd_seed_solver != "icem"
            or previous_elites is None
            or self.elite_retention <= 0
        ):
            return candidates
        n_inject = min(self.elite_retention, previous_elites.shape[1], candidates.shape[1] - 1)
        if n_inject <= 0:
            return candidates
        candidates[:, 1 : 1 + n_inject] = previous_elites[:, :n_inject]
        return candidates

    @staticmethod
    def select_seed_elites(
        candidates: torch.Tensor,
        costs: torch.Tensor,
        *,
        refine_topk: int,
        seed_solver: str = "manual",
    ) -> SeedSearchResult:
        topk_vals, topk_inds = torch.topk(costs, k=refine_topk, dim=1, largest=False)
        batch_indices = (
            torch.arange(candidates.shape[0], device=candidates.device)
            .unsqueeze(1)
            .expand(-1, refine_topk)
        )
        return SeedSearchResult(
            actions=candidates[batch_indices, topk_inds],
            costs=topk_vals,
            seed_solver=seed_solver,
            model_cost_evals=int(candidates.shape[1]),
        )

    def freeze_model_for_action_optimization(self, model: Any | None = None) -> None:
        target = self.model if model is None else model
        if target is None or not hasattr(target, "parameters"):
            return
        for parameter in target.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def new_action_params(seed_actions: torch.Tensor) -> torch.Tensor:
        return seed_actions.detach().clone().requires_grad_(True)

    def _sync_if_cuda(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _gd_cost_callable(self) -> Any:
        if self.model is None:
            raise RuntimeError("CEMGDSolver requires a model for gradient refinement")
        if not self.gd_compile_cost:
            eager = getattr(self.model, "_eager_get_cost_from_embeddings", None)
            if eager is not None:
                return eager
        if not hasattr(self.model, "get_cost_from_embeddings"):
            raise TypeError(
                f"CEM-GD requires get_cost_from_embeddings(), got model {type(self.model)}"
            )
        return self.model.get_cost_from_embeddings

    def _evaluate_gd_cost(
        self,
        emb: torch.Tensor,
        goal_emb: torch.Tensor,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        return self._gd_cost_callable()(emb, goal_emb, action_candidates)

    @staticmethod
    def _remember_cost_inputs(
        expanded_infos: dict[str, Any],
        cached_cost_inputs: dict[str, torch.Tensor],
    ) -> None:
        for cache_key in ("emb", "goal_emb"):
            cached_value = expanded_infos.get(cache_key)
            if torch.is_tensor(cached_value) and cached_value.ndim == 3:
                cached_cost_inputs[cache_key] = cached_value

    def _seed_source_from_mean(
        self,
        batch_mean: torch.Tensor,
        cached_cost_inputs: dict[str, torch.Tensor],
        seed_model_cost_evals: int,
    ) -> SeedSearchResult:
        if "emb" not in cached_cost_inputs or "goal_emb" not in cached_cost_inputs:
            raise RuntimeError("CEM-GD mean seed source requires cached emb and goal_emb")
        mean_actions = batch_mean.unsqueeze(1)
        costs = self._evaluate_gd_cost(
            cached_cost_inputs["emb"],
            cached_cost_inputs["goal_emb"],
            mean_actions,
        )
        return SeedSearchResult(
            actions=mean_actions,
            costs=costs,
            seed_solver=self.gd_seed_solver,
            model_cost_evals=seed_model_cost_evals + 1,
        )

    def _run_seed_search_batch(
        self,
        info_dict: dict[str, Any],
        *,
        start_idx: int,
        end_idx: int,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
    ) -> tuple[SeedSearchResult, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        current_bs = end_idx - start_idx
        previous_elites = None
        final_topk_candidates = None
        final_topk_vals = None
        cached_cost_inputs: dict[str, torch.Tensor] = {}
        seed_model_cost_evals = 0

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
            self._remember_cost_inputs(expanded_infos, cached_cost_inputs)
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
            final_topk_candidates = topk_candidates
            final_topk_vals = topk_vals

            elite_mean = topk_candidates.mean(dim=1)
            elite_var = topk_candidates.std(dim=1)
            if self.gd_seed_solver == "icem" and self.momentum_alpha:
                alpha = self.momentum_alpha
                batch_mean = alpha * batch_mean + (1.0 - alpha) * elite_mean
                batch_var = alpha * batch_var + (1.0 - alpha) * elite_var
            else:
                batch_mean = elite_mean
                batch_var = elite_var
            seed_model_cost_evals += n_samples

        assert final_topk_candidates is not None
        assert final_topk_vals is not None
        if self.gd_seed_source == "mean":
            seed = self._seed_source_from_mean(
                batch_mean,
                cached_cost_inputs,
                seed_model_cost_evals,
            )
        else:
            seed = SeedSearchResult(
                actions=final_topk_candidates[:, : self.gd_refine_topk],
                costs=final_topk_vals[:, : self.gd_refine_topk],
                seed_solver=self.gd_seed_solver,
                model_cost_evals=seed_model_cost_evals,
            )
        return seed, cached_cost_inputs, batch_mean, batch_var

    def _action_saturation_rate(self, actions: torch.Tensor) -> float | None:
        if self._action_low is None or self._action_high is None:
            return None
        low = self._action_low.to(device=actions.device, dtype=actions.dtype)
        high = self._action_high.to(device=actions.device, dtype=actions.dtype)
        saturated = (actions - low).abs().le(1e-5) | (actions - high).abs().le(1e-5)
        return float(saturated.float().mean().detach().cpu().item())

    def _refine_seed_actions(
        self,
        seed_actions: torch.Tensor,
        seed_costs: torch.Tensor,
        emb: torch.Tensor,
        goal_emb: torch.Tensor,
        *,
        enable_backtracking: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.freeze_model_for_action_optimization()
        params = self.new_action_params(seed_actions)
        optimizer = torch.optim.Adam([params], lr=self.gd_lr)
        current_costs = seed_costs.detach().clone()
        accepted_any = torch.zeros_like(current_costs, dtype=torch.bool)
        forward_times: list[float] = []
        backward_times: list[float] = []
        grad_norm_values: list[float] = []
        grad_norm_max = 0.0
        nan_or_inf_count = 0
        refine_width = int(seed_actions.shape[1])
        factors = [1.0, 0.5, 0.25, 0.1] if (
            self.gd_backtracking if enable_backtracking is None else enable_backtracking
        ) else [1.0]
        gd_model_cost_evals = 0

        if self.gd_steps == 0:
            final_actions = seed_actions.detach()
            final_costs = seed_costs.detach()
        else:
            final_actions = seed_actions.detach()
            final_costs = seed_costs.detach()
            for _ in range(self.gd_steps):
                optimizer.zero_grad(set_to_none=True)
                self._sync_if_cuda()
                started = time.perf_counter()
                with torch.enable_grad():
                    costs = self._evaluate_gd_cost(emb, goal_emb, params)
                    loss = costs.mean()
                self._sync_if_cuda()
                forward_times.append((time.perf_counter() - started) * 1000.0)
                gd_model_cost_evals += refine_width

                if not torch.isfinite(costs).all() or not torch.isfinite(loss):
                    nan_or_inf_count += int((~torch.isfinite(costs)).sum().detach().cpu().item())
                    nan_or_inf_count += int(not bool(torch.isfinite(loss).detach().cpu().item()))
                    break

                self._sync_if_cuda()
                started = time.perf_counter()
                loss.backward()
                self._sync_if_cuda()
                backward_times.append((time.perf_counter() - started) * 1000.0)

                if params.grad is None or not torch.isfinite(params.grad).all():
                    bad = 1 if params.grad is None else int(
                        (~torch.isfinite(params.grad)).sum().detach().cpu().item()
                    )
                    nan_or_inf_count += bad
                    break

                grad_norms = params.grad.detach().flatten(start_dim=2).norm(dim=2)
                grad_norm_values.extend(grad_norms.detach().cpu().flatten().tolist())
                grad_norm_max = max(grad_norm_max, float(grad_norms.max().detach().cpu().item()))

                before = params.detach().clone()
                previous_costs = current_costs.detach().clone()
                optimizer.step()
                full_step = params.detach().clone()
                update = full_step - before

                best_actions = before
                best_costs = previous_costs
                for factor in factors:
                    candidate_actions = before + factor * update
                    self._sync_if_cuda()
                    with torch.no_grad():
                        candidate_costs = self._evaluate_gd_cost(
                            emb,
                            goal_emb,
                            candidate_actions,
                        )
                    self._sync_if_cuda()
                    gd_model_cost_evals += refine_width
                    finite = torch.isfinite(candidate_costs)
                    if not bool(finite.all().detach().cpu().item()):
                        nan_or_inf_count += int((~finite).sum().detach().cpu().item())
                    improved = finite & (candidate_costs < best_costs)
                    best_actions = torch.where(
                        improved[..., None, None],
                        candidate_actions,
                        best_actions,
                    )
                    best_costs = torch.where(improved, candidate_costs, best_costs)

                accepted_any |= best_costs < previous_costs
                current_costs = best_costs.detach()
                with torch.no_grad():
                    params.copy_(best_actions)
                final_actions = params.detach().clone()
                final_costs = current_costs

        final_accept = torch.isfinite(final_costs) & (final_costs < seed_costs)
        accepted_actions = torch.where(
            final_accept[..., None, None],
            final_actions,
            seed_actions.detach(),
        )
        accepted_costs = torch.where(final_accept, final_costs, seed_costs.detach())
        candidate_count = int(seed_actions.shape[0] * seed_actions.shape[1])
        accept_count = int(final_accept.sum().detach().cpu().item())
        cost_delta = (accepted_costs - seed_costs.detach()).detach()
        peak_memory = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        summary = {
            "gd_seed_solver": self.gd_seed_solver,
            "gd_seed_source": self.gd_seed_source,
            "gd_refine_topk": self.gd_refine_topk,
            "gd_steps": self.gd_steps,
            "gd_lr": self.gd_lr,
            "gd_compile_cost": self.gd_compile_cost,
            "gd_forward_time_ms_mean": (
                sum(forward_times) / len(forward_times) if forward_times else 0.0
            ),
            "gd_grad_norm_mean": (
                sum(grad_norm_values) / len(grad_norm_values) if grad_norm_values else 0.0
            ),
            "gd_grad_norm_max": grad_norm_max,
            "gd_action_saturation_rate": self._action_saturation_rate(accepted_actions),
            "gd_backward_time_ms_mean": (
                sum(backward_times) / len(backward_times) if backward_times else 0.0
            ),
            "gd_peak_memory_mb": peak_memory,
            "gd_cost_delta_mean": float(cost_delta.mean().detach().cpu().item()),
            "gd_accept_rate": accept_count / candidate_count if candidate_count else 0.0,
            "gd_accept_count": accept_count,
            "gd_candidate_count": candidate_count,
            "gd_step_accept_rate": (
                float(accepted_any.float().mean().detach().cpu().item())
                if accepted_any.numel()
                else 0.0
            ),
            "gd_nan_or_inf_count": nan_or_inf_count,
            "gd_model_cost_evals": gd_model_cost_evals,
        }
        return accepted_actions.detach(), accepted_costs.detach(), summary

    @staticmethod
    def _combine_gd_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
        if not summaries:
            return {}

        def mean_field(key: str) -> float | None:
            values = [
                float(summary[key])
                for summary in summaries
                if isinstance(summary.get(key), (int, float))
            ]
            return sum(values) / len(values) if values else None

        candidate_count = sum(int(summary.get("gd_candidate_count", 0)) for summary in summaries)
        accept_count = sum(int(summary.get("gd_accept_count", 0)) for summary in summaries)
        cost_delta_total = sum(
            float(summary.get("gd_cost_delta_mean", 0.0))
            * int(summary.get("gd_candidate_count", 0))
            for summary in summaries
        )
        model_cost_evals = max(int(summary.get("gd_model_cost_evals", 0)) for summary in summaries)
        peak_values = [
            float(summary["gd_peak_memory_mb"])
            for summary in summaries
            if isinstance(summary.get("gd_peak_memory_mb"), (int, float))
        ]
        saturation = mean_field("gd_action_saturation_rate")
        return {
            "gd_seed_solver": summaries[0].get("gd_seed_solver"),
            "gd_seed_source": summaries[0].get("gd_seed_source"),
            "gd_refine_topk": summaries[0].get("gd_refine_topk"),
            "gd_steps": summaries[0].get("gd_steps"),
            "gd_lr": summaries[0].get("gd_lr"),
            "gd_compile_cost": summaries[0].get("gd_compile_cost"),
            "gd_forward_time_ms_mean": mean_field("gd_forward_time_ms_mean"),
            "gd_grad_norm_mean": mean_field("gd_grad_norm_mean"),
            "gd_grad_norm_max": max(
                float(summary.get("gd_grad_norm_max", 0.0)) for summary in summaries
            ),
            "gd_action_saturation_rate": saturation,
            "gd_backward_time_ms_mean": mean_field("gd_backward_time_ms_mean"),
            "gd_peak_memory_mb": max(peak_values) if peak_values else None,
            "gd_cost_delta_mean": (
                cost_delta_total / candidate_count if candidate_count else 0.0
            ),
            "gd_accept_rate": accept_count / candidate_count if candidate_count else 0.0,
            "gd_accept_count": accept_count,
            "gd_candidate_count": candidate_count,
            "gd_nan_or_inf_count": sum(
                int(summary.get("gd_nan_or_inf_count", 0)) for summary in summaries
            ),
            "gd_model_cost_evals": model_cost_evals,
        }

    def solve(
        self,
        info_dict: dict[str, Any],
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("CEMGDSolver.solve requires a model")
        start_time = time.time()
        outputs: dict[str, Any] = {
            "costs": [],
            "mean": [],
            "var": [],
        }
        total_envs = len(next(iter(info_dict.values())))

        with torch.inference_mode():
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

        final_actions = torch.empty(
            [total_envs, self.horizon, self.action_dim],
            device=self.device,
            dtype=self.dtype,
        )
        final_var = torch.empty_like(var)
        gd_summaries: list[dict[str, Any]] = []
        seed_model_cost_evals = 0

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            with torch.inference_mode():
                seed, cached_cost_inputs, batch_mean, batch_var = self._run_seed_search_batch(
                    info_dict,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    batch_mean=mean[start_idx:end_idx],
                    batch_var=var[start_idx:end_idx],
                )
                seed_model_cost_evals = max(seed_model_cost_evals, seed.model_cost_evals)
                if "emb" not in cached_cost_inputs or "goal_emb" not in cached_cost_inputs:
                    raise RuntimeError(
                        "CEM-GD requires cached emb and goal_emb from the seed search"
                    )
                seed_actions_inference = seed.actions
                seed_costs_inference = seed.costs
                gd_emb_inference = cached_cost_inputs["emb"]
                gd_goal_emb_inference = cached_cost_inputs["goal_emb"]

            seed_actions = seed_actions_inference.detach().clone()
            seed_costs = seed_costs_inference.detach().clone()
            gd_emb = gd_emb_inference.detach().clone()
            gd_goal_emb = gd_goal_emb_inference.detach().clone()

            refined_actions, refined_costs, gd_summary = self._refine_seed_actions(
                seed_actions,
                seed_costs,
                gd_emb,
                gd_goal_emb,
            )
            gd_summaries.append(gd_summary)

            best_costs, best_idx = torch.min(refined_costs, dim=1)
            batch_indices = torch.arange(end_idx - start_idx, device=self.device)
            best_actions = refined_actions[batch_indices, best_idx]
            final_actions[start_idx:end_idx] = best_actions
            final_var[start_idx:end_idx] = batch_var.detach()
            outputs["costs"].extend(best_costs.detach().cpu().tolist())

        outputs["actions"] = final_actions.detach().cpu()
        outputs["mean"] = [final_actions.detach().cpu()]
        outputs["var"] = [final_var.detach().cpu()]
        gd_summary = self._combine_gd_summaries(gd_summaries)
        gd_summary["gd_seed_model_cost_evals"] = seed_model_cost_evals
        outputs["gd_summary"] = gd_summary

        print(f"CEM-GD solve time: {time.time() - start_time:.4f} seconds")
        return outputs


class _QuadraticEmbeddingCost(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target)

    def get_cost_from_embeddings(
        self,
        emb: torch.Tensor,
        goal_emb: torch.Tensor,
        action_candidates: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        del emb, goal_emb, history_size
        return ((action_candidates - self.target) ** 2).sum(dim=(2, 3))


def run_dummy_quadratic_refinement(
    *,
    lr: float = 0.01,
    steps: int = 1,
    compile_cost: bool = False,
    enable_backtracking: bool = True,
) -> DummyRefinementResult:
    seed_actions = torch.zeros(2, 1, 5, 4, dtype=torch.float32)
    target = torch.ones_like(seed_actions)
    model = _QuadraticEmbeddingCost(target)
    model._eager_get_cost_from_embeddings = model.get_cost_from_embeddings
    if compile_cost:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile unavailable")
        model.get_cost_from_embeddings = torch.compile(
            model.get_cost_from_embeddings,
            mode="reduce-overhead",
            fullgraph=False,
        )
    emb = torch.zeros(2, 1, 3, dtype=torch.float32)
    goal_emb = torch.zeros_like(emb)
    before_cost = model._eager_get_cost_from_embeddings(emb, goal_emb, seed_actions)
    solver = CEMGDSolver(
        model=model,
        gd_steps=steps,
        gd_lr=lr,
        gd_compile_cost=compile_cost,
    )
    refined_actions, refined_costs, summary = solver._refine_seed_actions(
        seed_actions,
        before_cost,
        emb,
        goal_emb,
        enable_backtracking=enable_backtracking,
    )
    del refined_actions
    return DummyRefinementResult(
        before=float(before_cost.mean().item()),
        after=float(refined_costs.mean().item()),
        accepted=bool(summary["gd_accept_count"] > 0),
        summary=summary,
    )


__all__ = [
    "CEMGDSolver",
    "DummyRefinementResult",
    "SeedSearchResult",
    "run_dummy_quadratic_refinement",
]
