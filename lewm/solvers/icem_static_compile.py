"""Static tensor-only iCEM core for compile diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .cem_exact import prepare_init_action
from .icem import ICEMSolver


CostFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class StaticChainWarmupResult:
    enabled: bool
    elapsed_s: float | None
    sample_counts: list[int]
    mode: str
    fullgraph: bool
    input_shapes: dict[str, list[int]] | None


def _inject_retained_elites_static(
    candidates: torch.Tensor,
    previous_elites: torch.Tensor | None,
    elite_retention: int,
) -> torch.Tensor:
    if previous_elites is None or elite_retention <= 0:
        return candidates
    n_inject = min(elite_retention, previous_elites.shape[1], candidates.shape[1] - 1)
    if n_inject <= 0:
        return candidates
    return torch.cat(
        (
            candidates[:, :1],
            previous_elites[:, :n_inject],
            candidates[:, 1 + n_inject :],
        ),
        dim=1,
    )


def _gather_topk_candidates(
    candidates: torch.Tensor,
    topk_inds: torch.Tensor,
) -> torch.Tensor:
    gather_index = topk_inds[:, :, None, None].expand(
        -1,
        -1,
        candidates.shape[2],
        candidates.shape[3],
    )
    return torch.gather(candidates, dim=1, index=gather_index)


def run_static_icem_chain(
    cost_fn: CostFn,
    emb: torch.Tensor,
    goal_emb: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    noise_steps: Sequence[torch.Tensor],
    *,
    topk: int,
    elite_retention: int = 0,
    momentum_alpha: float = 0.0,
    return_best_elite: bool = False,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a fixed-shape iCEM update chain using tensor-only inputs."""
    batch_mean = mean
    batch_var = var
    previous_elites: torch.Tensor | None = None
    best_elite: torch.Tensor | None = None
    final_cost: torch.Tensor | None = None

    for noise in noise_steps:
        sampled = noise * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
        candidates = torch.cat((batch_mean.unsqueeze(1), sampled[:, 1:]), dim=1)
        candidates = _inject_retained_elites_static(
            candidates,
            previous_elites,
            elite_retention,
        )
        if action_low is not None and action_high is not None:
            candidates = candidates.clamp(action_low, action_high)

        costs = cost_fn(emb, goal_emb, candidates)
        topk_vals, topk_inds = torch.topk(costs, k=topk, dim=1, largest=False)
        topk_candidates = _gather_topk_candidates(candidates, topk_inds)
        previous_elites = topk_candidates
        best_elite = topk_candidates[:, 0]

        elite_mean = topk_candidates.mean(dim=1)
        elite_var = topk_candidates.std(dim=1)
        if momentum_alpha:
            batch_mean = momentum_alpha * batch_mean + (1.0 - momentum_alpha) * elite_mean
            batch_var = momentum_alpha * batch_var + (1.0 - momentum_alpha) * elite_var
        else:
            batch_mean = elite_mean
            batch_var = elite_var
        final_cost = topk_vals.mean(dim=1)

    assert final_cost is not None
    if return_best_elite:
        assert best_elite is not None
        return best_elite, batch_var, final_cost
    return batch_mean, batch_var, final_cost


class ICEMStaticCompileSolver(ICEMSolver):
    """iCEM variant with a static tensor-only optimizer core.

    This is a diagnostic solver. It preserves the accepted iCEM interface around
    the core, but moves the repeated candidate/model-cost/top-k/update chain into
    a tensor-only function that can be run eagerly or through ``torch.compile``.
    """

    defer_cpu_materialization = True

    def __init__(
        self,
        *args: Any,
        static_chain_compile: bool = False,
        static_chain_mode: str = "reduce-overhead",
        static_chain_fullgraph: bool = False,
        static_chain_warmup: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if static_chain_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(
                "static_chain_mode must be one of "
                "'default', 'reduce-overhead', or 'max-autotune', "
                f"got {static_chain_mode!r}"
            )
        self.static_chain_compile = bool(static_chain_compile)
        self.static_chain_mode = static_chain_mode
        self.static_chain_fullgraph = bool(static_chain_fullgraph)
        self.static_chain_warmup = bool(static_chain_warmup)
        self._compiled_static_core: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        self._eager_static_core: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        self._static_compile_warmed = False

    def _make_eager_static_core(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._eager_static_core is not None:
            return self._eager_static_core

        def core(
            emb: torch.Tensor,
            goal_emb: torch.Tensor,
            mean: torch.Tensor,
            var: torch.Tensor,
            *noise_steps: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return run_static_icem_chain(
                self.model.get_cost_from_embeddings,
                emb,
                goal_emb,
                mean,
                var,
                noise_steps,
                topk=self.topk,
                elite_retention=self.elite_retention,
                momentum_alpha=self.momentum_alpha,
                return_best_elite=self.return_best_elite,
                action_low=self._action_low,
                action_high=self._action_high,
            )

        self._eager_static_core = core
        return core

    def _static_core(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        core = self._make_eager_static_core()
        if not self.static_chain_compile:
            return core
        if self._compiled_static_core is None:
            if not hasattr(torch, "compile"):
                raise RuntimeError("static_chain_compile requires torch.compile")
            mode = None if self.static_chain_mode == "default" else self.static_chain_mode
            self._compiled_static_core = torch.compile(
                core,
                mode=mode,
                fullgraph=self.static_chain_fullgraph,
            )
        return self._compiled_static_core

    def _expand_info_for_embeddings(
        self,
        info_dict: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any]:
        return self._expand_info_for_samples(
            info_dict,
            start_idx,
            end_idx,
            1,
        )

    def _prepare_cost_embeddings(
        self,
        info_dict: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_infos = self._expand_info_for_embeddings(info_dict, start_idx, end_idx)
        current_bs = end_idx - start_idx
        dummy_candidates = torch.zeros(
            current_bs,
            1,
            self.horizon,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        if hasattr(self.model, "_ensure_cost_embeddings"):
            emb, goal_emb = self.model._ensure_cost_embeddings(expanded_infos, dummy_candidates)
            return emb, goal_emb

        costs = self.model.get_cost(expanded_infos, dummy_candidates)
        del costs
        emb = expanded_infos.get("emb")
        goal_emb = expanded_infos.get("goal_emb")
        if not torch.is_tensor(emb) or not torch.is_tensor(goal_emb):
            raise RuntimeError("Static iCEM requires tensor emb and goal_emb cost inputs")
        return emb, goal_emb

    def _sample_noise_steps(self, batch_size: int) -> list[torch.Tensor]:
        return [self._sample_noise(batch_size, n_samples) for n_samples in self.sample_counts()]

    def warmup_static_chain_compile(
        self,
        *,
        embed_dim: int,
        history: int = 1,
        batch_size: int | None = None,
    ) -> StaticChainWarmupResult:
        sample_counts = self.sample_counts()
        if not self.static_chain_compile or not self.static_chain_warmup:
            return StaticChainWarmupResult(
                enabled=False,
                elapsed_s=None,
                sample_counts=sample_counts,
                mode=self.static_chain_mode,
                fullgraph=self.static_chain_fullgraph,
                input_shapes=None,
            )

        warmup_batch = int(batch_size or self.batch_size)
        emb = torch.zeros(
            warmup_batch,
            history,
            embed_dim,
            device=self.device,
            dtype=self.dtype,
        )
        goal_emb = torch.zeros_like(emb)
        mean = torch.zeros(
            warmup_batch,
            self.horizon,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        var = self.var_scale * torch.ones_like(mean)
        noise_steps = [
            torch.zeros(
                warmup_batch,
                n_samples,
                self.horizon,
                self.action_dim,
                device=self.device,
                dtype=self.dtype,
            )
            for n_samples in sample_counts
        ]

        started = time.perf_counter()
        with torch.inference_mode():
            final_mean, final_var, final_cost = self._static_core()(
                emb,
                goal_emb,
                mean,
                var,
                *noise_steps,
            )
        del final_mean, final_var, final_cost
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        self._static_compile_warmed = True
        return StaticChainWarmupResult(
            enabled=True,
            elapsed_s=time.perf_counter() - started,
            sample_counts=sample_counts,
            mode=self.static_chain_mode,
            fullgraph=self.static_chain_fullgraph,
            input_shapes={
                "emb": list(emb.shape),
                "goal_emb": list(goal_emb.shape),
                "mean": list(mean.shape),
                "var": list(var.shape),
                "noise_steps": [list(noise.shape) for noise in noise_steps],
            },
        )

    @torch.inference_mode()
    def solve(
        self,
        info_dict: dict[str, Any],
        init_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
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

        core = self._static_core()
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            emb, goal_emb = self._prepare_cost_embeddings(info_dict, start_idx, end_idx)
            noise_steps = self._sample_noise_steps(end_idx - start_idx)

            final_actions, final_var, final_cost = core(
                emb,
                goal_emb,
                batch_mean,
                batch_var,
                *noise_steps,
            )
            mean[start_idx:end_idx] = final_actions
            var[start_idx:end_idx] = final_var
            outputs["costs"].extend(final_cost.detach().cpu().tolist())

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        outputs["static_chain"] = {
            "static_chain_compile": self.static_chain_compile,
            "static_chain_mode": self.static_chain_mode,
            "static_chain_fullgraph": self.static_chain_fullgraph,
            "static_chain_warmed": self._static_compile_warmed,
            "sample_counts_per_step": self.sample_counts(),
        }
        return outputs


__all__ = [
    "ICEMStaticCompileSolver",
    "StaticChainWarmupResult",
    "run_static_icem_chain",
]
