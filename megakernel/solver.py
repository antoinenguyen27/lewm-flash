"""iCEM solver whose whole per-env solve runs as one LeWM-MK megakernel launch.

Keeps every planner-level LeWM-Flash change (cached observation/goal embeddings, GPU-resident iCEM loop, the bucketed
300 -> 60 sample schedule, top-30 elites, 5 retained elites, var_scale init, warm-start mean) and replaces the
kernel/precision stack (torch.compile, TF32, BF16 autocast) with the megakernel. Noise comes from the kernel's Philox
stream instead of torch's generator, so trajectories differ draw-for-draw but follow the same distribution.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from lewm.solvers.cem_exact import prepare_init_action
from lewm.solvers.icem import ICEMSolver


class MegakernelICEMSolver(ICEMSolver):
    defer_cpu_materialization = True

    def __init__(self, *args: Any, build_dir: str | None = None, cluster_size: int | None = None,
                 clusters: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.colored_noise != "none" or self.momentum_alpha or self.return_best_elite or self.clip_actions:
            raise ValueError("MegakernelICEMSolver implements the accepted LeWM-Flash iCEM configuration only")
        from megakernel.runtime import Megakernel, load_ext

        ext = load_ext(build_dir or os.environ.get("LEWM_MK_BUILD", "/mk/build"))
        # 8-CTA clusters (15 co-resident) for batched plans; 16-CTA clusters (7 co-resident, half the per-CTA weight
        # stream) are faster when a single env replans.
        self.mk = Megakernel(self.model, ext, cl=cluster_size or 8, clusters=clusters,
                             max_envs=int(os.environ.get("LEWM_MK_MAX_ENVS", "100")))
        self.mk_small = (Megakernel(self.model, ext, cl=16, max_envs=1)
                         if cluster_size is None and ext.query_max_clusters(16) >= 4 else None)
        self.mk.seed = int(kwargs.get("seed", 1234)) * 1_000_003
        if self.mk_small is not None:
            self.mk_small.seed = self.mk.seed + 500_000
        self.counts = self.sample_counts()

    @torch.inference_mode()
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        start = time.time()
        total_envs = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(self.model, info_dict, init_action, self.horizon, n_envs=total_envs,
                                          action_dim=self.action_dim)
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean, var = mean.to(self.device), var.to(self.device)

        # One encoder pass per plan for all envs (observation + goal), exactly as the cached LeWM-Flash path.
        one = self._expand_info_for_samples(info_dict, 0, total_envs, 1)
        dummy = torch.zeros(total_envs, 1, self.horizon, self.action_dim, device=self.device)
        self.model.set_embedding_cache(True)
        emb, goal_emb = self.model._ensure_cost_embeddings(one, dummy)
        emb = emb.reshape(total_envs, -1, emb.shape[-1])[:, -1].float()
        goal_emb = goal_emb.reshape(total_envs, -1, goal_emb.shape[-1])[:, -1].float()

        # Every replanning env's full 30-step iCEM solve in one persistent-kernel launch (envs are independent
        # problems with their own schedule, elites, distribution and Philox stream).
        costs = []
        mk = self.mk_small if (self.mk_small is not None and total_envs <= self.mk_small.max_envs) else self.mk
        for b0 in range(0, total_envs, mk.max_envs):
            b1 = min(total_envs, b0 + mk.max_envs)
            m, s, c = mk.solve_batch(emb[b0:b1], goal_emb[b0:b1], init_mean=mean[b0:b1], counts=self.counts,
                                          init_std=float(self.var_scale), topk=self.topk,
                                          elite_ret=self.elite_retention)
            mean[b0:b1] = m
            var[b0:b1] = s
            costs.append(c.clone())
        outputs = {"costs": torch.cat(costs).cpu().tolist(), "actions": mean.detach().cpu()}
        outputs["mean"] = [outputs["actions"]]
        outputs["var"] = [var.detach().cpu()]
        print(f"iCEM-MK solve time: {time.time() - start:.4f} seconds")
        return outputs


__all__ = ["MegakernelICEMSolver"]
