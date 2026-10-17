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
                 clusters: int | None = None, fused_encoder: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.colored_noise != "none" or self.momentum_alpha or self.return_best_elite or self.clip_actions:
            raise ValueError("MegakernelICEMSolver implements the accepted LeWM-Flash iCEM configuration only")
        from megakernel.runtime import Megakernel, load_ext

        ext = load_ext(build_dir or os.environ.get("LEWM_MK_BUILD", "/mk/build"))
        # 8-CTA clusters (15 co-resident) for batched plans; 16-CTA clusters (7 co-resident, half the per-CTA weight
        # stream) are faster when a single env replans.
        self.fused_encoder = bool(fused_encoder)
        self._goal_cache: dict[bytes, torch.Tensor] = {}
        self.mk = Megakernel(self.model, ext, cl=cluster_size or 8, clusters=clusters,
                             max_envs=int(os.environ.get("LEWM_MK_MAX_ENVS", "100")), encoder=self.fused_encoder)
        self.mk_small = (Megakernel(self.model, ext, cl=16, max_envs=1, encoder=self.fused_encoder)
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

        timing = os.environ.get("LEWM_MK_TIMING") == "1"
        if timing:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
        if self.fused_encoder:
            return self._solve_fused(info_dict, mean, var, total_envs, start, timing, ev if timing else None)
        # One encoder pass per plan for all envs (observation + goal), exactly as the cached LeWM-Flash path.
        one = self._expand_info_for_samples(info_dict, 0, total_envs, 1)
        dummy = torch.zeros(total_envs, 1, self.horizon, self.action_dim, device=self.device)
        self.model.set_embedding_cache(True)
        emb, goal_emb = self.model._ensure_cost_embeddings(one, dummy)
        emb = emb.reshape(total_envs, -1, emb.shape[-1])[:, -1].float()
        goal_emb = goal_emb.reshape(total_envs, -1, goal_emb.shape[-1])[:, -1].float()
        if timing:
            ev[1].record()

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
        if timing:
            ev[2].record()
        outputs = {"costs": torch.cat(costs).cpu().tolist(), "actions": mean.detach().cpu()}
        if timing:
            ev[3].record()
            torch.cuda.synchronize()
            print(f"MK_TIMING envs={total_envs} encode_ms={ev[0].elapsed_time(ev[1]):.3f} "
                  f"solve_ms={ev[1].elapsed_time(ev[2]):.3f} to_host_ms={ev[2].elapsed_time(ev[3]):.3f} "
                  f"wall_ms={(time.time() - start) * 1000:.3f}", flush=True)
        outputs["mean"] = [outputs["actions"]]
        outputs["var"] = [var.detach().cpu()]
        print(f"iCEM-MK solve time: {time.time() - start:.4f} seconds")
        return outputs


    def _solve_fused(self, info_dict, mean, var, total_envs, start, timing, ev):
        """Raw uint8 frames -> one launch per plan (fused ViT encoder + every env's iCEM). The goal embedding is
        cached per goal image (exact: same pixels, same embedding), so a new episode encodes its goal once."""
        import hashlib

        import numpy as np

        obs = torch.as_tensor(info_dict["pixels"])[:, -1]  # [E, H, W, C] uint8 (history 1)
        goal = torch.as_tensor(info_dict["goal"])[:, -1]
        goal_np = goal.cpu().numpy()
        keys = [hashlib.blake2b(np.ascontiguousarray(goal_np[b]).tobytes(), digest_size=16).digest()
                for b in range(total_envs)]
        mk = self.mk_small if (self.mk_small is not None and total_envs <= self.mk_small.max_envs) else self.mk
        if total_envs > mk.max_envs:
            raise ValueError(f"{total_envs} replanning envs > LEWM_MK_MAX_ENVS={mk.max_envs}")
        new = [b for b in range(total_envs) if keys[b] not in self._goal_cache]
        for b in range(total_envs):
            if keys[b] in self._goal_cache:
                mk.goal[b].copy_(self._goal_cache[keys[b]])
        if timing:
            ev[1].record()
        m, s, c = mk.solve_images(obs, goal[new] if new else None, new, init_mean=mean, counts=self.counts,
                                  init_std=float(self.var_scale), topk=self.topk, elite_ret=self.elite_retention)
        for b in new:
            if len(self._goal_cache) > 4096:
                self._goal_cache.clear()
            self._goal_cache[keys[b]] = mk.goal[b].clone()
        mean.copy_(m)
        var.copy_(s)
        if timing:
            ev[2].record()
        outputs = {"costs": c.clone().cpu().tolist(), "actions": mean.detach().cpu()}
        outputs["mean"] = [outputs["actions"]]
        outputs["var"] = [var.detach().cpu()]
        if timing:
            ev[3].record()
            torch.cuda.synchronize()
            print(f"MK_TIMING envs={total_envs} fused new_goals={len(new)} host_prep_ms={ev[0].elapsed_time(ev[1]):.3f} "
                  f"encode+solve_ms={ev[1].elapsed_time(ev[2]):.3f} to_host_ms={ev[2].elapsed_time(ev[3]):.3f} "
                  f"wall_ms={(time.time() - start) * 1000:.3f}", flush=True)
        print(f"iCEM-MK solve time: {time.time() - start:.4f} seconds")
        return outputs


__all__ = ["MegakernelICEMSolver"]
