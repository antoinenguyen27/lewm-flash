from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import torch
    from torch import nn

    from lewm.jepa import JEPA
    from lewm.solvers.icem_static_compile import (
        ICEMStaticCompileSolver,
        run_static_icem_chain,
    )
except ModuleNotFoundError:
    np = None
    torch = None
    nn = None
    ICEMStaticCompileSolver = None
    run_static_icem_chain = None
    JEPA = None


if torch is not None:

    class DummyBoxSpace:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape
            self.low = -np.ones(shape, dtype=np.float32)
            self.high = np.ones(shape, dtype=np.float32)


    class TensorOnlyQuadraticModel(nn.Module):
        def __init__(self, target: torch.Tensor):
            super().__init__()
            self.register_buffer("target", target)

        def get_cost_from_embeddings(
            self,
            emb: torch.Tensor,
            goal_emb: torch.Tensor,
            candidates: torch.Tensor,
        ) -> torch.Tensor:
            del emb, goal_emb
            return ((candidates - self.target) ** 2).sum(dim=(2, 3))


    class DummyPredictor(nn.Module):
        def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
            del act_emb
            return emb


    class DummyActionEncoder(nn.Module):
        def forward(self, action: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (*action.shape[:-1], 1),
                device=action.device,
                dtype=action.dtype,
            )


@unittest.skipIf(torch is None, "torch is required for static iCEM tests")
class ICEMStaticCompileTests(unittest.TestCase):
    def test_jepa_model_cost_autocast_returns_float_cost(self) -> None:
        model = JEPA(
            encoder=nn.Identity(),
            predictor=DummyPredictor(),
            action_encoder=DummyActionEncoder(),
        )
        emb = torch.zeros(1, 1, 2)
        goal_emb = torch.ones(1, 1, 2)
        action = torch.zeros(1, 2, 1, 1)

        self.assertEqual(model.model_cost_autocast_mode(), "none")
        model.set_model_cost_autocast("bf16")
        self.assertTrue(model.model_cost_autocast_enabled())
        self.assertEqual(model.model_cost_autocast_dtype(), torch.bfloat16)

        cost = model.get_cost_from_embeddings(emb, goal_emb, action)

        self.assertEqual(cost.dtype, torch.float32)
        self.assertEqual(tuple(cost.shape), (1, 2))

    def test_static_chain_decreases_dummy_quadratic_cost(self) -> None:
        target = torch.ones(1, 1, 2, 1)
        model = TensorOnlyQuadraticModel(target)
        emb = torch.zeros(1, 1, 4)
        goal_emb = torch.zeros_like(emb)
        mean = torch.zeros(1, 2, 1)
        var = torch.ones_like(mean)
        noise = torch.tensor(
            [[[[0.0], [0.0]], [[1.0], [1.0]], [[2.0], [2.0]], [[-1.0], [-1.0]]]]
        )

        before = model.get_cost_from_embeddings(emb, goal_emb, mean.unsqueeze(1))[:, 0]
        final_mean, final_var, final_cost = run_static_icem_chain(
            model.get_cost_from_embeddings,
            emb,
            goal_emb,
            mean,
            var,
            [noise],
            topk=2,
        )

        self.assertLessEqual(float(final_cost.item()), float(before.item()))
        self.assertEqual(final_mean.shape, mean.shape)
        self.assertEqual(final_var.shape, var.shape)

    def test_compiled_static_chain_matches_eager_when_available(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")
        target = torch.ones(1, 1, 2, 1)
        model = TensorOnlyQuadraticModel(target)
        emb = torch.zeros(1, 1, 4)
        goal_emb = torch.zeros_like(emb)
        mean = torch.zeros(1, 2, 1)
        var = torch.ones_like(mean)
        noise = torch.tensor(
            [[[[0.0], [0.0]], [[1.0], [1.0]], [[2.0], [2.0]], [[-1.0], [-1.0]]]]
        )

        def core(
            emb_in: torch.Tensor,
            goal_in: torch.Tensor,
            mean_in: torch.Tensor,
            var_in: torch.Tensor,
            noise_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return run_static_icem_chain(
                model.get_cost_from_embeddings,
                emb_in,
                goal_in,
                mean_in,
                var_in,
                [noise_in],
                topk=2,
            )

        eager = core(emb, goal_emb, mean, var, noise)
        compiled_core = torch.compile(core, mode="reduce-overhead", fullgraph=False)
        compiled = compiled_core(emb, goal_emb, mean, var, noise)

        for eager_tensor, compiled_tensor in zip(eager, compiled):
            torch.testing.assert_close(compiled_tensor, eager_tensor)

    def test_solver_sample_counts_match_bucketed_icem(self) -> None:
        solver = ICEMStaticCompileSolver(
            model=TensorOnlyQuadraticModel(torch.zeros(1, 1, 5, 2)),
            num_samples=300,
            n_steps=30,
            topk=30,
            sample_decay=1.25,
            sample_count_bucket_size=60,
        )

        self.assertFalse(solver.static_chain_compile)
        self.assertEqual(
            solver.sample_counts(),
            [300, 240, 240, 180, 180, 120, 120, 120] + [60] * 22,
        )
        self.assertEqual(solver.model_cost_evals_per_solve(), 2820)

    def test_static_warmup_reports_shapes_when_not_compiling(self) -> None:
        solver = ICEMStaticCompileSolver(
            model=TensorOnlyQuadraticModel(torch.zeros(1, 1, 5, 2)),
            batch_size=1,
            num_samples=8,
            n_steps=2,
            topk=2,
            device="cpu",
            static_chain_compile=False,
        )
        solver.configure(
            action_space=DummyBoxSpace(shape=(1, 2)),
            n_envs=1,
            config=SimpleNamespace(horizon=5, action_block=1),
        )

        result = solver.warmup_static_chain_compile(embed_dim=4)

        self.assertFalse(result.enabled)
        self.assertEqual(result.sample_counts, [8, 8])


if __name__ == "__main__":
    unittest.main()
