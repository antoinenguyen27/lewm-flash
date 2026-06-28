from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from lewm.solvers.cem_gd import CEMGDSolver, run_dummy_quadratic_refinement
    from tools.benchmarks.lewm_plan_benchmark import CEMTelemetry, install_cem_telemetry
    from tools.benchmarks.metrics import read_jsonl
except ModuleNotFoundError:
    np = None
    torch = None
    nn = None
    CEMGDSolver = None
    run_dummy_quadratic_refinement = None
    CEMTelemetry = None
    install_cem_telemetry = None
    read_jsonl = None


if torch is not None:

    class DummyBoxSpace:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape
            self.low = -np.ones(shape, dtype=np.float32)
            self.high = np.ones(shape, dtype=np.float32)


    class QuadraticCEMGDModel(nn.Module):
        def __init__(self, target: torch.Tensor):
            super().__init__()
            self.register_buffer("target", target)

        def get_cost(self, info_dict: dict, candidates: torch.Tensor) -> torch.Tensor:
            batch_size = candidates.shape[0]
            info_dict["emb"] = torch.zeros(
                batch_size,
                1,
                3,
                device=candidates.device,
                dtype=candidates.dtype,
            )
            info_dict["goal_emb"] = torch.ones(
                batch_size,
                1,
                1,
                device=candidates.device,
                dtype=candidates.dtype,
            )
            return self.get_cost_from_embeddings(
                info_dict["emb"],
                info_dict["goal_emb"],
                candidates,
            )

        def get_cost_from_embeddings(
            self,
            emb: torch.Tensor,
            goal_emb: torch.Tensor,
            action_candidates: torch.Tensor,
            history_size: int = 3,
        ) -> torch.Tensor:
            del emb, history_size
            target = goal_emb.reshape(goal_emb.shape[0], 1, 1, 1).expand_as(
                action_candidates
            )
            return F.mse_loss(action_candidates, target, reduction="none").sum(dim=(2, 3))


    class PlanConfig:
        horizon = 5
        action_block = 1


@unittest.skipIf(torch is None, "torch is required for CEM-GD solver tests")
class CEMGDSolverTests(unittest.TestCase):
    def test_cem_gd_only_action_params_require_grad(self) -> None:
        model = nn.Linear(4, 4)
        seed_actions = torch.zeros(2, 1, 5, 10)
        solver = CEMGDSolver(gd_steps=1, gd_lr=0.01)

        solver.freeze_model_for_action_optimization(model)
        params = solver.new_action_params(seed_actions)

        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(params.requires_grad)
        torch.testing.assert_close(params.detach(), seed_actions)

    def test_cem_gd_refinement_decreases_quadratic_dummy_cost(self) -> None:
        result = run_dummy_quadratic_refinement(lr=0.01, steps=1)

        self.assertLessEqual(result.after, result.before)
        self.assertTrue(result.accepted)

    def test_cem_gd_reverts_worsening_refinement(self) -> None:
        result = run_dummy_quadratic_refinement(
            lr=100.0,
            steps=1,
            enable_backtracking=False,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.after, result.before)

    def test_cem_gd_uses_best_elite_seed_source(self) -> None:
        costs = torch.tensor([[3.0, 1.0, 2.0]])
        candidates = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(1, 3, 5, 2)

        seed = CEMGDSolver.select_seed_elites(candidates, costs, refine_topk=1)

        torch.testing.assert_close(seed.costs, torch.tensor([[1.0]]))
        torch.testing.assert_close(seed.actions, candidates[:, 1:2])

    def test_gd_compile_cost_autograd_matches_eager_when_available(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")

        eager = run_dummy_quadratic_refinement(lr=0.01, steps=1, compile_cost=False)
        compiled = run_dummy_quadratic_refinement(lr=0.01, steps=1, compile_cost=True)

        self.assertLessEqual(abs(compiled.after - eager.after), 1e-5)

    def test_cem_gd_telemetry_records_gd_summary(self) -> None:
        solver = CEMGDSolver(
            model=QuadraticCEMGDModel(torch.ones(1, 5, 2, dtype=torch.float32)),
            batch_size=1,
            num_samples=8,
            var_scale=1.0,
            n_steps=2,
            topk=2,
            device="cpu",
            seed=13,
            gd_seed_solver="icem",
            gd_refine_topk=1,
            gd_steps=1,
            gd_lr=0.01,
        )
        solver.configure(
            action_space=DummyBoxSpace(shape=(1, 2)),
            n_envs=1,
            config=PlanConfig(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.jsonl"
            telemetry = CEMTelemetry(
                metrics_path=metrics_path,
                seed=0,
                mode="summary",
                warmup_plans=0,
                max_recorded_calls=None,
            )
            telemetry.current_plan_index = 7

            self.assertTrue(install_cem_telemetry(solver, telemetry))
            outputs = solver.solve({"state": torch.zeros(1, 1, dtype=torch.float32)})
            rows = read_jsonl(metrics_path)

        cem_rows = [row for row in rows if row.get("type") == "cem_call"]
        self.assertEqual(len(cem_rows), 1)
        row = cem_rows[0]
        self.assertEqual(row["solver_class"], "CEMGDSolver")
        self.assertEqual(row["gd_seed_solver"], "icem")
        self.assertEqual(row["gd_seed_source"], "best_elite")
        self.assertEqual(row["gd_refine_topk"], 1)
        self.assertEqual(row["gd_steps"], 1)
        self.assertEqual(row["gd_model_cost_evals"], 5)
        self.assertEqual(row["model_cost_evals"], 16)
        self.assertEqual(row["model_cost_evals_including_gd"], 21)
        self.assertLess(row["gd_cost_delta_mean"], 0.0)
        self.assertGreater(row["gd_accept_rate"], 0.0)
        self.assertEqual(row["gd_nan_or_inf_count"], 0)
        self.assertEqual(outputs["actions"].shape, (1, 5, 2))


if __name__ == "__main__":
    unittest.main()
