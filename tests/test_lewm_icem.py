from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import torch
    from torch import nn

    from lewm.solvers.cem_exact import CEMExactSolver
    from lewm.solvers.icem import ICEMSolver, ICEMStepSummaryCallback, make_colored_noise
    from tools.benchmarks.lewm_plan_benchmark import CEMTelemetry, install_cem_telemetry
    from tools.benchmarks.metrics import read_jsonl
except ModuleNotFoundError:
    np = None
    torch = None
    nn = None
    CEMExactSolver = None
    ICEMSolver = None
    ICEMStepSummaryCallback = None
    make_colored_noise = None
    CEMTelemetry = None
    install_cem_telemetry = None
    read_jsonl = None


if torch is not None:

    class DummyBoxSpace:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape
            self.low = -np.ones(shape, dtype=np.float32)
            self.high = np.ones(shape, dtype=np.float32)


    class QuadraticCostModel(nn.Module):
        def __init__(self, target: torch.Tensor):
            super().__init__()
            self.register_buffer("target", target)

        def get_cost(self, info_dict: dict, candidates: torch.Tensor) -> torch.Tensor:
            del info_dict
            return ((candidates - self.target) ** 2).sum(dim=(2, 3))


    class PlanConfig:
        horizon = 5
        action_block = 1


@unittest.skipIf(torch is None, "torch is required for iCEM solver tests")
class ICEMSolverTests(unittest.TestCase):
    def test_feature_disabled_path_matches_cem_exact(self) -> None:
        target = torch.tensor(
            [
                [[0.25, -0.5], [0.75, 0.5]],
            ],
            dtype=torch.float32,
        )
        info = {"state": torch.zeros(3, 1, dtype=torch.float32)}
        config = SimpleNamespace(horizon=2, action_block=1)
        action_space = DummyBoxSpace(shape=(3, 2))

        common_kwargs = {
            "batch_size": 2,
            "num_samples": 16,
            "var_scale": 1.0,
            "n_steps": 4,
            "topk": 4,
            "device": "cpu",
            "seed": 17,
        }
        cem = CEMExactSolver(model=QuadraticCostModel(target), **common_kwargs)
        icem = ICEMSolver(
            model=QuadraticCostModel(target),
            elite_retention=0,
            colored_noise="none",
            momentum_alpha=0.0,
            sample_decay=1.0,
            return_best_elite=False,
            clip_actions=False,
            **common_kwargs,
        )
        cem.configure(action_space=action_space, n_envs=3, config=config)
        icem.configure(action_space=action_space, n_envs=3, config=config)

        cem_outputs = cem.solve(info)
        icem_outputs = icem.solve(info)

        torch.testing.assert_close(icem_outputs["actions"], cem_outputs["actions"])
        torch.testing.assert_close(icem_outputs["mean"][0], cem_outputs["mean"][0])
        torch.testing.assert_close(icem_outputs["var"][0], cem_outputs["var"][0])
        self.assertEqual(icem_outputs["costs"], cem_outputs["costs"])

    def test_colored_noise_shape_and_device(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        noise = make_colored_noise(
            batch_size=2,
            num_samples=8,
            horizon=5,
            action_dim=10,
            beta=1.0,
            device=device,
            dtype=torch.float32,
        )

        self.assertEqual(noise.shape, (2, 8, 5, 10))
        self.assertEqual(noise.device, device)
        self.assertTrue(torch.isfinite(noise).all().item())

    def test_raw_time_colored_noise_flattens_action_blocks(self) -> None:
        noise = make_colored_noise(
            batch_size=2,
            num_samples=8,
            horizon=5,
            action_dim=10,
            beta=1.0,
            device="cpu",
            dtype=torch.float32,
            action_block=5,
            time_axis="raw",
        )

        self.assertEqual(noise.shape, (2, 8, 5, 10))
        self.assertTrue(torch.isfinite(noise).all().item())

    def test_sample_decay_counts_keep_at_least_two_topk(self) -> None:
        solver = ICEMSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 2, dtype=torch.float32)),
            num_samples=8,
            n_steps=4,
            topk=2,
            sample_decay=2.0,
        )

        self.assertEqual(solver.sample_counts(), [8, 4, 4, 4])
        self.assertEqual(solver.model_cost_evals_per_solve(), 20)

    def test_sample_decay_bucket_counts_round_up_to_stable_shapes(self) -> None:
        solver = ICEMSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 2, dtype=torch.float32)),
            num_samples=12,
            n_steps=4,
            topk=2,
            sample_decay=2.0,
            sample_count_bucket_size=4,
        )

        self.assertEqual(solver.sample_counts(), [12, 8, 4, 4])
        self.assertEqual(solver.model_cost_evals_per_solve(), 28)

    def test_elite_retention_preserves_mean_candidate_slot(self) -> None:
        solver = ICEMSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 10, dtype=torch.float32)),
            num_samples=8,
            n_steps=2,
            topk=2,
            elite_retention=2,
            colored_noise="none",
        )
        previous_elites = torch.arange(2 * 2 * 5 * 10, dtype=torch.float32).reshape(
            2,
            2,
            5,
            10,
        )
        fresh_candidates = torch.zeros(2, 8, 5, 10)

        injected = solver.inject_retained_elites(fresh_candidates.clone(), previous_elites)

        torch.testing.assert_close(injected[:, 1:3], previous_elites)
        torch.testing.assert_close(injected[:, 0], fresh_candidates[:, 0])
        torch.testing.assert_close(injected[:, 3:], fresh_candidates[:, 3:])

    def test_solve_returns_actions_mean_and_var(self) -> None:
        solver = ICEMSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 2, dtype=torch.float32)),
            batch_size=2,
            num_samples=8,
            var_scale=1.0,
            n_steps=2,
            topk=2,
            device="cpu",
            seed=11,
            elite_retention=1,
        )
        solver.configure(
            action_space=DummyBoxSpace(shape=(3, 2)),
            n_envs=3,
            config=PlanConfig(),
        )

        outputs = solver.solve({"state": torch.zeros(3, 1, dtype=torch.float32)})

        self.assertEqual(outputs["actions"].shape, (3, 5, 2))
        self.assertEqual(outputs["mean"][0].shape, (3, 5, 2))
        self.assertEqual(outputs["var"][0].shape, (3, 5, 2))
        self.assertEqual(len(outputs["costs"]), 3)

    def test_telemetry_records_actual_decayed_sample_counts(self) -> None:
        solver = ICEMSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 2, dtype=torch.float32)),
            batch_size=2,
            num_samples=8,
            var_scale=1.0,
            n_steps=3,
            topk=2,
            device="cpu",
            seed=17,
            sample_decay=2.0,
        )
        solver.configure(
            action_space=DummyBoxSpace(shape=(3, 2)),
            n_envs=3,
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
            telemetry.current_plan_index = 5

            self.assertTrue(install_cem_telemetry(solver, telemetry))
            solver.solve({"state": torch.zeros(3, 1, dtype=torch.float32)})
            rows = read_jsonl(metrics_path)

        cem_rows = [row for row in rows if row.get("type") == "cem_call"]
        self.assertEqual(len(cem_rows), 1)
        row = cem_rows[0]
        self.assertEqual(row["solver_class"], "ICEMSolver")
        self.assertEqual(row["sample_counts_per_step"], [8, 4, 4])
        self.assertEqual(row["model_cost_evals"], 16)
        self.assertEqual(row["model_cost_env_candidate_evals"], 48)
        self.assertFalse(row["cpu_sync_in_loop"])
        self.assertEqual(row["loop_cpu_materialization_count_estimate"], 0)
        self.assertEqual(row["output_actions_shape"], [3, 5, 2])

    def test_icem_step_summary_records_per_step_metrics(self) -> None:
        solver = ICEMSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 2, dtype=torch.float32)),
            batch_size=1,
            num_samples=8,
            var_scale=1.0,
            n_steps=3,
            topk=2,
            device="cpu",
            seed=23,
            elite_retention=1,
            colored_noise="none",
            sample_decay=2.0,
            sample_count_bucket_size=4,
            callbacks=[ICEMStepSummaryCallback()],
        )
        solver.configure(
            action_space=DummyBoxSpace(shape=(1, 2)),
            n_envs=1,
            config=PlanConfig(),
        )

        outputs = solver.solve({"state": torch.zeros(1, 1, dtype=torch.float32)})
        rows = outputs["callbacks"]["icem_step_summary"]

        self.assertEqual([row["sample_count"] for row in rows], [8, 4, 4])
        self.assertEqual([row["step"] for row in rows], [0, 1, 2])
        for key in (
            "best_cost_mean",
            "mean_candidate_cost_mean",
            "topk_mean_cost_mean",
            "topk_std_cost_mean",
            "mean_delta_norm_mean",
            "var_mean",
            "var_min",
            "var_max",
        ):
            self.assertIn(key, rows[0])
            self.assertIsInstance(rows[0][key], float)
        self.assertIn("action_saturation_rate", rows[0])
        self.assertIsNone(rows[0]["action_saturation_rate"])


if __name__ == "__main__":
    unittest.main()
