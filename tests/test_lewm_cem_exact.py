from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import torch
    from torch import nn

    from lewm.solvers.cem_exact import CEMExactSolver, prepare_init_action
    from tools.benchmarks.lewm_plan_benchmark import CEMTelemetry, install_cem_telemetry
    from tools.benchmarks.metrics import read_jsonl
except ModuleNotFoundError:
    np = None
    torch = None
    nn = None
    CEMExactSolver = None
    prepare_init_action = None
    CEMTelemetry = None
    install_cem_telemetry = None
    read_jsonl = None


if torch is not None:

    class DummyBoxSpace:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape


    class QuadraticCostModel(nn.Module):
        def __init__(self, target: torch.Tensor):
            super().__init__()
            self.register_buffer("target", target)

        def get_cost(self, info_dict: dict, candidates: torch.Tensor) -> torch.Tensor:
            del info_dict
            return ((candidates - self.target) ** 2).sum(dim=(2, 3))


    class LoopSyncReferenceCEMSolver(CEMExactSolver):
        @torch.inference_mode()
        def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
            outputs = {"costs": [], "mean": [], "var": []}
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
                        value_batch = np.repeat(
                            value_batch[:, None, ...],
                            self.num_samples,
                            axis=1,
                        )
                    expanded_infos[key] = value_batch

                final_batch_cost = None
                for _ in range(self.n_steps):
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
                    topk_vals, topk_inds = torch.topk(
                        costs,
                        k=self.topk,
                        dim=1,
                        largest=False,
                    )
                    batch_indices = (
                        torch.arange(current_bs, device=self.device)
                        .unsqueeze(1)
                        .expand(-1, self.topk)
                    )
                    topk_candidates = candidates[batch_indices, topk_inds]
                    batch_mean = topk_candidates.mean(dim=1)
                    batch_var = topk_candidates.std(dim=1)
                    final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

                mean[start_idx:end_idx] = batch_mean
                var[start_idx:end_idx] = batch_var
                outputs["costs"].extend(final_batch_cost)

            outputs["actions"] = mean.detach().cpu()
            outputs["mean"] = [mean.detach().cpu()]
            outputs["var"] = [var.detach().cpu()]
            return outputs


@unittest.skipIf(torch is None, "torch is required for CEM exact solver tests")
class CEMExactSolverTests(unittest.TestCase):
    def test_matches_loop_sync_reference_on_quadratic_cost(self) -> None:
        target = torch.tensor(
            [
                [[0.25, -0.5], [0.75, 0.5]],
            ],
            dtype=torch.float32,
        )
        info = {"state": torch.zeros(3, 1, dtype=torch.float32)}
        config = SimpleNamespace(horizon=2, action_block=1)
        action_space = DummyBoxSpace(shape=(3, 2))

        def build_solver(cls):
            solver = cls(
                model=QuadraticCostModel(target),
                batch_size=2,
                num_samples=16,
                var_scale=1.0,
                n_steps=4,
                topk=4,
                device="cpu",
                seed=17,
            )
            solver.configure(action_space=action_space, n_envs=3, config=config)
            return solver

        reference = build_solver(LoopSyncReferenceCEMSolver)
        candidate = build_solver(CEMExactSolver)

        reference_outputs = reference.solve(info)
        candidate_outputs = candidate.solve(info)

        torch.testing.assert_close(
            candidate_outputs["actions"],
            reference_outputs["actions"],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            candidate_outputs["mean"][0],
            reference_outputs["mean"][0],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            candidate_outputs["var"][0],
            reference_outputs["var"][0],
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(candidate_outputs["costs"], reference_outputs["costs"])

    def test_warm_start_shifts_previous_distribution_tail(self) -> None:
        config = SimpleNamespace(horizon=5, receding_horizon=2, action_block=1)
        action_space = DummyBoxSpace(shape=(2, 2))
        solver = CEMExactSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 2, dtype=torch.float32)),
            batch_size=2,
            num_samples=8,
            var_scale=1.0,
            n_steps=2,
            topk=2,
            device="cpu",
            seed=31,
            warm_start=True,
        )
        solver.configure(action_space=action_space, n_envs=2, config=config)
        previous_mean = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
        previous_var = previous_mean + 100.0
        solver._remember_warm_start_distribution(previous_mean, previous_var)

        mean, var = solver.init_action_distrib(2)
        next_mean, next_var = solver._shift_warm_start_distribution(mean, var)

        torch.testing.assert_close(next_mean[:, :3], previous_mean[:, 2:])
        torch.testing.assert_close(next_var[:, :3], previous_var[:, 2:])
        torch.testing.assert_close(next_mean[:, 3:], torch.zeros(2, 2, 2))
        torch.testing.assert_close(next_var[:, 3:], torch.ones(2, 2, 2))
        self.assertTrue(solver.last_warm_start_info["used"])

    def test_warm_start_resets_on_shape_change(self) -> None:
        config = SimpleNamespace(horizon=5, receding_horizon=2, action_block=1)
        action_space = DummyBoxSpace(shape=(2, 2))
        solver = CEMExactSolver(
            model=QuadraticCostModel(torch.zeros(1, 5, 2, dtype=torch.float32)),
            batch_size=2,
            num_samples=8,
            var_scale=1.0,
            n_steps=2,
            topk=2,
            device="cpu",
            seed=37,
            warm_start=True,
        )
        solver.configure(action_space=action_space, n_envs=2, config=config)
        solver._remember_warm_start_distribution(
            torch.ones(2, 5, 2, dtype=torch.float32),
            torch.ones(2, 5, 2, dtype=torch.float32),
        )

        mean, var = solver.init_action_distrib(3)
        next_mean, next_var = solver._shift_warm_start_distribution(mean, var)

        torch.testing.assert_close(next_mean, torch.zeros(3, 5, 2))
        torch.testing.assert_close(next_var, torch.ones(3, 5, 2))
        self.assertFalse(solver.last_warm_start_info["used"])
        self.assertTrue(solver.last_warm_start_info["reset"])
        self.assertEqual(solver.last_warm_start_info["reset_reason"], "signature_mismatch")
        self.assertIsNone(solver._warm_start_mean)
        self.assertIsNone(solver._warm_start_var)

    def test_warm_start_state_is_used_on_second_solve(self) -> None:
        config = SimpleNamespace(horizon=2, receding_horizon=1, action_block=1)
        action_space = DummyBoxSpace(shape=(2, 2))
        info = {"state": torch.zeros(1, 1, dtype=torch.float32)}
        solver = CEMExactSolver(
            model=QuadraticCostModel(torch.zeros(1, 2, 2, dtype=torch.float32)),
            batch_size=1,
            num_samples=8,
            var_scale=1.0,
            n_steps=2,
            topk=2,
            device="cpu",
            seed=41,
            warm_start=True,
        )
        solver.configure(action_space=action_space, n_envs=1, config=config)

        first_outputs = solver.solve(info)
        second_outputs = solver.solve(info)

        self.assertFalse(first_outputs["warm_start"]["used"])
        self.assertTrue(second_outputs["warm_start"]["used"])

    def test_telemetry_records_exact_solver_without_loop_cpu_logging(self) -> None:
        target = torch.zeros(1, 2, 2, dtype=torch.float32)
        info = {"state": torch.zeros(3, 1, dtype=torch.float32)}
        config = SimpleNamespace(horizon=2, action_block=1)
        action_space = DummyBoxSpace(shape=(3, 2))
        solver = CEMExactSolver(
            model=QuadraticCostModel(target),
            batch_size=2,
            num_samples=8,
            var_scale=1.0,
            n_steps=2,
            topk=2,
            device="cpu",
            seed=29,
        )
        solver.configure(action_space=action_space, n_envs=3, config=config)

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.jsonl"
            telemetry = CEMTelemetry(
                metrics_path=metrics_path,
                seed=0,
                mode="summary",
                warmup_plans=0,
                max_recorded_calls=None,
            )
            telemetry.current_plan_index = 3

            self.assertTrue(install_cem_telemetry(solver, telemetry))
            outputs = solver.solve(info)

            rows = read_jsonl(metrics_path)

        cem_rows = [row for row in rows if row.get("type") == "cem_call"]
        self.assertEqual(len(cem_rows), 1)
        row = cem_rows[0]
        self.assertEqual(row["solver_class"], "CEMExactSolver")
        self.assertEqual(row["plan_index"], 3)
        self.assertFalse(row["warmup"])
        self.assertEqual(row["num_samples"], 8)
        self.assertEqual(row["n_steps"], 2)
        self.assertEqual(row["topk"], 2)
        self.assertEqual(row["horizon"], 2)
        self.assertEqual(row["action_dim"], 2)
        self.assertEqual(row["total_envs"], 3)
        self.assertEqual(row["output_cost_count"], 3)
        self.assertEqual(row["output_actions_shape"], [3, 2, 2])
        self.assertFalse(row["cpu_sync_in_loop"])
        self.assertEqual(row["loop_cpu_materialization_count_estimate"], 0)
        self.assertEqual(row["final_cpu_materialization_count_estimate"], 5)
        self.assertEqual(len(outputs["costs"]), 3)
        self.assertIn("cem.cost_reduce_gpu", row["region_count"])
        self.assertIn("cem.cpu_final_cost", row["region_count"])
        self.assertNotIn("cem.cpu_logging", row["region_count"])

if __name__ == "__main__":
    unittest.main()
