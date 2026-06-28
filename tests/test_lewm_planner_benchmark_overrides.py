from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from tools.benchmarks.lewm_plan_benchmark import (
    build_arg_parser,
    effective_planner_settings,
    planner_hydra_overrides,
    resolve_compile_warmup_sample_counts,
    validate_variant_solver_contract,
)
from tools.benchmarks.metrics import write_jsonl
from tools.benchmarks.run_manifest import build_manifest, validate_manifest
from tools.benchmarks.summarize_results import compare_pusht_non_regression


class PlannerOverrideTests(unittest.TestCase):
    def test_planner_overrides_parse(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--variant",
                "warm_start_cem",
                "--solver",
                "cem_exact",
                "--samples",
                "150",
                "--cem-steps",
                "15",
                "--topk",
                "20",
                "--horizon",
                "5",
                "--receding-horizon",
                "1",
                "--action-block",
                "5",
                "--warm-start",
                "true",
                "--solver-warm-start",
                "false",
                "--baseline-run-id",
                "lewm_model_cost_tf32_telem100_l40s",
                "--quality-gate-result",
                "diagnostic",
                "--icem-step-telemetry",
                "summary",
            ]
        )

        self.assertEqual(args.samples, 150)
        self.assertEqual(args.cem_steps, 15)
        self.assertEqual(args.topk, 20)
        self.assertEqual(args.horizon, 5)
        self.assertEqual(args.receding_horizon, 1)
        self.assertEqual(args.action_block, 5)
        self.assertEqual(args.warm_start, "true")
        self.assertEqual(args.solver_warm_start, "false")
        self.assertEqual(args.baseline_run_id, "lewm_model_cost_tf32_telem100_l40s")
        self.assertEqual(args.quality_gate_result, "diagnostic")
        self.assertEqual(args.icem_step_telemetry, "summary")

    def test_planner_hydra_overrides_cover_current_config_fields(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--solver",
                "cem_exact",
                "--samples",
                "150",
                "--cem-steps",
                "15",
                "--topk",
                "20",
                "--horizon",
                "5",
                "--receding-horizon",
                "2",
                "--action-block",
                "5",
                "--warm-start",
                "true",
                "--solver-warm-start",
                "false",
            ]
        )

        self.assertEqual(
            planner_hydra_overrides(args),
            [
                "solver.num_samples=150",
                "solver.n_steps=15",
                "solver.topk=20",
                "plan_config.horizon=5",
                "plan_config.receding_horizon=2",
                "plan_config.action_block=5",
                "plan_config.warm_start=true",
                "solver.warm_start=true",
                "solver.warm_start=false",
            ],
        )

    def test_solver_warm_start_override_is_ignored_for_non_exact_solver(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--solver",
                "cem",
                "--warm-start",
                "true",
                "--solver-warm-start",
                "false",
            ]
        )

        self.assertEqual(
            planner_hydra_overrides(args),
            [
                "plan_config.warm_start=true",
            ],
        )

    def test_icem_solver_overrides_cover_within_solve_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--solver",
                "icem",
                "--elite-retention",
                "5",
                "--colored-noise",
                "pink",
                "--colored-noise-time",
                "raw",
                "--momentum-alpha",
                "0.1",
                "--sample-decay",
                "1.25",
                "--sample-count-bucket-size",
                "60",
                "--return-best-elite",
                "true",
                "--clip-actions",
                "false",
            ]
        )

        self.assertEqual(
            planner_hydra_overrides(args),
            [
                "solver.elite_retention=5",
                "solver.colored_noise=pink",
                "solver.colored_noise_time=raw",
                "solver.momentum_alpha=0.1",
                "solver.sample_decay=1.25",
                "solver.sample_count_bucket_size=60",
                "solver.return_best_elite=true",
                "solver.clip_actions=false",
            ],
        )

    def test_effective_planner_settings_resolve_hydra_values_and_metadata(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--warm-start",
                "true",
                "--elite-retention",
                "5",
                "--colored-noise",
                "pink",
                "--colored-noise-time",
                "raw",
                "--momentum-alpha",
                "0.1",
                "--sample-decay",
                "1.25",
                "--sample-count-bucket-size",
                "60",
                "--return-best-elite",
                "true",
                "--clip-actions",
                "false",
                "--mppi-temperature",
                "0.7",
                "--gd-seed-solver",
                "icem",
                "--gd-seed-source",
                "best_elite",
                "--gd-refine-topk",
                "3",
                "--gd-steps",
                "2",
                "--gd-lr",
                "0.05",
                "--gd-compile-cost",
                "false",
                "--model-cost-autocast",
                "bf16",
                "--icem-step-telemetry",
                "summary",
            ]
        )
        cfg = SimpleNamespace(
            solver=SimpleNamespace(
                num_samples=150,
                n_steps=15,
                topk=20,
                warm_start=False,
                elite_retention=5,
                colored_noise="pink",
                colored_noise_time="raw",
                momentum_alpha=0.1,
                sample_decay=1.25,
                sample_count_bucket_size=60,
                return_best_elite=True,
                clip_actions=False,
                gd_seed_solver="icem",
                gd_seed_source="best_elite",
                gd_refine_topk=3,
                gd_steps=2,
                gd_lr=0.05,
                gd_compile_cost=False,
            ),
            plan_config=SimpleNamespace(horizon=5, receding_horizon=2, action_block=5),
        )

        settings = effective_planner_settings(args, cfg)

        self.assertEqual(settings["samples"], 150)
        self.assertEqual(settings["cem_steps"], 15)
        self.assertEqual(settings["model_cost_evals_per_cem_call"], 1230)
        self.assertEqual(settings["horizon"], 5)
        self.assertEqual(settings["receding_horizon"], 2)
        self.assertEqual(settings["action_block"], 5)
        self.assertTrue(settings["warm_start"])
        self.assertFalse(settings["solver_warm_start"])
        self.assertEqual(settings["elite_retention"], 5)
        self.assertEqual(settings["colored_noise"], "pink")
        self.assertEqual(settings["colored_noise_time"], "raw")
        self.assertEqual(settings["momentum_alpha"], 0.1)
        self.assertEqual(settings["sample_decay"], 1.25)
        self.assertEqual(settings["sample_count_bucket_size"], 60)
        self.assertTrue(settings["return_best_elite"])
        self.assertFalse(settings["clip_actions"])
        self.assertEqual(settings["mppi_temperature"], 0.7)
        self.assertEqual(settings["gd_seed_solver"], "icem")
        self.assertEqual(settings["gd_seed_source"], "best_elite")
        self.assertEqual(settings["gd_refine_topk"], 3)
        self.assertEqual(settings["gd_steps"], 2)
        self.assertEqual(settings["gd_lr"], 0.05)
        self.assertFalse(settings["gd_compile_cost"])
        self.assertEqual(settings["model_cost_autocast"], "bf16")
        self.assertEqual(settings["icem_step_telemetry"], "summary")

    def test_cem_gd_solver_overrides_cover_seed_and_gd_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--solver",
                "cem_gd",
                "--elite-retention",
                "5",
                "--colored-noise",
                "none",
                "--colored-noise-time",
                "raw",
                "--momentum-alpha",
                "0.0",
                "--sample-decay",
                "1.25",
                "--sample-count-bucket-size",
                "60",
                "--return-best-elite",
                "false",
                "--clip-actions",
                "false",
                "--gd-seed-solver",
                "icem",
                "--gd-seed-source",
                "best_elite",
                "--gd-refine-topk",
                "1",
                "--gd-steps",
                "1",
                "--gd-lr",
                "0.01",
                "--gd-compile-cost",
                "false",
            ]
        )

        self.assertEqual(
            planner_hydra_overrides(args),
            [
                "solver.elite_retention=5",
                "solver.colored_noise=none",
                "solver.colored_noise_time=raw",
                "solver.momentum_alpha=0.0",
                "solver.sample_decay=1.25",
                "solver.sample_count_bucket_size=60",
                "solver.return_best_elite=false",
                "solver.clip_actions=false",
                "solver.gd_seed_solver=icem",
                "solver.gd_seed_source=best_elite",
                "solver.gd_refine_topk=1",
                "solver.gd_steps=1",
                "solver.gd_lr=0.01",
                "solver.gd_compile_cost=false",
            ],
        )

    def test_static_compile_solver_overrides_cover_static_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--solver",
                "icem_static_compile",
                "--elite-retention",
                "5",
                "--colored-noise",
                "none",
                "--colored-noise-time",
                "raw",
                "--momentum-alpha",
                "0.0",
                "--sample-decay",
                "1.25",
                "--sample-count-bucket-size",
                "60",
                "--return-best-elite",
                "false",
                "--clip-actions",
                "false",
                "--static-chain-compile",
                "true",
                "--static-chain-mode",
                "reduce-overhead",
                "--static-chain-fullgraph",
                "false",
                "--static-chain-warmup",
                "true",
            ]
        )

        self.assertEqual(
            planner_hydra_overrides(args),
            [
                "solver.elite_retention=5",
                "solver.colored_noise=none",
                "solver.colored_noise_time=raw",
                "solver.momentum_alpha=0.0",
                "solver.sample_decay=1.25",
                "solver.sample_count_bucket_size=60",
                "solver.return_best_elite=false",
                "solver.clip_actions=false",
                "solver.static_chain_compile=true",
                "solver.static_chain_mode=reduce-overhead",
                "solver.static_chain_fullgraph=false",
                "solver.static_chain_warmup=true",
            ],
        )

    def test_static_compile_variant_requires_static_solver(self) -> None:
        parser = build_arg_parser()
        bad_args = parser.parse_args(["--variant", "icem_static_compile", "--solver", "icem"])

        with self.assertRaisesRegex(ValueError, "--solver icem_static_compile"):
            validate_variant_solver_contract(bad_args)

        good_args = parser.parse_args(
            [
                "--variant",
                "icem_static_compile",
                "--solver",
                "icem_static_compile",
                "--compile-diagnostics",
                "none",
            ]
        )

        validate_variant_solver_contract(good_args)

    def test_icem_gd_variant_requires_icem_seed_solver(self) -> None:
        parser = build_arg_parser()
        bad_args = parser.parse_args(["--variant", "icem_gd", "--solver", "cem_gd"])

        with self.assertRaisesRegex(ValueError, "--gd-seed-solver icem"):
            validate_variant_solver_contract(bad_args)

        good_args = parser.parse_args(
            [
                "--variant",
                "icem_gd",
                "--solver",
                "cem_gd",
                "--gd-seed-solver",
                "icem",
                "--cem-telemetry",
                "summary",
                "--compile-diagnostics",
                "none",
            ]
        )

        validate_variant_solver_contract(good_args)

    def test_model_cost_eval_count_uses_bucketed_sample_decay(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--sample-decay", "1.25", "--sample-count-bucket-size", "60"])
        cfg = SimpleNamespace(
            solver=SimpleNamespace(
                num_samples=300,
                n_steps=30,
                topk=30,
                sample_decay=1.25,
                sample_count_bucket_size=60,
            ),
            plan_config=SimpleNamespace(horizon=5, receding_horizon=5, action_block=5),
        )

        settings = effective_planner_settings(args, cfg)

        self.assertEqual(settings["model_cost_evals_per_cem_call"], 2820)

    def test_compile_warmup_sample_counts_auto_uses_unique_bucketed_counts(self) -> None:
        cfg = SimpleNamespace(
            solver=SimpleNamespace(
                num_samples=300,
                n_steps=30,
                topk=30,
                sample_decay=1.25,
                sample_count_bucket_size=60,
            ),
        )

        counts = resolve_compile_warmup_sample_counts("auto", cfg)

        self.assertEqual(counts, [300, 240, 180, 120, 60])

    def test_compile_warmup_sample_counts_parse_explicit_list(self) -> None:
        cfg = SimpleNamespace(solver=SimpleNamespace(num_samples=300))

        counts = resolve_compile_warmup_sample_counts("300,240,240,60", cfg)

        self.assertEqual(counts, [300, 240, 60])

    def test_planner_manifest_validation_requires_quality_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.jsonl"
            metrics.write_text("", encoding="utf-8")
            manifest = build_manifest(
                run_id="run_1",
                model="lewm-pusht",
                variant="warm_start_cem",
                semantic_class="planner",
                cwd=root,
                command=["python", "x.py"],
                weights_sha256="a" * 64,
                artifacts={"metrics_jsonl": str(metrics)},
                baseline_run_id="lewm_model_cost_tf32_telem100_l40s",
                quality_gate_result="diagnostic",
                seed_list=[0],
                num_eval_episodes=2,
            )
            manifest["environment"]["torch"] = "test"
            manifest["environment"]["gpu_name"] = "test-gpu"
            manifest["base_git_commit"] = "b" * 40
            manifest["git_diff_sha256"] = "c" * 64

            self.assertEqual(
                validate_manifest(manifest, require_gpu_environment=False, planner_changing=True),
                [],
            )

            incomplete = json.loads(json.dumps(manifest))
            incomplete["baseline_run_id"] = None
            errors = validate_manifest(
                incomplete,
                require_gpu_environment=False,
                planner_changing=True,
            )
            self.assertIn("planner run missing field: baseline_run_id", errors)

    def test_pusht_quality_gate_uses_success_and_model_cost_counts(self) -> None:
        baseline = [
            {
                "type": "evaluation_result",
                "metrics": {"episode_successes": [True] * 20},
                "cem_calls_total": 2,
                "model_cost_evals_total": 600,
            }
        ]
        candidate = [
            {
                "type": "evaluation_result",
                "metrics": {"episode_successes": [True] * 19 + [False]},
                "cem_calls_total": 1,
                "model_cost_evals_total": 150,
            }
        ]

        result = compare_pusht_non_regression(baseline, candidate)

        self.assertTrue(result["quality_pass"])
        self.assertAlmostEqual(result["success_rate_delta"], -0.05)
        self.assertEqual(result["baseline_cem_calls"], 2)
        self.assertEqual(result["candidate_cem_calls"], 1)
        self.assertEqual(result["baseline_model_cost_evals"], 600)
        self.assertEqual(result["candidate_model_cost_evals"], 150)
        self.assertFalse(result["task_score_gate_applied"])

    def test_pusht_quality_gate_fails_large_success_regression(self) -> None:
        baseline = [
            {
                "type": "evaluation_result",
                "metrics": {"episode_successes": [True] * 10},
            }
        ]
        candidate = [
            {
                "type": "evaluation_result",
                "metrics": {"episode_successes": [True] * 4 + [False] * 6},
            }
        ]

        result = compare_pusht_non_regression(baseline, candidate)

        self.assertFalse(result["quality_pass"])
        self.assertAlmostEqual(result["success_rate_delta"], -0.6)

    def test_quality_gate_cli_writes_expected_json_schema(self) -> None:
        from tools.benchmarks.summarize_results import main as summarize_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            out_path = root / "quality_gate.json"
            write_jsonl(
                baseline_path,
                [
                    {
                        "type": "evaluation_result",
                        "metrics": {"episode_successes": [True] * 20},
                        "cem_calls_total": 2,
                        "model_cost_evals_total": 600,
                    }
                ],
            )
            write_jsonl(
                candidate_path,
                [
                    {
                        "type": "evaluation_result",
                        "metrics": {"episode_successes": [True] * 19 + [False]},
                        "cem_calls_total": 1,
                        "model_cost_evals_total": 150,
                    }
                ],
            )

            with redirect_stdout(StringIO()):
                result = summarize_main(
                    [
                        "--baseline",
                        str(baseline_path),
                        "--candidate",
                        str(candidate_path),
                        "--quality-gate",
                        "pusht_non_regression",
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(result, 0)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["quality_pass"])
            self.assertIn("candidate_model_cost_evals", payload)


if __name__ == "__main__":
    unittest.main()
