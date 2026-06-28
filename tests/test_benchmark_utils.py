from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.benchmarks.check_lewm_cache_exactness import (
    build_arg_parser as build_exactness_arg_parser,
    relative_error,
    summarize_candidate_exactness_result,
    summarize_exactness_result,
    summarize_numerical_candidate_result,
)
from tools.benchmarks.hash_artifacts import hash_path
from tools.benchmarks.lewm_plan_benchmark import (
    build_arg_parser as build_lewm_plan_arg_parser,
    compare_compile_signatures,
    compile_cost_requested,
    configure_model_cost_autocast,
    configure_model_for_variant,
    dataset_candidates,
    diff_numeric_trees,
    effective_compile_diagnostics,
    prepare_run_artifacts,
    precomputed_hash,
    validate_variant_solver_contract,
)
from tools.benchmarks import profile_lewm_cem, profile_lewm_model_cost
from tools.benchmarks.metrics import (
    append_jsonl,
    read_jsonl,
    summarize_cem_calls,
    summarize_evaluations,
    summarize_plan_latencies,
    summarize_wall_attribution,
)
from tools.benchmarks.optimization_ledger import main as optimization_ledger_main
from tools.benchmarks.optimization_ledger import upsert_row
from tools.benchmarks.run_manifest import (
    build_manifest,
    discover_cgroup_cpu_quota,
    discover_environment,
    discover_runtime_resources,
    git_metadata,
    validate_manifest,
    write_manifest,
)
from tools.benchmarks.stage_lewm_weights import (
    dataset_paths,
    hydra_kwargs,
    load_model_config,
    norm_target_name,
)
from tools.benchmarks.summarize_results import summarize_quality
from tools.benchmarks.summarize_torch_trace import summarize_events


class BenchmarkUtilityTests(unittest.TestCase):
    def test_cache_exactness_summary_requires_cost_rank_and_reuse_checks(self) -> None:
        self.assertTrue(
            summarize_exactness_result(
                baseline_cache_allclose=True,
                cache_repeat_allclose=True,
                argmin_equal=True,
                topk_equal=True,
                baseline_encode_total=60,
                cache_encode_total=2,
                cache_repeat_encode_delta=0,
            )
        )
        self.assertFalse(
            summarize_exactness_result(
                baseline_cache_allclose=True,
                cache_repeat_allclose=True,
                argmin_equal=True,
                topk_equal=False,
                baseline_encode_total=60,
                cache_encode_total=2,
                cache_repeat_encode_delta=0,
            )
        )
        self.assertFalse(
            summarize_exactness_result(
                baseline_cache_allclose=True,
                cache_repeat_allclose=True,
                argmin_equal=True,
                topk_equal=True,
                baseline_encode_total=60,
                cache_encode_total=2,
                cache_repeat_encode_delta=1,
            )
        )

    def test_candidate_exactness_summary_requires_cost_repeat_and_rankings(self) -> None:
        self.assertTrue(
            summarize_candidate_exactness_result(
                control_candidate_allclose=True,
                candidate_repeat_allclose=True,
                argmin_equal=True,
                topk_equal=True,
            )
        )
        self.assertFalse(
            summarize_candidate_exactness_result(
                control_candidate_allclose=True,
                candidate_repeat_allclose=False,
                argmin_equal=True,
                topk_equal=True,
            )
        )
        self.assertFalse(
            summarize_candidate_exactness_result(
                control_candidate_allclose=True,
                candidate_repeat_allclose=True,
                argmin_equal=True,
                topk_equal=False,
            )
        )

    def test_numerical_candidate_summary_allows_control_cost_drift_but_requires_rankings(self) -> None:
        self.assertTrue(
            summarize_numerical_candidate_result(
                candidate_repeat_allclose=True,
                argmin_equal=True,
                topk_equal=True,
            )
        )
        self.assertFalse(
            summarize_numerical_candidate_result(
                candidate_repeat_allclose=False,
                argmin_equal=True,
                topk_equal=True,
            )
        )
        self.assertFalse(
            summarize_numerical_candidate_result(
                candidate_repeat_allclose=True,
                argmin_equal=False,
                topk_equal=True,
            )
        )

    def test_exactness_parser_accepts_rollout_action_cache_candidate_mode(self) -> None:
        parser = build_exactness_arg_parser()

        args = parser.parse_args(["--candidate-mode", "rollout_action_cache"])

        self.assertEqual(args.candidate_mode, "rollout_action_cache")

    def test_exactness_parser_accepts_compile_cost_candidate_mode(self) -> None:
        parser = build_exactness_arg_parser()

        args = parser.parse_args(["--candidate-mode", "compile_cost"])

        self.assertEqual(args.candidate_mode, "compile_cost")

    def test_exactness_parser_accepts_model_cost_tf32_candidate_mode(self) -> None:
        parser = build_exactness_arg_parser()

        args = parser.parse_args(["--candidate-mode", "model_cost_tf32"])

        self.assertEqual(args.candidate_mode, "model_cost_tf32")
        self.assertEqual(args.control_matmul_precision, "default")
        self.assertEqual(args.candidate_matmul_precision, "high")

    def test_relative_error_handles_zero_reference_scale(self) -> None:
        self.assertEqual(relative_error(0.0, 0.0), 0.0)
        self.assertEqual(relative_error(2.0, 4.0), 0.5)
        self.assertEqual(relative_error(1.0, 0.0), float("inf"))

    def test_compile_warmup_history_defaults_to_runtime_observation_history(self) -> None:
        parser = build_lewm_plan_arg_parser()

        args = parser.parse_args(["--variant", "compile_cost", "--solver", "cem_exact"])

        self.assertEqual(args.compile_warmup_history, 1)

    def test_compile_cost_variant_requires_cem_exact_solver(self) -> None:
        parser = build_lewm_plan_arg_parser()

        bad_args = parser.parse_args(["--variant", "compile_cost"])
        with self.assertRaisesRegex(ValueError, "--solver cem_exact"):
            validate_variant_solver_contract(bad_args)

        missing_telemetry = parser.parse_args(["--variant", "compile_cost", "--solver", "cem_exact"])
        with self.assertRaisesRegex(ValueError, "compile diagnostics require"):
            validate_variant_solver_contract(missing_telemetry)

        good_args = parser.parse_args(
            [
                "--variant",
                "compile_cost",
                "--solver",
                "cem_exact",
                "--cem-telemetry",
                "summary",
            ]
        )
        validate_variant_solver_contract(good_args)

    def test_model_cost_tf32_variant_uses_compile_cost_contract(self) -> None:
        parser = build_lewm_plan_arg_parser()

        args = parser.parse_args(["--variant", "model_cost_tf32", "--solver", "cem_exact"])

        self.assertTrue(compile_cost_requested(args))
        self.assertEqual(effective_compile_diagnostics(args), "summary")

        missing_telemetry = parser.parse_args(
            ["--variant", "model_cost_tf32", "--solver", "cem_exact"]
        )
        with self.assertRaisesRegex(ValueError, "compile diagnostics require"):
            validate_variant_solver_contract(missing_telemetry)

        good_args = parser.parse_args(
            [
                "--variant",
                "model_cost_tf32",
                "--solver",
                "cem_exact",
                "--cem-telemetry",
                "summary",
            ]
        )
        validate_variant_solver_contract(good_args)

        with self.assertRaisesRegex(ValueError, "--solver cem_exact"):
            validate_variant_solver_contract(
                parser.parse_args(["--variant", "model_cost_tf32", "--solver", "cem"])
            )

    def test_model_cost_autocast_parser_accepts_bf16(self) -> None:
        parser = build_lewm_plan_arg_parser()

        args = parser.parse_args(["--model-cost-autocast", "bf16"])

        self.assertEqual(args.model_cost_autocast, "bf16")

    def test_configure_model_cost_autocast_sets_supported_model(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.mode = "none"

            def set_model_cost_autocast(self, mode: str) -> None:
                self.mode = mode

            def model_cost_autocast_mode(self) -> str:
                return self.mode

            def model_cost_autocast_enabled(self) -> bool:
                return self.mode != "none"

            def model_cost_autocast_dtype(self) -> str | None:
                return "torch.bfloat16" if self.mode == "bf16" else None

        info = configure_model_cost_autocast(DummyModel(), "bf16")

        self.assertEqual(info["active"], "bf16")
        self.assertTrue(info["enabled"])
        self.assertEqual(info["dtype"], "torch.bfloat16")

    def test_retired_l5_variants_fail_loudly(self) -> None:
        parser = build_lewm_plan_arg_parser()

        for variant in (
            "tensor_cem_compile",
            "tensor_cem_compile_no_inner_compile",
            "tensor_cem_static",
        ):
            args = parser.parse_args(["--variant", variant, "--solver", "cem_exact"])
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "retired after the L5 closeout"):
                    validate_variant_solver_contract(args)

    def test_compile_diagnostics_defaults_to_summary_for_compile_cost(self) -> None:
        parser = build_lewm_plan_arg_parser()

        compile_args = parser.parse_args(["--variant", "compile_cost", "--solver", "cem_exact"])
        baseline_args = parser.parse_args(["--variant", "baseline"])
        disabled_args = parser.parse_args(
            ["--variant", "compile_cost", "--solver", "cem_exact", "--compile-diagnostics", "none"]
        )

        self.assertEqual(effective_compile_diagnostics(compile_args), "summary")
        self.assertEqual(effective_compile_diagnostics(baseline_args), "none")
        self.assertEqual(effective_compile_diagnostics(disabled_args), "none")

    def test_compile_signature_comparison_reports_shape_dtype_and_device(self) -> None:
        expected = {
            "emb": {"shape": [1, 3, 192], "dtype": "torch.float32", "device": "cuda:0"},
            "goal_emb": {"shape": [1, 3, 192], "dtype": "torch.float32", "device": "cuda:0"},
            "action_candidates": {
                "shape": [1, 300, 5, 10],
                "dtype": "torch.float32",
                "device": "cuda:0",
            },
        }
        actual = {
            "emb": {"shape": [1, 2, 192], "dtype": "torch.float32", "device": "cuda:0"},
            "goal_emb": {"shape": [1, 3, 192], "dtype": "torch.float16", "device": "cuda:0"},
            "action_candidates": {
                "shape": [1, 300, 5, 10],
                "dtype": "torch.float32",
                "device": "cpu",
            },
        }

        comparison = compare_compile_signatures(expected, actual)

        self.assertFalse(comparison["matches"])
        self.assertEqual(
            comparison["mismatch_fields"],
            [
                {
                    "tensor": "action_candidates",
                    "field": "device",
                    "expected": "cuda:0",
                    "actual": "cpu",
                },
                {
                    "tensor": "emb",
                    "field": "shape",
                    "expected": [1, 3, 192],
                    "actual": [1, 2, 192],
                },
                {
                    "tensor": "goal_emb",
                    "field": "dtype",
                    "expected": "torch.float32",
                    "actual": "torch.float16",
                },
            ],
        )

    def test_diff_numeric_trees_keeps_only_changed_numeric_leaves(self) -> None:
        before = {
            "sources": {
                "dynamo": {
                    "frames": {"total": 1, "ok": 1},
                    "graph_break": {"count": 0},
                }
            }
        }
        after = {
            "sources": {
                "dynamo": {
                    "frames": {"total": 3, "ok": 2},
                    "graph_break": {"count": 0},
                    "inductor": {"fxgraph_cache_hit": 1},
                }
            }
        }

        self.assertEqual(
            diff_numeric_trees(before, after),
            {
                "sources": {
                    "dynamo": {
                        "frames": {"ok": 1, "total": 2},
                        "inductor": {"fxgraph_cache_hit": 1},
                    }
                }
            },
        )

    def test_configure_model_for_variant_toggles_embedding_cache(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.cache_enabled = None
                self.rollout_prealloc_enabled_value = None
                self.rollout_action_cache_enabled_value = None
                self.compile_cost_enabled_value = None

            def set_embedding_cache(self, enabled: bool = True):
                self.cache_enabled = enabled
                return self

            def set_rollout_prealloc(self, enabled: bool = True):
                self.rollout_prealloc_enabled_value = enabled
                return self

            def set_rollout_action_cache(self, enabled: bool = True):
                self.rollout_action_cache_enabled_value = enabled
                return self

            def set_compile_cost(self, enabled: bool = True):
                self.compile_cost_enabled_value = enabled
                return self

        model = DummyModel()

        configure_model_for_variant(model, "cache_embeddings")
        self.assertTrue(model.cache_enabled)
        self.assertFalse(model.rollout_prealloc_enabled_value)
        self.assertFalse(model.rollout_action_cache_enabled_value)

        configure_model_for_variant(model, "cem_exact_no_cpu_sync")
        self.assertTrue(model.cache_enabled)
        self.assertFalse(model.rollout_prealloc_enabled_value)
        self.assertFalse(model.rollout_action_cache_enabled_value)

        configure_model_for_variant(model, "cem_exact_no_cpu_sync_only")
        self.assertFalse(model.cache_enabled)
        self.assertFalse(model.rollout_prealloc_enabled_value)
        self.assertFalse(model.rollout_action_cache_enabled_value)
        self.assertFalse(model.compile_cost_enabled_value)

        configure_model_for_variant(model, "compile_cost")
        self.assertTrue(model.cache_enabled)
        self.assertFalse(model.rollout_prealloc_enabled_value)
        self.assertFalse(model.rollout_action_cache_enabled_value)
        self.assertFalse(model.compile_cost_enabled_value)

        configure_model_for_variant(model, "model_cost_tf32")
        self.assertTrue(model.cache_enabled)
        self.assertFalse(model.rollout_prealloc_enabled_value)
        self.assertFalse(model.rollout_action_cache_enabled_value)
        self.assertFalse(model.compile_cost_enabled_value)

        configure_model_for_variant(model, "rollout_prealloc")
        self.assertTrue(model.cache_enabled)
        self.assertTrue(model.rollout_prealloc_enabled_value)
        self.assertFalse(model.rollout_action_cache_enabled_value)

        configure_model_for_variant(model, "rollout_action_cache")
        self.assertTrue(model.cache_enabled)
        self.assertFalse(model.rollout_prealloc_enabled_value)
        self.assertTrue(model.rollout_action_cache_enabled_value)

        configure_model_for_variant(model, "baseline")
        self.assertFalse(model.cache_enabled)
        self.assertFalse(model.rollout_prealloc_enabled_value)
        self.assertFalse(model.rollout_action_cache_enabled_value)
        self.assertFalse(model.compile_cost_enabled_value)

    def test_environment_snapshot_records_runtime_package_versions(self) -> None:
        env = discover_environment()

        self.assertIn("packages", env)
        self.assertIn("resources", env)
        self.assertIn("stable_worldmodel", env["packages"])
        self.assertIn("stable_pretraining", env["packages"])
        self.assertIn("modal", env["packages"])
        self.assertIn("cpu_count", env["resources"])
        self.assertIn("cgroup_cpu", env["resources"])
        self.assertIn("torch_num_threads", env["resources"])

    def test_runtime_resource_snapshot_records_thread_and_benchmark_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "8",
                "JEPA_MODAL_GPU_TYPE": "L40S",
                "JEPA_MODAL_CPU_REQUEST": "8.0",
                "JEPA_MODAL_CPU_LIMIT": "8.0",
            },
        ):
            resources = discover_runtime_resources()

        self.assertEqual(resources["thread_env"]["OMP_NUM_THREADS"], "8")
        self.assertEqual(resources["benchmark_resource_env"]["JEPA_MODAL_GPU_TYPE"], "L40S")
        self.assertEqual(resources["benchmark_resource_env"]["JEPA_MODAL_CPU_REQUEST"], "8.0")
        self.assertEqual(resources["benchmark_resource_env"]["JEPA_MODAL_CPU_LIMIT"], "8.0")
        self.assertIn("cuda_device_count", resources)
        self.assertIn("cgroup_cpu", resources)

    def test_runtime_resource_snapshot_parses_cgroup_v2_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cpu.max").write_text("800000 100000\n", encoding="utf-8")

            quota = discover_cgroup_cpu_quota(root)

        self.assertEqual(quota["source"], "cgroup_v2")
        self.assertEqual(quota["quota_us"], 800000)
        self.assertEqual(quota["period_us"], 100000)
        self.assertEqual(quota["quota_cores"], 8.0)

    def test_runtime_resource_snapshot_parses_unlimited_cgroup_v2_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cpu.max").write_text("max 100000\n", encoding="utf-8")

            quota = discover_cgroup_cpu_quota(root)

        self.assertEqual(quota["source"], "cgroup_v2")
        self.assertEqual(quota["quota"], "max")
        self.assertIsNone(quota["quota_cores"])

    def test_git_metadata_prefers_injected_modal_source_snapshot(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JEPA_BASE_GIT_COMMIT": "b" * 40,
                "JEPA_GIT_DIFF_SHA256": "c" * 64,
            },
        ):
            metadata = git_metadata(Path("/path/without/git"))

        self.assertEqual(
            metadata,
            {
                "base_git_commit": "b" * 40,
                "git_diff_sha256": "c" * 64,
            },
        )

    def test_hash_path_hashes_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "a.txt"
            file_path.write_text("hello\n", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.txt").write_text("world\n", encoding="utf-8")

            file_hash = hash_path(file_path)
            dir_hash = hash_path(root)

            self.assertEqual(file_hash["type"], "file")
            self.assertEqual(file_hash["size_bytes"], 6)
            self.assertEqual(dir_hash["type"], "directory")
            self.assertEqual(dir_hash["file_count"], 2)
            self.assertEqual(len(file_hash["sha256"]), 64)
            self.assertEqual(len(dir_hash["sha256"]), 64)

    def test_jsonl_latency_summary_excludes_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics.jsonl"
            append_jsonl(metrics, {"type": "plan_latency", "warmup": True, "plan_latency_ms": 100.0})
            append_jsonl(metrics, {"type": "plan_latency", "warmup": False, "plan_latency_ms": 10.0})
            append_jsonl(metrics, {"type": "plan_latency", "warmup": False, "plan_latency_ms": 20.0})

            summary = summarize_plan_latencies(read_jsonl(metrics))

            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["mean"], 15.0)
            self.assertEqual(summary["p50"], 15.0)

    def test_prepare_run_artifacts_clears_benchmark_owned_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metrics = run_dir / "metrics.jsonl"
            manifest = run_dir / "manifest.json"
            stdout = run_dir / "stdout.txt"
            profiler = run_dir / "profiler"
            videos = run_dir / "videos"

            metrics.write_text('{"type":"old"}\n', encoding="utf-8")
            manifest.write_text('{"old":true}\n', encoding="utf-8")
            (run_dir / "profiler_summary.json").write_text('{"old":true}\n', encoding="utf-8")
            stdout.write_text("parent log\n", encoding="utf-8")
            profiler.mkdir()
            (profiler / "old.trace.json").write_text("{}", encoding="utf-8")
            videos.mkdir()
            (videos / "old.mp4").write_bytes(b"old")

            prepare_run_artifacts(run_dir, metrics, manifest, profile="torch")

            self.assertFalse(metrics.exists())
            self.assertFalse(manifest.exists())
            self.assertFalse((run_dir / "profiler_summary.json").exists())
            self.assertFalse(profiler.exists())
            self.assertFalse(videos.exists())
            self.assertTrue(stdout.exists())

    def test_profile_entrypoint_defaults_to_torch_with_cuda_event_cem_telemetry(self) -> None:
        with patch.object(profile_lewm_cem.lewm_plan_benchmark, "main", return_value=0) as main:
            result = profile_lewm_cem.main(["--variant", "cache_embeddings", "--seed", "0"])

        self.assertEqual(result, 0)
        forwarded = main.call_args.args[0]
        self.assertEqual(forwarded[forwarded.index("--profile") + 1], "torch")
        self.assertEqual(forwarded[forwarded.index("--cem-telemetry") + 1], "cuda-events")

    def test_model_cost_profile_parser_defaults_to_phase3r_control_shape(self) -> None:
        parser = profile_lewm_model_cost.build_arg_parser()

        args = parser.parse_args([])

        self.assertEqual(args.variant, "compile_cost")
        self.assertEqual(args.solver, "cem_exact")
        self.assertEqual(args.profile, "torch")
        self.assertEqual([args.samples, args.horizon, args.action_dim or 10], [300, 5, 10])
        self.assertEqual(args.compile_cost_mode, "reduce-overhead")
        self.assertFalse(args.compile_cost_fullgraph)
        self.assertFalse(args.model_cost_region_profile)

        diagnostic_args = parser.parse_args(["--model-cost-region-profile"])
        self.assertTrue(diagnostic_args.model_cost_region_profile)

    def test_model_cost_profile_rejects_non_phase3r_candidate_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[B, 300, 5, 10\]"):
            profile_lewm_model_cost.validate_profile_shape(
                batch_size=1,
                samples=128,
                horizon=5,
                action_dim=10,
                require_phase3r_shape=True,
            )

    def test_model_cost_summary_excludes_warmup_iterations(self) -> None:
        summary = profile_lewm_model_cost.summarize_model_cost_rows(
            [
                {
                    "type": "model_cost_iteration",
                    "warmup": True,
                    "elapsed_ms": 100.0,
                    "timing_source": "cuda_events",
                },
                {
                    "type": "model_cost_iteration",
                    "warmup": False,
                    "elapsed_ms": 4.0,
                    "timing_source": "cuda_events",
                },
                {
                    "type": "model_cost_iteration",
                    "warmup": False,
                    "elapsed_ms": 6.0,
                    "timing_source": "cuda_events",
                },
            ]
        )

        self.assertEqual(summary["iteration_count"], 2)
        self.assertEqual(summary["elapsed_ms"]["mean"], 5.0)
        self.assertEqual(summary["timing_source"], ["cuda_events"])

    def test_score_reduction_gate_uses_named_profiler_ranges(self) -> None:
        summary = profile_lewm_model_cost.summarize_score_reduction_gate(
            {
                "user_ranges": {
                    "lewm.model_cost_profile.get_cost_from_embeddings": {"total_ms": 100.0},
                    "lewm.cost_from_rollout": {"total_ms": 4.0},
                }
            }
        )

        self.assertEqual(summary["model_cost_total_ms"], 100.0)
        self.assertEqual(summary["score_total_ms"], 4.0)
        self.assertEqual(summary["score_to_model_cost_ratio"], 0.04)
        self.assertFalse(summary["passes_model_cost_gate"])

        passing = profile_lewm_model_cost.summarize_score_reduction_gate(
            {
                "user_ranges": {
                    "lewm.model_cost_profile.get_cost_from_embeddings": {"total_ms": 100.0},
                    "lewm.cost_from_rollout": {"total_ms": 12.0},
                }
            }
        )

        self.assertTrue(passing["passes_model_cost_gate"])

    def test_score_reduction_cuda_gate_uses_measured_iteration_rows(self) -> None:
        rows = [
            {
                "type": "model_cost_iteration",
                "warmup": True,
                "elapsed_ms": 100.0,
                "region_cuda_ms": {"lewm.cost_from_rollout": 50.0},
            },
            {
                "type": "model_cost_iteration",
                "warmup": False,
                "elapsed_ms": 10.0,
                "region_cuda_ms": {
                    "lewm.rollout_from_embeddings": 9.5,
                    "lewm.cost_from_rollout": 0.4,
                },
            },
            {
                "type": "model_cost_iteration",
                "warmup": False,
                "elapsed_ms": 20.0,
                "region_cuda_ms": {
                    "lewm.rollout_from_embeddings": 18.5,
                    "lewm.cost_from_rollout": 0.8,
                },
            },
        ]

        regions = profile_lewm_model_cost.summarize_model_cost_region_rows(rows)
        gate = profile_lewm_model_cost.summarize_score_reduction_cuda_gate(rows)

        self.assertEqual(regions["lewm.cost_from_rollout"]["count"], 2)
        self.assertAlmostEqual(regions["lewm.cost_from_rollout"]["total_ms"], 1.2)
        self.assertEqual(gate["model_cost_total_ms"], 30.0)
        self.assertAlmostEqual(gate["score_total_ms"], 1.2)
        self.assertAlmostEqual(gate["score_to_model_cost_ratio"], 0.04)
        self.assertFalse(gate["passes_model_cost_gate"])

    def test_cem_summary_excludes_warmup_and_aggregates_regions(self) -> None:
        rows = [
            {
                "type": "cem_call",
                "warmup": True,
                "cpu_wall_ms": 100.0,
                "cuda_elapsed_ms": 90.0,
                "region_cpu_ms": {"cem.cpu_logging": 80.0},
                "region_count": {"cem.cpu_logging": 1},
            },
            {
                "type": "cem_call",
                "warmup": False,
                "cpu_wall_ms": 10.0,
                "cuda_elapsed_ms": 8.0,
                "total_envs": 1,
                "output_cost_count": 1,
                "loop_cpu_materialization_count_estimate": 0,
                "final_cpu_materialization_count_estimate": 4,
                "cpu_materialization_count_estimate": 1,
                "active_env_count": 1,
                "cuda_memory_peak_allocated_mb": 512.0,
                "compile_diagnostics_enabled": True,
                "compile_runtime_signature_matches_expected": True,
                "compile_runtime_signature_unique_count": 1,
                "compile_runtime_signature_observation_count": 30,
                "compile_counter_snapshot_available": True,
                "region_cpu_ms": {"cem.model_get_cost": 6.0, "cem.topk": 1.0},
                "region_count": {"cem.model_get_cost": 2, "cem.topk": 2},
            },
            {
                "type": "cem_call",
                "warmup": False,
                "cpu_wall_ms": 30.0,
                "cuda_elapsed_ms": 24.0,
                "total_envs": 1,
                "output_cost_count": 1,
                "loop_cpu_materialization_count_estimate": 30,
                "final_cpu_materialization_count_estimate": 3,
                "cpu_materialization_count_estimate": 3,
                "active_env_count": 1,
                "cuda_memory_peak_allocated_mb": 768.0,
                "compile_diagnostics_enabled": True,
                "compile_runtime_signature_matches_expected": False,
                "compile_runtime_signature_unique_count": 2,
                "compile_runtime_signature_observation_count": 30,
                "compile_counter_snapshot_available": False,
                "region_cpu_ms": {"cem.model_get_cost": 20.0, "cem.topk": 2.0},
                "region_count": {"cem.model_get_cost": 2, "cem.topk": 2},
            },
        ]

        summary = summarize_cem_calls(rows)

        self.assertEqual(summary["call_count"], 2)
        self.assertEqual(summary["cpu_wall_ms"]["mean"], 20.0)
        self.assertEqual(summary["cuda_elapsed_ms"]["p50"], 16.0)
        self.assertEqual(
            summary["region_cpu_ms_total"],
            {"cem.model_get_cost": 26.0, "cem.topk": 3.0},
        )
        self.assertEqual(summary["loop_cpu_materialization_count_estimate"]["mean"], 15.0)
        self.assertEqual(summary["final_cpu_materialization_count_estimate"]["mean"], 3.5)
        self.assertEqual(summary["cpu_materialization_count_estimate"]["mean"], 2.0)
        self.assertEqual(summary["active_env_count"]["max"], 1.0)
        self.assertEqual(summary["region_count"], {"cem.model_get_cost": 4, "cem.topk": 4})
        self.assertEqual(summary["memory"]["cuda_memory_peak_allocated_mb"]["max"], 768.0)
        self.assertEqual(
            summary["compile_diagnostics"],
            {
                "call_count": 2,
                "counter_snapshot_available_count": 1,
                "observation_counts": {
                    "count": 2,
                    "max": 30.0,
                    "mean": 30.0,
                    "min": 30.0,
                    "p50": 30.0,
                    "p90": 30.0,
                    "p99": 30.0,
                },
                "signature_mismatch_count": 1,
                "signature_unknown_count": 0,
                "unique_signature_counts": {
                    "count": 2,
                    "max": 2.0,
                    "mean": 1.5,
                    "min": 1.0,
                    "p50": 1.5,
                    "p90": 1.9,
                    "p99": 1.99,
                },
            },
        )

    def test_evaluation_summary_records_encode_and_call_totals(self) -> None:
        rows = [
            {
                "type": "evaluation_result",
                "evaluation_time_s": 2.0,
                "encode_counts_total": {"total": 10, "goal": 5},
                "plan_calls_total": 7,
                "cem_calls_total": 3,
            },
            {
                "type": "evaluation_result",
                "evaluation_time_s": 4.0,
                "encode_counts_total": {"total": 8, "initial": 4},
                "plan_calls_total": 6,
                "cem_calls_total": 2,
            },
        ]

        summary = summarize_evaluations(rows)

        self.assertEqual(summary["evaluation_time_s"]["mean"], 3.0)
        self.assertEqual(summary["encode_counts_total"], {"total": 18, "goal": 5, "initial": 4})
        self.assertEqual(summary["plan_calls_total"], 13)
        self.assertEqual(summary["cem_calls_total"], 5)

    def test_wall_attribution_splits_cem_policy_and_outside_policy(self) -> None:
        rows = [
            {"type": "plan_latency", "plan_latency_ms": 100.0},
            {"type": "plan_latency", "plan_latency_ms": 50.0},
            {
                "type": "evaluation_result",
                "evaluation_time_s": 1.0,
                "cem_cpu_wall_ms_total": 40.0,
                "cem_cuda_elapsed_ms_total": 35.0,
                "model_cost_evals_total": 120,
                "plan_calls_total": 2,
                "cem_calls_total": 1,
            },
        ]

        summary = summarize_wall_attribution(rows)

        self.assertEqual(summary["evaluation_time_s_total"], 1.0)
        self.assertEqual(summary["plan_latency_ms_total"], 150.0)
        self.assertEqual(summary["cem_cpu_wall_ms_total"], 40.0)
        self.assertEqual(summary["cem_cuda_elapsed_ms_total"], 35.0)
        self.assertEqual(summary["model_cost_evals_total"], 120)
        self.assertEqual(summary["policy_non_cem_ms_total"], 110.0)
        self.assertEqual(summary["outside_policy_ms_total"], 850.0)
        self.assertEqual(summary["cem_cpu_fraction_of_eval_wall"], 0.04)
        self.assertEqual(summary["policy_fraction_of_eval_wall"], 0.15)
        self.assertEqual(summary["outside_policy_fraction_of_eval_wall"], 0.85)
        self.assertTrue(summary["plan_latency_complete"])

    def test_wall_attribution_cli_writes_markdown_report(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO

        from tools.benchmarks.summarize_results import main as summarize_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "metrics.jsonl"
            report_path = root / "wall.md"
            append_jsonl(metrics_path, {"type": "plan_latency", "plan_latency_ms": 100.0})
            append_jsonl(metrics_path, {"type": "plan_latency", "plan_latency_ms": 50.0})
            append_jsonl(
                metrics_path,
                {
                    "type": "evaluation_result",
                    "evaluation_time_s": 1.0,
                    "cem_cpu_wall_ms_total": 40.0,
                    "cem_cuda_elapsed_ms_total": 35.0,
                    "model_cost_evals_total": 120,
                    "plan_calls_total": 2,
                    "cem_calls_total": 1,
                },
            )

            with redirect_stdout(StringIO()) as stdout:
                result = summarize_main(
                    [
                        "--inputs",
                        str(metrics_path),
                        "--wall-attribution",
                        "--out",
                        str(report_path),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertIn("wrote", stdout.getvalue())
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("LeWM Phase 4 Wall Attribution", report)
            self.assertIn("Outside Policy / Eval", report)

    def test_wall_attribution_derives_legacy_cem_model_cost_evals(self) -> None:
        summary = summarize_wall_attribution(
            [
                {
                    "type": "evaluation_result",
                    "evaluation_time_s": 1.0,
                    "model_cost_evals_total": 0,
                },
                {
                    "type": "cem_call",
                    "num_samples": 300,
                    "n_steps": 30,
                },
                {
                    "type": "cem_call",
                    "sample_counts_per_step": [300, 240, 180],
                },
            ]
        )

        self.assertEqual(summary["model_cost_evals_total"], 9720)

    def test_quality_summary_prefers_model_cost_evals_including_gd(self) -> None:
        summary = summarize_quality(
            [
                {
                    "type": "evaluation_result",
                    "metrics": {"success_rate": 1.0, "task_score_mean": 0.9},
                    "model_cost_evals_total": 100,
                    "gd_model_cost_evals_total": 5,
                    "model_cost_evals_total_including_gd": 105,
                }
            ]
        )

        self.assertEqual(summary["model_cost_evals"], 105)

    def test_torch_trace_summary_counts_user_ranges_and_cuda_events(self) -> None:
        summary = summarize_events(
            [
                {"ph": "X", "name": "cem.model_get_cost", "cat": "user_annotation", "dur": 2500},
                {"ph": "X", "name": "lewm.rollout", "cat": "user_annotation", "dur": 1500},
                {"ph": "X", "name": "void kernel()", "cat": "kernel", "dur": 750},
                {"ph": "X", "name": "cudaMemcpyAsync", "cat": "memcpy", "dur": 125},
                {"ph": "X", "name": "cudaLaunchKernel", "cat": "cuda_runtime", "dur": 300},
                {"ph": "M", "name": "metadata", "cat": "__metadata", "dur": 999},
            ]
        )

        self.assertEqual(summary["trace_event_count"], 5)
        self.assertEqual(summary["user_ranges"]["cem.model_get_cost"]["total_ms"], 2.5)
        self.assertEqual(summary["user_ranges"]["lewm.rollout"]["total_ms"], 1.5)
        self.assertEqual(summary["cuda_kernel"]["count"], 1)
        self.assertEqual(summary["cuda_kernel"]["total_ms"], 0.75)
        self.assertEqual(summary["cuda_memcpy"]["total_ms"], 0.125)
        self.assertEqual(summary["cuda_runtime"]["total_ms"], 0.3)

    def test_manifest_write_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.jsonl"
            metrics.write_text("", encoding="utf-8")
            manifest = build_manifest(
                run_id="run_1",
                model="lewm-pusht",
                variant="baseline",
                semantic_class="baseline",
                cwd=root,
                command=["python", "x.py"],
                weights_sha256="a" * 64,
                artifacts={"metrics_jsonl": str(metrics)},
            )
            manifest["environment"]["torch"] = "test"
            manifest["environment"]["gpu_name"] = "test-gpu"
            manifest["base_git_commit"] = "b" * 40
            manifest["git_diff_sha256"] = "c" * 64
            manifest_path = root / "manifest.json"

            write_manifest(manifest_path, manifest)
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            errors = validate_manifest(loaded)

            self.assertEqual(loaded["run_id"], "run_1")
            self.assertEqual(errors, [])

    def test_manifest_local_gpu_relaxation_keeps_speedup_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.jsonl"
            metrics.write_text("", encoding="utf-8")
            manifest = build_manifest(
                run_id="run_1",
                model="lewm-pusht",
                variant="baseline",
                semantic_class="baseline",
                cwd=root,
                command=["python", "x.py"],
                weights_sha256="unknown",
                artifacts={"metrics_jsonl": str(metrics)},
            )
            manifest["environment"]["torch"] = None
            manifest["environment"]["gpu_name"] = None
            manifest["base_git_commit"] = "b" * 40
            manifest["git_diff_sha256"] = "c" * 64

            errors = validate_manifest(manifest, require_gpu_environment=False)

            self.assertIn("missing or unknown field: weights_sha256", errors)
            self.assertNotIn("missing environment.torch", errors)
            self.assertNotIn("missing environment.gpu_name", errors)

    def test_precomputed_hash_uses_staged_artifact_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "artifact_hashes" / "lewm_pusht.json"
            report.parent.mkdir()
            report.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "path": "/weights/datasets/pusht_expert_train.h5",
                                "sha256": "d" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                precomputed_hash(root, Path("/weights/pusht_expert_train.h5")),
                "unknown",
            )
            self.assertEqual(
                precomputed_hash(root, Path("/weights/datasets/pusht_expert_train.h5")),
                "d" * 64,
            )
            self.assertEqual(precomputed_hash(root, Path("/weights/missing.pt")), "unknown")

    def test_dataset_paths_match_stable_worldmodel_cache_layout(self) -> None:
        canonical, legacy = dataset_paths(Path("/weights"), "pusht_expert_train.h5.zst")

        self.assertEqual(canonical, Path("/weights/datasets/pusht_expert_train.h5"))
        self.assertEqual(legacy, Path("/weights/pusht_expert_train.h5"))
        self.assertEqual(
            dataset_candidates(Path("/weights"), "pusht_expert_train")[0],
            canonical,
        )

    def test_stage_config_helpers_handle_hydra_targets(self) -> None:
        section = {
            "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
            "num_frames": 3,
            "input_dim": 192,
        }

        self.assertEqual(hydra_kwargs(section), {"num_frames": 3, "input_dim": 192})
        self.assertEqual(
            norm_target_name(
                {
                    "_target_": "torch.nn.BatchNorm1d",
                    "_partial_": True,
                }
            ),
            "torch.nn.BatchNorm1d",
        )

    def test_load_model_config_accepts_hf_top_level_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "_target_": "stable_worldmodel.wm.lewm.LeWM",
                        "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf"},
                        "predictor": {"_target_": "stable_worldmodel.wm.lewm.module.Predictor"},
                        "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder"},
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_model_config(root)

            self.assertIn("encoder", cfg)
            self.assertIn("predictor", cfg)
            self.assertIn("action_encoder", cfg)

    def test_ledger_upsert_replaces_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.md"
            first = "| L0 | run_1 | model | baseline | baseline | none | ref | mean 1.00 ms | n/a | running | first |"
            second = "| L0 | run_1 | model | baseline | baseline | none | ref | mean 2.00 ms | n/a | accepted | second |"

            upsert_row(ledger, first, "run_1")
            upsert_row(ledger, second, "run_1")
            text = ledger.read_text(encoding="utf-8")

            self.assertIn("mean 2.00 ms", text)
            self.assertNotIn("mean 1.00 ms", text)

    def test_ledger_cli_accepts_experimental_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.jsonl"
            metrics.write_text("", encoding="utf-8")
            manifest = build_manifest(
                run_id="run_l3b",
                model="LeWM Push-T",
                variant="rollout_action_cache",
                semantic_class="exact",
                cwd=root,
                command=["python", "x.py"],
                weights_sha256="a" * 64,
                artifacts={"metrics_jsonl": str(metrics)},
            )
            manifest_path = root / "manifest.json"
            write_manifest(manifest_path, manifest)
            ledger = root / "ledger.md"

            result = optimization_ledger_main(
                [
                    "--ledger",
                    str(ledger),
                    "--manifest",
                    str(manifest_path),
                    "--status",
                    "experimental",
                    "--ladder",
                    "L3b",
                    "--notes",
                    "diagnostic only",
                ]
            )

            text = ledger.read_text(encoding="utf-8")
            self.assertEqual(result, 0)
            self.assertIn("| L3b | run_l3b |", text)
            self.assertIn("| experimental | diagnostic only |", text)


if __name__ == "__main__":
    unittest.main()
