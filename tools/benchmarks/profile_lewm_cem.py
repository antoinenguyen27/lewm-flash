"""Run a short LeWM profile with benchmark-compatible output paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks import lewm_plan_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--solver", default="cem")
    parser.add_argument("--plans", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cem-telemetry", choices=["summary", "cuda-events"], default="cuda-events")
    parser.add_argument("--torch-profiler-with-stack", action="store_true")
    parser.add_argument("--compile-cost", choices=["true", "false"], default=None)
    parser.add_argument("--compile-diagnostics", choices=["none", "summary"], default=None)
    parser.add_argument("--compile-warmup-history", type=int, default=None)
    parser.add_argument("--max-compile-signature-records", type=int, default=None)
    parser.add_argument("--torch-compile-logs", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or f"lewm_profile_{args.variant}_seed{args.seed}"
    forwarded = [
        "--policy",
        args.policy,
        "--solver",
        args.solver,
        "--variant",
        args.variant,
        "--seed",
        str(args.seed),
        "--run-id",
        run_id,
        "--results-root",
        args.results_root,
        "--profile",
        "torch",
        "--cem-telemetry",
        args.cem_telemetry,
        "--max-recorded-plans",
        str(args.plans),
        "--max-recorded-cem-calls",
        str(args.plans),
    ]
    if args.torch_profiler_with_stack:
        forwarded.append("--torch-profiler-with-stack")
    if args.compile_cost is not None:
        forwarded.extend(["--compile-cost", args.compile_cost])
    if args.compile_diagnostics is not None:
        forwarded.extend(["--compile-diagnostics", args.compile_diagnostics])
    if args.compile_warmup_history is not None:
        forwarded.extend(["--compile-warmup-history", str(args.compile_warmup_history)])
    if args.max_compile_signature_records is not None:
        forwarded.extend(
            ["--max-compile-signature-records", str(args.max_compile_signature_records)]
        )
    if args.torch_compile_logs:
        forwarded.extend(["--torch-compile-logs", args.torch_compile_logs])
    if args.episodes is not None:
        forwarded.extend(["--episodes", str(args.episodes)])
    if args.weights_root:
        forwarded.extend(["--weights-root", args.weights_root])
    if args.out:
        forwarded.extend(["--out", args.out])
    return lewm_plan_benchmark.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
