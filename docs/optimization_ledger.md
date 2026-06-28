# Optimization Ledger

| Ladder | Run ID | Model | Variant | Semantic Class | Files Changed | Expected Effect | Actual Latency | Quality Result | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | lewm_baseline_seed0-2_l40s | LeWM Push-T | baseline | baseline | none | Reference measurement | CEM solve sum 116.34 s; eval wall 194.72 s; encode total 11100 | 130/150 success = 86.67% | accepted | Authoritative Modal L40S baseline across paired seeds 0, 1, 2. Run IDs: `lewm_baseline_seed0_l40s`, `lewm_baseline_seed1_l40s`, `lewm_baseline_seed2_l40s`. |
| L1 | lewm_cache_embeddings_seed0-2_l40s | LeWM Push-T | cache_embeddings | exact | `lewm/jepa.py` | Remove repeated current/goal encodes | CEM solve sum 64.55 s, 1.80x vs L0; eval wall 138.96 s, 1.40x; encode total 370, 30.0x fewer | 129/150 success = 86.00%; delta -0.67 pp; Push-T non-regression gate pass; exactness check pass | accepted | Exactness check `lewm_cache_exactness_seed0_l40s` passed: max cost diff 0.0, argmin/topk equal, baseline encodes 60 vs cache 2 with zero repeat encodes. Seed 0 has one extra task failure, but fixed-candidate costs and rankings are identical. Run IDs: `lewm_cache_embeddings_seed0_l40s`, `lewm_cache_embeddings_seed1_l40s`, `lewm_cache_embeddings_seed2_l40s`. |
| L2 | lewm_l1_l2_telem100_seed0-2_l40s + lewm_profile_l1_l2_seed0_l40s | LeWM Push-T | cem_exact_no_cpu_sync | exact | `lewm/solvers/cem_exact.py` plus accepted L1 `lewm/jepa.py` path | Remove loop-internal GPU-to-CPU sync | Telemetry sweep: CEM solve sum 187.11 s, 1.16x vs L1 and 1.77x vs baseline; eval wall 366.88 s, 1.12x vs L1 and 1.43x vs baseline; profiler CEM path 1.29x vs L1 | 262/300 success = 87.33%; equals baseline and equals focused L1 profile quality at 17/20 | accepted | Accepted as the best current exact CEM path when paired with L1, with L1-only retained as future control because L2 is synchronization/runtime-sensitive. L2-only remains diagnostic, not product target. |
| L3a | unit_lewm_jepa_rollout_prealloc; smoke_lewm_rollout_prealloc_seed0; lewm_profile_l3_rollout_prealloc_seed0_l40s | LeWM Push-T | rollout_prealloc | exact | `lewm/jepa.py` | Remove growing `cat` and repeated action encoding | Focused profile versus L1+L2: eval wall 141.98 s vs 117.99 s (`0.83x`), CEM CUDA 18.73 s vs 9.40 s (`0.50x`), `lewm.action_encoder` 0.41x but `lewm.predict_step` 2.16x slower | Focused profile matched L1+L2 success sequence exactly: 8/10; smoke success 2/2 | rejected | L3a is behaviorally safe in the focused pair but not performance-safe. The profiler shows the intended action-encoder reduction is overwhelmed by slower predictor/rollout execution, likely from feeding the transformer predictor less favorable static-buffer slice layouts. Do not inherit this variant into later phases. |
| L3b | unit_lewm_jepa_rollout_action_cache_v2; smoke_lewm_rollout_action_cache_v2_seed0; lewm_profile_l1_l2_for_l3b_seed0_l40s + lewm_profile_l3b_rollout_action_cache_seed0_l40s; lewm_l1_l2_current_telem100_seed0-2_l40s + lewm_l3b_current_telem100_seed0-2_l40s | LeWM Push-T | rollout_action_cache | exact | `lewm/jepa.py` | Precompute candidate action embeddings once while keeping dynamic embedding rollout | Focused profile was positive on mechanism but failed strict quality equality; clean 3x100 CUDA-event sweep regressed aggregate CEM CUDA 184.95 s -> 214.27 s and eval wall 360.26 s -> 408.64 s versus L1+L2 | Exactness check passed on synthetic fixed candidates; smoke 2/2; focused profile 8/10 vs L1+L2 9/10; broad sweep 261/300 vs L1+L2 263/300 | experimental | Keep isolated for diagnostics only. Do not make first-class, do not use as default, and do not inherit into the forward ladder. The next accepted ladder branch is L1+L2+L4, not L1+L2+L3b+L4. |
| L4 | unit_lewm_l4_compile_cost_v3; lewm_compile_cost_exactness_seed0_l40s_v3; smoke_lewm_compile_cost_diag_seed0_v2; lewm_l1_l2_for_l4_telem100_seed0-2_l40s + lewm_compile_cost_telem100_seed0-2_l40s | LeWM Push-T | compile_cost | numerical | `lewm/jepa.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `tools/benchmarks/check_lewm_cache_exactness.py`, `modal/benchmark_app.py` | Reduce dispatch overhead in the tensor-only model-cost path | Paired 3x100 warmed-runtime confirmation: post-warm CEM CUDA 22.34 s -> 18.16 s (`1.23x`); single post-warm CEM 7.45 s -> 6.05 s; warmup-excluded eval wall 182.71 s -> 180.75 s (`1.01x`); raw eval wall 285.95 s -> 298.33 s due first-runtime overhead | 264/300 success vs L1+L2 262/300; fixed-candidate exactness pass (`max_abs=6.10e-05`, `max_rel=4.70e-07`, argmin/top-k equal, repeat stable); steady-state compile signatures stable | accepted | Promoted for warmed-runtime branch L1+L2+L4. Requires real full-CEM warmup before serving; not a cold-start speedup. L3 rollout flags stay disabled. |
| L5b | lewm_tensor_cem_exactness_seed0_l40s; lewm_compile_cost_l5_pinned_control_seed0_ep50_l40s + lewm_tensor_cem_nested_pinned_seed0_ep50_l40s + lewm_tensor_cem_no_inner_pinned_seed0_ep50_l40s | LeWM Push-T | tensor_cem_compile; tensor_cem_compile_no_inner_compile | numerical | historical: `lewm/solvers/cem_tensor.py`, `lewm/solvers/cem_exact.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `tools/benchmarks/check_lewm_cache_exactness.py`, `modal/benchmark_app.py` | Compile fixed-shape CEM tensor loop around accepted L4 cost path; isolate nested-compile overhead | Pinned seed-0 closeout: L4 `276.09 ms/output`, nested `279.81 ms/output`, no-inner `277.18 ms/output`; no meaningful post-warm uplift | Pinned closeout quality matched at `46/50` for all three runs | closed | Implemented and tested, but not promoted. L5b repackaged Python/CEM glue into a wider compiled helper, but final synchronization absorbed the savings. Nested compile was not the blocker. Active runtime/CLI/exactness wiring has been removed; keep only as historical diagnostic evidence. |
| L5c | lewm_profile_compile_cost_l5c_b1_control_seed0_l40s + lewm_profile_tensor_cem_static_b1_seed0_l40s | LeWM Push-T | tensor_cem_static | numerical | historical: `lewm/solvers/cem_tensor.py`, `lewm/solvers/cem_exact.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `tools/benchmarks/check_lewm_cache_exactness.py`, `tools/benchmarks/metrics.py`, `tools/benchmarks/profile_lewm_cem.py`, `modal/benchmark_app.py` | Reuse planner-side tensors and defer output materialization at the CEM boundary | Focused profile: L4 post-warm CEM `862.29 ms`; L5c `899.54 ms` (`0.96x`); kernel total unchanged (`3586.37 ms` vs `3584.69 ms`); wait shifted from `cem.cpu_final_cost` into `cem.static_expand_info` + `cem.output_edge_cpu` | Same focused success sequence as L4: `8/10` | closed | Not promoted. L5c did not make the inner CEM/model-cost path cheaper; it deferred synchronization into later staging/output ranges and added workspace copy/noise staging around unchanged candidate evaluation. Active runtime/CLI/exactness wiring has been removed; the earlier active-env batching attempt is invalid for the core ladder and is not acceptance evidence. |
| L6 | lewm_profile_compile_cost_phase3_inner_seed0_l40s | LeWM Push-T | fused_cem_glue | numerical | `lewm/solvers/cem_exact.py`, `tools/benchmarks/lewm_plan_benchmark.py` | Reduce CEM sample/top-k/elite/update glue | Gate failed: glue bucket `20.06 ms`, `3.56%` of post-warm CEM CUDA and `3.44%` of the CEM-bearing plan call | Focused profile success `9/10`; no behavior change tested because no implementation is justified | rejected | Downgraded/no-go before Phase 3R. Do not implement fused CEM glue unless a future accepted-branch profile shows the bucket exceeds the materiality gate by a wide margin. |
| L6R-a | lewm_model_cost_compile_cost_baseline_seed0_l40s; lewm_model_cost_tf32_high_seed0_l40s; lewm_model_cost_tf32_exactness_seed0_l40s_v2; lewm_compile_cost_tf32_control_telem100_l40s + lewm_model_cost_tf32_telem100_l40s | LeWM Push-T | model_cost_tf32 | numerical | `tools/benchmarks/lewm_plan_benchmark.py`, `tools/benchmarks/profile_lewm_model_cost.py`, `tools/benchmarks/check_lewm_cache_exactness.py`, `modal/benchmark_app.py`, `tests/test_benchmark_utils.py` | Use TF32-capable matmul kernels in the repeated model-cost GEMM path | Fixed-candidate model-cost mean `9.930 ms -> 7.623 ms` (`1.303x`); paired 3x100 post-warm CEM CUDA/output `280.27 ms -> 219.50 ms` (`1.28x`); post-warm CEM CUDA total `18.50 s -> 14.93 s` (`1.24x`) | Push-T `263/300` vs control `264/300` (`-0.33 pp`); fixed-candidate ranking drift passed under near-tie rule; compile signatures stable | accepted | Promoted as the Phase 3R warmed-runtime win. Cumulative no-opt -> L1+L2+L4+TF32: post-warm CEM CUDA/output `898.16 ms -> 219.50 ms` (`4.09x`) and eval wall `524.18 s -> 262.76 s` (`1.99x`) with no quality regression. |
| L6R-b | lewm_model_cost_default_seed0_l40s; lewm_model_cost_fullgraph_seed0_l40s; lewm_model_cost_max_autotune_seed0_l40s | LeWM Push-T | compile_default; compile_cost_fullgraph; compile_cost_max_autotune | numerical | `tools/benchmarks/profile_lewm_model_cost.py`, `modal/benchmark_app.py` | Test compile/capture settings for the tensor-only model-cost helper | `compile_default` `0.983x`; `compile_fullgraph` `0.994x`; `compile_max_autotune` `1.006x` only after about `739 s` warmup and Inductor autotune resource warnings | No full Push-T sweep; all failed the isolated model-cost promotion gate | rejected | No further sweep. Fullgraph/default do not improve the accepted helper, and max-autotune has an unacceptable warmup tradeoff for a negligible warmed-runtime gain. |
| L6R-c | lewm_model_cost_score_regions_tf32_high_seed0_l40s | LeWM Push-T | fused_score | numerical | `tools/benchmarks/profile_lewm_model_cost.py`, `modal/benchmark_app.py`, `tests/test_benchmark_utils.py` | Fuse terminal latent MSE score reduction only if material | Diagnostic score attribution: `lewm.cost_from_rollout` mean `0.180 ms`, total `1.801 ms` over 10 measured iterations, `0.692%` of model-cost CUDA versus the `10%` gate | No behavior change tested because no fused implementation is justified | rejected | No fused terminal-score kernel. The bottleneck remains `rollout_from_embeddings()` and its action-encoder/predictor chain, not the final MSE reduction. |
| L7a | unit_phase4_1_cem_warm_start; unit_phase4_1_planner_overrides; lewm_warm_start_rh1_same_budget_seed0_l40s; lewm_warm_start_rh2_same_budget_seed0_l40s | LeWM Push-T | warm_start_cem | planner | `lewm/solvers/cem_exact.py`, `lewm/config/eval/solver/cem_exact.yaml`, `lewm/config/eval/pusht.yaml`, `tools/benchmarks/lewm_plan_benchmark.py`, `tests/test_lewm_cem_exact.py` | Shift previous CEM mean/var across replans at the same 300x30 CEM budget | rh1: eval wall 114.14 s, 10 CEM calls, 90,000 model-cost evals, post-warm CEM CUDA mean 8421.95 ms; rh2: eval wall 73.06 s, 5 CEM calls, 45,000 model-cost evals, post-warm CEM CUDA mean 8897.13 ms | Failed Push-T gate: rh1 18/50 successes (36%); rh2 17/50 successes (34%) versus accepted TF32 baseline 263/300 (87.67%) | rejected | Distribution warm-start state was active (`used` on 5/10 rh1 CEM rows and 2/5 rh2 rows), with resets when active environment batch shape changed. Do not proceed to reduced-budget warm-start sweeps from this branch. |
| L7b | unit_phase4_1b_action_tail_override; lewm_action_tail_only_rh1_same_budget_seed0_l40s; lewm_action_tail_only_rh2_same_budget_seed0_l40s | LeWM Push-T | warm_start_cem | planner | `tools/benchmarks/lewm_plan_benchmark.py`, `modal/benchmark_app.py`, `tests/test_lewm_planner_benchmark_overrides.py` | Isolate action-tail warm-start by setting `plan_config.warm_start=true` and `solver.warm_start=false` at the same 300x30 CEM budget | rh1: eval wall 119.22 s, 10 CEM calls, 90,000 model-cost evals, post-warm CEM CUDA mean 8801.33 ms; rh2: eval wall 71.09 s, 5 CEM calls, 45,000 model-cost evals, post-warm CEM CUDA mean 8417.61 ms | Failed Push-T gate: rh1 18/50 successes (36%); rh2 21/50 successes (42%) versus accepted TF32 baseline 263/300 (87.67%) | rejected | Confirms solver mean/var reuse is not the main failure. RH2 improved without copied variance, but receding-horizon/action-tail semantics still collapse quality. Keep Task 4.2 closed and move to iCEM. |
| L7 | unit_phase4_3_icem_v2; unit_phase4_3_icem_bucket_decay; unit_phase4_3_bucket_planner_overrides; unit_phase4_3_compile_prewarm_overrides; lewm_icem_parity_rh5_s300_i30_seed0_l40s_v2; lewm_icem_rh5_s150_i15_topk15_er5_pinkraw_seed0_l40s; lewm_icem_rh5_s300_i30_topk30_er5_decay125_seed0_l40s; lewm_icem_rh5_s300_i30_topk30_er5_decay125_bucket60_seed0_l40s; lewm_icem_rh5_s300_i30_topk30_er5_decay125_bucket60_nocompile_seed0_l40s; lewm_icem_bucket60_nocompile_seed0-2_l40s; lewm_icem_bucket60_compile_prewarm_seed0_l40s; lewm_icem_bucket60_compile_prewarm_seed0-2_l40s | LeWM Push-T | icem | planner | `lewm/solvers/icem.py`, `lewm/config/eval/solver/icem.yaml`, `lewm/solvers/__init__.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `modal/benchmark_app.py`, `tests/test_lewm_icem.py`, `tests/test_lewm_planner_benchmark_overrides.py` | Reduce samples/iterations with within-solve iCEM features while keeping `horizon=receding_horizon=5`, no cross-MPC solver posterior reuse, and benchmark-side multi-shape compile warmup | Parity v2: eval wall 41.45 s, 18,000 model-cost evals, post-warm iCEM CUDA 3341.31 ms; raw decay: 78.12 s, 5,140 evals, post-warm 1015.17 ms; uncompiled bucket60 3x100: 274.16 s aggregate wall; compiled prewarmed bucket60 3x100: 253.34 s aggregate wall vs accepted TF32 baseline 262.76 s (`1.037x`), recorded model-cost evals 16,920 vs theoretical exact-CEM budget 54,000 (`3.19x` fewer), second iCEM call sum 8.47 s vs 14.93 s (`1.76x`) | Parity v2 passed seed-0 gate at 44/50; aggressive 150x15 failed at 41/50; bucket60 uncompiled passed 3x100 at 268/300; compiled prewarmed bucket60 passed 3x100 at 267/300 versus baseline 263/300 | accepted | Accepted only as a planner-changing warmed-runtime branch. It requires prewarming the bucketed model-cost shapes `[300,240,180,120,60]`; cold-start compile warmup is not free, and episode-wall uplift is modest because non-CEM work now dominates. Keep exact CEM as the exact/numerical control; future MPPI, CEM-GD, or iCEM-GD candidates should beat this branch to become the best planner-changing path. |
| L7c | unit_phase4_3d_icem_step_telemetry; unit_phase4_3d_icem_step_telemetry_overrides; lewm_icem_bucket60_steptelemetry_seed0_l40s; lewm_icem_bucket60_bestelite_seed0_l40s; lewm_icem_bucket60_clip_seed0_l40s; lewm_icem_bucket60_mom01_seed0_l40s; lewm_icem_bucket60_pinkraw_seed0_l40s; lewm_icem_bucket60_pinkblock_seed0_l40s; lewm_icem_bucket60_i25_seed0_l40s; lewm_icem_bucket60_i20_seed0_l40s; lewm_icem_bucket60_i20_seed0-2_l40s; lewm_icem_bucket60_i25_seed0-2_l40s | LeWM Push-T | icem | planner | `lewm/solvers/icem.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `modal/benchmark_app.py`, `tests/test_lewm_icem.py`, `tests/test_lewm_planner_benchmark_overrides.py` | Add per-step iCEM convergence telemetry, test zero-cost upstream toggles, and probe safe `cem_steps` cuts before CEM-GD | Step telemetry showed continued optimization: top-k mean cost improved 32.6% from step 20 to 29 and 16.3% from step 25 to 29. `return_best_elite`: 47/50, 36.33 s, post-warm iCEM CUDA 942.23 ms. `i20` 3x100: 184.77 s, 13,320 evals, post-warm CUDA 7.32 s. `i25` 3x100: 187.80 s, 15,120 evals, post-warm CUDA 8.10 s. Accepted iCEM: 253.34 s, 16,920 evals, post-warm CUDA 8.47 s | `clip_actions`, pink/raw, and pink/block failed seed-0 quality; `momentum_alpha=0.1` preserved quality but regressed wall. Reduced-iteration 3x100 runs passed broad exact-TF32 non-regression but failed accepted-iCEM replacement quality: `i20` 252/300, `i25` 259/300, accepted iCEM 267/300 | diagnostic | Accepted iCEM control remains `lewm_icem_bucket60_compile_prewarm_seed0-2_l40s`. `i20` and `i25` are useful low-budget/iCEM-GD targets, not standalone baselines. `i15` was intentionally not run because telemetry did not plateau by step 15 and `i20` had no quality margin. |
| L7d | docs-only post-analysis decision after `lewm_icem_bucket60_i20_seed0-2_l40s` and `lewm_icem_bucket60_i25_seed0-2_l40s` | LeWM Push-T | icem | planner | `docs/research/lewm_phase4_icem_report.md`, `docs/optimization_ledger.md` | Decide whether to spend more budget on `i26`-`i29` standalone iCEM interpolation | No new run. Existing evidence: `i25` was much faster than accepted iCEM (`187.80 s` vs `253.34 s`) but only slightly cheaper in post-warm CUDA (`8.10 s` vs `8.47 s`); `i20` was similarly fast (`184.77 s`) but lower quality | Do not promote: `i25` missed the accepted-iCEM replacement threshold by two successes (`259/300` vs required `261/300`), and `i20` was further below at `252/300` | closed | Do not spend budget on `i26`-`i29` standalone interpolation now. Reserve at most one `i27` or `i28` 3x100 only if CEM-GD/iCEM-GD fails or a pure-iCEM Pareto fallback is needed. |
| L7e | results/lewm_phase4_wall_attribution.md | LeWM Push-T | wall_attribution | diagnostic | `tools/benchmarks/metrics.py`, `tools/benchmarks/summarize_results.py`, `tests/test_benchmark_utils.py`, `docs/research/lewm_phase4_icem_report.md` | Attribute accepted exact TF32 and accepted iCEM eval wall into CEM, policy non-CEM, and outside-policy buckets | Exact TF32: eval wall 262.76 s, CEM CPU 83.15 s (31.6%), outside policy 161.46 s (61.4%). Accepted iCEM: eval wall 253.34 s, CEM CPU 54.89 s (21.7%), outside policy 177.83 s (70.2%) | No behavior change; artifact-only analysis with complete plan-latency rows for both controls | complete | CEM is below the 0.35 continuation threshold after accepted iCEM. Continue CEM-GD/iCEM-GD as low-budget quality recovery, but profile `world.evaluate` / env stepping / video / dataset overhead before claiming large remaining wall-clock headroom. |
| L7f | unit_phase4_5_cem_gd_v2; lewm_icem_gd_bucket60_i25_g1_seed0_l40s | LeWM Push-T | icem_gd | planner | `lewm/solvers/cem_gd.py`, `lewm/config/eval/solver/cem_gd.yaml`, `lewm/solvers/__init__.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `tools/benchmarks/metrics.py`, `tools/benchmarks/summarize_results.py`, `modal/benchmark_app.py`, `tests/test_lewm_cem_gd.py`, `tests/test_lewm_planner_benchmark_overrides.py`, `tests/test_benchmark_utils.py` | Recover low-budget i25 quality by refining evaluated iCEM elites with eager action-only GD | Seed-0 i25+g1: eval wall `40.04 s`, post-warm solver CUDA `1,702.50 ms`, model-cost evals `5,050` including GD. Standalone i25 seed-0 was `34.33 s`, `843.50 ms`, `5,040`; accepted i30 seed-0 was `34.19 s`, `1,176.04 ms`, `5,640` | Matched seed-0 quality at `46/50`; GD accepted every measured candidate and lowered model cost (`gd_cost_delta_mean=-0.277890`, no NaN/inf) | diagnostic | Not promoted. The first Modal attempt exposed and fixed an inference-tensor autograd boundary. Current iCEM-GD makes the model objective better but wall worse because the GD path adds small-S forward/backward/backtracking evaluations per active batch. Do not run 3x100, `i20_g1`, or `i25_g3` until the GD re-evaluation overhead is addressed. See `docs/research/lewm_phase4_cem_gd_report.md`. |
| L7g | unit_phase4_8_icem_static_compile_v2; lewm_model_cost_tf32_clean_seed0_l40s; lewm_icem_bucket60_clean_seed0_l40s; lewm_model_cost_tf32_intrusive_current_seed0_l40s; lewm_icem_bucket60_intrusive_current_seed0_l40s; lewm_icem_static_chain_eager_seed0_l40s; lewm_icem_static_chain_compile_seed0_l40s | LeWM Push-T | icem_static_compile | planner | `lewm/solvers/icem_static_compile.py`, `lewm/config/eval/solver/icem_static_compile.yaml`, `lewm/solvers/__init__.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `modal/benchmark_app.py`, `tests/test_lewm_icem_static_compile.py`, `tests/test_lewm_planner_benchmark_overrides.py` | Test a static tensor-only iCEM chain and opt-in whole-chain `torch.compile` under clean planner telemetry | Static eager seed-0: eval wall `37.13 s`, manifest plan p50/p90 `27.25/35.37 ms` vs clean iCEM `43.83/54.69 ms`. Static compiled seed-0: eval wall `87.74 s`, compile warmup `812.83 s`, first measured plan `58.21 s`, generated kernels `4358` | Static eager `47/50`, static compiled `47/50`, clean accepted iCEM `46/50`; intrusive current iCEM also `46/50` | diagnostic | Do not promote static compiled iCEM or run its 3x100 promotion. Compile is finite but pathological. Static eager is a possible 3x100 clean follow-up if we want to test whether the p50/p90 win survives seeds 0-2. See `docs/research/lewm_phase4_static_icem_compile_report.md`. |
| L7h | unit_phase4_9_model_cost_autocast; lewm_icem_bucket60_bf16_guard_seed0_l40s; lewm_icem_static_chain_eager_bf16_guard_seed0_l40s; lewm_icem_bucket60_bf16_seed0-2_l40s; lewm_icem_static_chain_eager_seed0-2_l40s; lewm_icem_static_chain_eager_bf16_seed0-2_l40s | LeWM Push-T | model_cost_bf16 | planner | `lewm/jepa.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `modal/benchmark_app.py`, `tests/test_benchmark_utils.py`, `tests/test_lewm_planner_benchmark_overrides.py`, `tests/test_lewm_icem_static_compile.py`, `docs/research/lewm_phase4_precision_autocast_report.md` | Use CUDA BF16 autocast only inside JEPA model-cost rollout while returning planner costs as `float32`; confirm static eager at 3x100 | Accepted iCEM + BF16 3x100: eval wall `253.34 s -> 210.55 s` (`1.20x`), warmup-excluded plan sum `28.37 s -> 23.62 s` (`1.20x`), plan p50/p90 `144.01/182.50 ms -> 116.30/159.32 ms`. Static eager standard reached `250.27 s` but had worse aggregate plan stability; static eager BF16 regressed to `267.92 s` | Accepted iCEM + BF16 passed at `265/300` versus accepted iCEM `267/300` and replacement threshold `261/300`; static eager standard `268/300`; static eager BF16 `266/300` | accepted | Promote `lewm_icem_bucket60_bf16_seed0-2_l40s` as the current best LeWM planner branch. Keep static eager as diagnostic only because plan-index-26 spikes made aggregate plan sum worse despite good normal p50. See `docs/research/lewm_phase4_precision_autocast_report.md`. |

## Telemetry And Capture Boundary - 2026-06-28

L2 and L4 remain accepted under the telemetry audit. L2 is exact CEM cleanup that
removes loop-internal GPU-to-CPU materialization. L4 compiles only
`model.get_cost_from_embeddings()`; CEM telemetry wraps solver calls outside that
compiled helper and does not invalidate L4's fixed-candidate or Push-T evidence.

Benchmark numbers in this ledger are valid for like-for-like benchmark
comparisons when the compared runs use the same telemetry/profiling mode. They
must not be relabeled as whole-planner CUDA graph capture or clean
no-instrumentation serving latency. For graph/capture or serving-latency checks,
run with:

```text
--profile none
--cem-telemetry none
--compile-diagnostics none
--icem-step-telemetry none
```

The iCEM step-telemetry run `lewm_icem_bucket60_steptelemetry_seed0_l40s` is
diagnostic-only. The accepted iCEM timing control remains
`lewm_icem_bucket60_compile_prewarm_seed0-2_l40s`.

Task 4.8 added a current-code telemetry check. Clean exact-CEM and intrusive
exact-CEM seed-0 controls had essentially identical plan p50/p90, but intrusive
iCEM reported lower CUDA-event p50/p90 than clean iCEM. Treat historical iCEM
telemetry-on timings as attribution/like-for-like benchmark data, not clean
whole-planner serving latency. Use `lewm_icem_bucket60_clean_seed0_l40s` as the
seed-0 clean iCEM timing control for future static-chain comparisons.

## L1 Cache Embeddings Authoritative Result - 2026-06-25

Scope:

- Modal GPU: NVIDIA L40S.
- Torch: `2.12.1+cu130`.
- Weights SHA-256: `a7f1ae0cfbfad8aca613f737d66d12220fa2a8e345c5b46de8b89496c44ced62`.
- Dataset SHA-256: `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`.
- Git base: `116b23347cfa966358876320d0a2e1e97a3bbd4e`.
- Git diff hash: `39fd1dbcc27c2853f3d46f75c5f7602800f04c73de49cf8e51802ab89c6bb890`.
- All six run manifests validate with `tools/benchmarks/run_manifest.py validate`.

Measurement note:

The manifest `plan_latency_ms` summary excludes the first two `get_action` calls as warmup. In these batched Push-T runs, the expensive CEM replans occur in the high-latency early/action-buffer-refresh calls, while most later calls are action-buffer returns. The L1 speed claim below therefore uses captured `CEM solve time` lines and the high-latency CUDA-event rows, not the manifest p50 alone.

Per-seed results:

| Seed | Baseline CEM Solve Sum | Cache CEM Solve Sum | CEM Speedup | Baseline Eval Wall | Cache Eval Wall | Eval Speedup | Baseline Success | Cache Success | Baseline Encodes | Cache Encodes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31.71 s | 20.16 s | 1.57x | 59.64 s | 43.18 s | 1.38x | 94.0% | 92.0% | 3660 | 122 |
| 1 | 28.15 s | 21.92 s | 1.28x | 50.09 s | 47.86 s | 1.05x | 86.0% | 86.0% | 3660 | 122 |
| 2 | 56.48 s | 22.48 s | 2.51x | 84.99 s | 47.91 s | 1.77x | 80.0% | 80.0% | 3780 | 126 |
| Total | 116.34 s | 64.55 s | 1.80x | 194.72 s | 138.96 s | 1.40x | 86.67% | 86.00% | 11100 | 370 |

Conclusions:

- L1 hit the intended bottleneck. Encoder calls fell from 11100 to 370 across the paired runs, exactly 30.0x fewer calls, matching the CEM `n_steps: 30` repeated-encode pathology.
- The measured CEM solve budget improved from 116.34 s to 64.55 s across paired seeds, an aggregate 1.80x speedup. Full evaluation wall time improved 1.40x; this includes environment, dataset, video, and action-buffer overhead outside the model-cost path.
- Task success did not regress under the plan's Push-T non-regression gate: cache success was 86.00% versus baseline 86.67%, a -0.67 percentage-point delta. The gate threshold would allow up to a 5-point drop.
- The exactness check `lewm_cache_exactness_seed0_l40s` passed on 2026-06-25 with `samples=300`, `cem_steps=30`, `horizon=5`, and `topk=30`. Baseline/cache fixed-candidate costs were bitwise identical (`max_abs=0.0`), argmin and top-k rankings matched, baseline used 60 encodes, cache used 2 encodes, and a repeat cached pass used 0 additional encodes.
- Seed 0 still had one extra task failure: baseline failed episodes `[20, 45, 49]`, while cache failed `[6, 20, 45, 49]`. Seeds 1 and 2 had identical failure sets between baseline and cache. Given the fixed-candidate exactness result, this is not evidence of a cache-induced cost change, but the episode-level difference remains recorded.

## L2 Exact CEM No CPU Sync Result - 2026-06-25

Scope:

- Modal GPU: NVIDIA L40S.
- Torch: `2.12.1+cu130`.
- Runtime packages recorded on L2 manifests include `stable_worldmodel==0.1.1`, `stable_pretraining==0.1.7`, `hydra_core==1.3.3`, `omegaconf==2.3.1`, `datasets==2.19.2`, and `pyarrow==20.0.0`.
- Weights SHA-256: `a7f1ae0cfbfad8aca613f737d66d12220fa2a8e345c5b46de8b89496c44ced62`.
- Dataset SHA-256: `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`.
- All three L2 run manifests validate with `tools/benchmarks/run_manifest.py validate`.

Per-seed L1-vs-L2 results:

| Seed | L1 CEM Solve Sum | L2 CEM Solve Sum | L2 CEM Multiplier | L1 Eval Wall | L2 Eval Wall | L2 Eval Multiplier | L1 Success | L2 Success | L1 Encodes | L2 Encodes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.16 s | 35.35 s | 0.57x | 43.18 s | 61.13 s | 0.71x | 92.0% | 94.0% | 122 | 122 |
| 1 | 21.92 s | 19.97 s | 1.10x | 47.86 s | 42.41 s | 1.13x | 86.0% | 86.0% | 122 | 122 |
| 2 | 22.48 s | 20.44 s | 1.10x | 47.91 s | 43.00 s | 1.11x | 80.0% | 80.0% | 126 | 126 |
| Total | 64.55 s | 75.76 s | 0.85x | 138.96 s | 146.54 s | 0.95x | 86.00% | 86.67% | 370 | 370 |

Conclusions:

- At this stage, L2 did not pass the latency acceptance gate. Aggregate CEM solve time regressed from 64.55 s to 75.76 s, and aggregate evaluation wall time regressed from 138.96 s to 146.54 s.
- Seeds 1 and 2 show the intended effect: CEM solve time improved by about 9-10%, and evaluation wall time improved by about 10-11%. Seed 0 regressed by 75% on CEM solve time, dominating the aggregate.
- Quality did not regress. L2 success was 130/150 = 86.67%, versus L1 at 129/150 = 86.00%. Failure sets were identical for seeds 1 and 2; seed 0 improved by one episode.
- Encode counts are unchanged from L1 at 370 total, as expected. L2 should only affect CEM bookkeeping, not model encoding.
- A fresh paired seed-0 repeat initially confirmed a slowdown: `lewm_cache_embeddings_l2repeat_seed0_l40s` had CEM solve sum 20.26 s and eval wall 42.38 s, while `lewm_cem_exact_no_cpu_sync_l2repeat_seed0_l40s` had CEM solve sum 36.75 s and eval wall 63.73 s. Success was identical at 92.0%, failure sets were identical, and encode totals were identical at 122.
- A later 100-episode factorial sweep changed the picture. L1+L2 improved aggregate CEM solve time from 188.66 s to 183.16 s (`1.03x`) and eval wall from 366.70 s to 357.24 s (`1.03x`) versus L1, with unchanged encode count. Quality was slightly lower: 262/300 successes versus L1 263/300.
- L2-only without L1 cache is not viable: it regressed to 267.35 s CEM solve time and 444.60 s eval wall, with 22020 encode calls versus 734 for cached variants.
- Historical status at this stage was `needs_repeat`, not `accepted`. The wide sweep showed L1+L2 may be a small positive optimization, but the effect was small and previous 50-episode evidence was unstable. The later telemetry/profiler confirmation below supersedes this status.

## 100-Episode Factorial With Baseline - 2026-06-25

Scope:

- Modal GPU: NVIDIA L40S.
- Torch: `2.12.1+cu130`.
- Weights SHA-256: `a7f1ae0cfbfad8aca613f737d66d12220fa2a8e345c5b46de8b89496c44ced62`.
- Dataset SHA-256: `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`.
- All twelve manifests validate: baseline, L1, L2-only, and L1+L2 across seeds 0, 1, and 2.

Aggregate results:

| Variant | Meaning | CEM Solve Sum | Eval Wall | Success | Encode Total | CEM vs Baseline | Eval vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline` | stock CEM, no cache | 277.74 s | 469.61 s | 262/300 = 87.33% | 22020 | 1.00x | 1.00x |
| `cache_embeddings` | L1 only | 188.66 s | 366.70 s | 263/300 = 87.67% | 734 | 1.47x | 1.28x |
| `cem_exact_no_cpu_sync_only` | L2 only, no cache | 267.35 s | 444.60 s | 262/300 = 87.33% | 22020 | 1.04x | 1.06x |
| `cem_exact_no_cpu_sync` | L1+L2 | 183.16 s | 357.24 s | 262/300 = 87.33% | 734 | 1.52x | 1.31x |

Interpretation:

- L1 is the dominant optimization. It reduces encode calls 30.0x and accounts for most of the wall-clock speedup.
- L2-only is a weak standalone optimization: 1.04x CEM and 1.06x eval wall versus baseline, with no encode reduction.
- L1+L2 is the fastest aggregate in this sweep: 1.52x CEM and 1.31x eval wall versus baseline, and 1.03x versus L1.
- L1+L2 has the same aggregate success as baseline, but one fewer success than L1. The L1+L2 quality delta is small (-0.33 pp versus L1) and within the Push-T non-regression gate. At this stage it argued against accepting L2 without one more confirmation or direct profiler explanation; the later telemetry/profiler confirmation below provides that evidence.

## L2 Telemetry And Torch-Profiler Confirmation - 2026-06-26

Scope:

- Modal GPU: NVIDIA L40S.
- Environment and artifact hashes match the prior LeWM Push-T Modal runs.
- Twelve 100-episode telemetry runs validated: baseline, L1, L2-only, and L1+L2 for seeds 0, 1, and 2.
- Two focused torch-profiler runs validated: `lewm_profile_l1_seed0_l40s` and `lewm_profile_l1_l2_seed0_l40s`.
- The focused L1 profile directory contained one stale interrupted `KeyboardInterrupt` `cem_call` row from a reused run ID. That row is excluded from the analysis below. The run-id hygiene issue is now recorded as an operational lesson and patched in the benchmark runner.

Aggregate telemetry results:

| Variant | Meaning | CEM Solve Sum | Eval Wall | Success | Encode Total | Loop CPU Materializations | CEM vs Baseline | Eval vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | stock CEM, no cache | 331.66 s | 524.18 s | 262/300 = 87.33% | 22020 | 11010 | 1.00x | 1.00x |
| `cache_embeddings` | L1 only | 216.43 s | 410.56 s | 261/300 = 87.00% | 734 | 11010 | 1.53x | 1.28x |
| `cem_exact_no_cpu_sync_only` | L2 only, no cache | 220.73 s | 394.14 s | 261/300 = 87.00% | 22020 | 0 | 1.50x | 1.33x |
| `cem_exact_no_cpu_sync` | L1+L2 | 187.11 s | 366.88 s | 262/300 = 87.33% | 734 | 0 | 1.77x | 1.43x |

L1+L2 versus L1:

| Metric | L1 | L1+L2 | Multiplier |
|---|---:|---:|---:|
| CEM solve sum | 216.43 s | 187.11 s | 1.16x |
| Eval wall | 410.56 s | 366.88 s | 1.12x |
| Success | 261/300 | 262/300 | no regression |
| Encode total | 734 | 734 | unchanged |
| Loop CPU materializations | 11010 | 0 | removed |

Focused torch-profiler results:

| Metric | L1 | L1+L2 | Multiplier |
|---|---:|---:|---:|
| CEM stdout total | 23.68 s | 18.36 s | 1.29x |
| CEM CUDA-event total | 23.68 s | 18.36 s | 1.29x |
| `cem.model_get_cost` profiler range | 23.08 s | 17.85 s | 1.29x |
| `lewm.rollout` profiler range | 43.37 s | 33.41 s | 1.30x |
| CUDA kernel count | 585528 | 585528 | unchanged |
| CUDA runtime calls | 657587 | 656195 | 1392 fewer |
| CUDA memcpy calls | 11304 | 9912 | 1392 fewer |

Conclusions:

- L1 remains the dominant and most deterministic optimization. It removes repeated current/goal encodes and reduces encoder calls by exactly 30.0x in the 100-episode telemetry sweep.
- L2 is accepted when paired with L1. It does not make kernels faster and does not reduce model compute; it removes loop-internal host materialization and changes the synchronization schedule. The profiler confirms the intended structural change: loop materializations fall to zero, runtime/memcpy event counts fall, and the surrounding CEM/model-cost envelope is lower.
- L2 is runtime-sensitive. In seed-level telemetry, seeds 1 and 2 showed small L1+L2 slowdowns versus L1, concentrated inside `cem.model_get_cost`, not inside the new L2 bookkeeping regions. This is recorded as scheduling/synchronization variance rather than algorithmic degradation.
- L2-only is not the target configuration. It is useful diagnostically for isolating solver behavior, but accepted deployment and future exact-cleanup work should use L1+L2 as the current fastest exact variant and keep L1-only as the control.
- Quality did not regress under the plan's gate. L1+L2 matched baseline at 262/300 successes in the telemetry sweep and matched L1 at 17/20 successes in the focused profile.
- Torch profiler is not used for broad timing sweeps because it changes runtime characteristics and creates large artifacts. It is required for deep mechanism claims, while CUDA-event telemetry remains the benchmark timing path.
- Reusing a run ID previously allowed stale JSONL rows to contaminate `metrics.jsonl`; the benchmark runner now clears benchmark-owned artifacts before starting a run.

## L3 Rollout Cleanup Findings - 2026-06-26

Scope:

- Focused torch-profiler comparison:
  - Control: `lewm_profile_l1_l2_seed0_l40s`
  - Candidate: `lewm_profile_l3_rollout_prealloc_seed0_l40s`
- Both runs used `--solver cem_exact`, `--profile torch`, CEM CUDA-event telemetry, seed `0`, and `10` Push-T episodes.
- Both runs used the same base commit, diff hash, weights hash, and dataset hash.
- The L3a candidate manifest confirmed `embedding_cache_enabled=true`, `rollout_prealloc_enabled=true`, and solver `cem_exact`.

Focused profile result:

| Metric | L1+L2 | L3a `rollout_prealloc` | L3a / L1+L2 |
|---|---:|---:|---:|
| Eval wall | 117.99 s | 141.98 s | 0.83x |
| CEM CPU wall | 9.40 s | 18.74 s | 0.50x |
| CEM CUDA elapsed | 9.40 s | 18.73 s | 0.50x |
| Non-warmup recorded CEM | 1.77 s | 3.16 s | 0.56x |
| Plan p50 | 13.38 ms | 17.44 ms | 0.77x |
| Plan p90 | 16.24 ms | 27.61 ms | 0.59x |
| Success | 8/10 | 8/10 | no change |

Mechanism profile:

| Profiler Range | L1+L2 | L3a | Interpretation |
|---|---:|---:|---|
| `lewm.action_encoder` calls | 3926 | 806 | Intended call reduction happened. |
| `lewm.action_encoder` total | 995.94 ms | 412.14 ms | Action encoding got cheaper. |
| `lewm.predict_step` total | 15.34 s | 33.19 s | Predictor path became much slower. |
| `lewm.rollout` total | 17.04 s | 34.64 s | Rollout envelope doubled. |
| CUDA kernel count | 317161 | 305071 | Kernel count did not explain the slowdown. |
| CUDA kernel total | 3.90 s | 3.84 s | GPU math did not increase. |
| CUDA runtime total | 1.26 s | 2.69 s | Runtime/dispatch overhead increased sharply. |

Conclusion:

- L3a is not accepted. It is exact enough for the focused behavioral check, but it is not a useful optimization in its current form.
- The useful part of L3a is the action-embedding reuse. `lewm.module.Embedder` uses `Conv1d(kernel_size=1)` plus a per-timestep MLP, so full-sequence action encoding is mathematically compatible with prefix reuse.
- The harmful part is the static embedding-buffer rollout. The profiler points to slower predictor execution rather than action encoding or CEM bookkeeping.
- L3b `rollout_action_cache` isolates the useful mechanism: precompute action embeddings once while retaining the dynamic embedding rollout. It remains experimental only after the follow-up evidence below.

## L3b Action-Cache Diagnostic - 2026-06-26

Scope:

- Unit/smoke verification:
  - `unit_lewm_jepa_rollout_action_cache_v2`
  - `smoke_lewm_rollout_action_cache_v2_seed0`
- Fixed-candidate exactness:
  - `lewm_rollout_action_cache_exactness_seed0_l40s`
- Focused torch-profiler pair:
  - Control: `lewm_profile_l1_l2_for_l3b_seed0_l40s`
  - Candidate: `lewm_profile_l3b_rollout_action_cache_seed0_l40s`
- Current-source 100-episode CUDA-event sweep:
  - Control: `lewm_l1_l2_current_telem100_seed0_l40s`, `lewm_l1_l2_current_telem100_seed1_l40s`, `lewm_l1_l2_current_telem100_seed2_l40s`
  - Candidate: `lewm_l3b_current_telem100_seed0_l40s`, `lewm_l3b_current_telem100_seed1_l40s`, `lewm_l3b_current_telem100_seed2_l40s`

Focused profile:

| Metric | L1+L2 | L3b `rollout_action_cache` | L3b / L1+L2 |
|---|---:|---:|---:|
| Eval wall | 149.85 s | 128.47 s | 1.17x |
| CEM CUDA elapsed | 18.70 s | 11.39 s | 1.64x |
| Non-warmup CEM CUDA | 2.49 s | 1.96 s | 1.27x |
| Plan p50 | 22.48 ms | 17.81 ms | 1.26x |
| Plan p90 | 51.34 ms | 27.41 ms | 1.87x |
| Success | 9/10 | 8/10 | -1 episode |

Focused mechanism profile:

| Profiler Range | L1+L2 | L3b | Interpretation |
|---|---:|---:|---|
| `lewm.action_encoder` calls | 3624 | 806 | Intended action-encoder reduction happened. |
| `lewm.action_encoder` total | 1.74 s | 0.34 s | Action encoding got cheaper. |
| `lewm.predict_step` total | 31.47 s | 19.19 s | Candidate profile looked better in this pair. |
| `lewm.rollout` total | 34.57 s | 20.29 s | Candidate profile looked better in this pair. |
| `lewm.model_get_cost` total | 18.22 s | 11.00 s | Candidate profile looked better in this pair. |

Clean current-source CUDA-event sweep:

| Metric | L1+L2 | L3b | L3b / L1+L2 |
|---|---:|---:|---:|
| Eval wall, 3x100 episodes | 360.26 s | 408.64 s | 0.88x |
| CEM CUDA total | 184.95 s | 214.27 s | 0.86x |
| `cem.model_get_cost` total | 179.12 s | 209.05 s | 0.86x |
| Success | 263/300 | 261/300 | -0.67 pp |

Per-seed CEM CUDA:

| Seed | L1+L2 | L3b | Interpretation |
|---:|---:|---:|---|
| 0 | 40.06 s | 74.63 s | L1+L2 control was anomalously fast. |
| 1 | 73.33 s | 72.19 s | L3b was approximately neutral/slightly faster. |
| 2 | 71.56 s | 67.45 s | L3b was modestly faster. |

Conclusion:

- L3b is not accepted. The implementation is useful as a diagnostic branch, but the effect size is small relative to Modal/GPU runtime variance and it did not pass the broad timing or quality gate.
- The broad negative result is dominated by an anomalously fast seed-0 L1+L2 control run, but that does not rescue L3b: first-class optimizations must be robust enough to beat the accepted path without special pleading.
- The action encoder is a small `Conv1d(kernel_size=1)` plus MLP. Reusing its outputs is mathematically reasonable, but the predictor/CEM dispatch envelope dominates this workload.
- The exactness check passed on synthetic fixed candidates, but the runtime benchmark starts from shorter action history shapes. That gap is recorded as a reason not to elevate L3b without further targeted work.
- Future LeWM exact/numerical work skips L3b. L4 starts from accepted L1+L2 (`cem_exact_no_cpu_sync`) and should be recorded as L1+L2+L4, not as L1+L2+L3b+L4.

## L4 Compile Cost Confirmation - 2026-06-26

Scope:

- Implementation/unit run: `unit_lewm_l4_compile_cost_v3`
- Fixed-candidate exactness: `lewm_compile_cost_exactness_seed0_l40s_v3`
- Smoke run: `smoke_lewm_compile_cost_diag_seed0_v2`
- Paired confirmation control:
  - `lewm_l1_l2_for_l4_telem100_seed0_l40s`
  - `lewm_l1_l2_for_l4_telem100_seed1_l40s`
  - `lewm_l1_l2_for_l4_telem100_seed2_l40s`
- Paired confirmation candidate:
  - `lewm_compile_cost_telem100_seed0_l40s`
  - `lewm_compile_cost_telem100_seed1_l40s`
  - `lewm_compile_cost_telem100_seed2_l40s`
- All six paired confirmation manifests validate. They use the same L40S
  runtime, torch `2.12.1+cu130`, CUDA `13.0`, weights hash
  `a7f1ae0cfbfad8aca613f737d66d12220fa2a8e345c5b46de8b89496c44ced62`,
  dataset hash
  `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`,
  `samples=300`, `cem_steps=30`, `horizon=5`, and `topk=30`.

Branch contract:

- L4 starts from accepted L1+L2 (`cem_exact_no_cpu_sync`).
- `compile_cost` enables the accepted embedding cache plus the compiled
  tensor-only cost helper.
- L3a/L3b remain disabled. Smoke `model_info` records
  `rollout_prealloc_enabled=false` and `rollout_action_cache_enabled=false`.

Verification:

| Check | Result |
|---|---|
| Modal unit | `Ran 14 tests in 0.745s OK` |
| Fixed-candidate exactness | pass |
| Control vs candidate max absolute diff | `6.103515625e-05` |
| Control vs candidate max relative diff | `4.704991172260313e-07` |
| Argmin/top-k ranking equality | true / true |
| Candidate repeat stability | `max_abs=0.0`, allclose true |
| Smoke quality | `2/2` success |
| Smoke manifest | valid |

Smoke timing notes:

| Metric | Value |
|---|---:|
| Compile warmup | `24.76 s` |
| Warmup CEM CUDA | `2.70 s` |
| Warmup `cem.model_get_cost` | `2.12 s` |
| Later measured CEM CUDA | `279.04 ms` |
| Later measured `cem.model_get_cost` | `14.81 ms` |
| Measured signature mismatch count | `0` |
| Measured compile-counter delta | `{}` |

Paired confirmation:

| Metric | L1+L2 control | L1+L2+L4 | Result |
|---|---:|---:|---:|
| Success | `262/300` | `264/300` | no regression |
| Raw eval wall | `285.95 s` | `298.33 s` | `0.96x` |
| Warmup-excluded eval wall | `182.71 s` | `180.75 s` | `1.01x` |
| All CEM CUDA, including first runtime CEM | `123.83 s` | `134.49 s` | `0.92x` |
| Post-warm measured CEM CUDA | `22.34 s` | `18.16 s` | `1.23x` |
| Single post-warm CEM | `7.45 s` | `6.05 s` | `1.23x` |

Baseline context:

| Branch | Single post-warm CEM | Multiplier vs baseline |
|---|---:|---:|
| Baseline, no optimizations | `20.06 s` | `1.00x` |
| L1+L2+L4 | `6.05 s` | `3.31x` |

Mechanism:

| Region | L1+L2 | L1+L2+L4 | Interpretation |
|---|---:|---:|---|
| `cem.model_get_cost` | `21.53 s` | `0.88 s` | Python/dispatch boundary is much cheaper. |
| `cem.cpu_final_cost` | `0.26 s` | `16.74 s` | Queued GPU work is paid at final CPU materialization. |
| Combined | `21.79 s` | `17.62 s` | Effective region speedup is about `1.24x`. |

Conclusion:

- L4 is implemented and behaviorally sane on fixed candidates, with only small
  floating-point drift expected from a numerical compile path.
- L4 is promoted as the warmed-runtime branch. The accepted forward path is
  L1+L2+L4 (`compile_cost`), not any L3 branch.
- L4 is not a cold-start speedup. The first helper compile and first real CEM
  call pay compiler/capture/cache overhead; deployment claims assume a real
  full-CEM warmup before serving.
- The honest speed claim is post-warm CEM CUDA: `1.23x` over paired L1+L2, and
  `3.31x` single post-warm CEM versus the latest no-optimization baseline.

## L5 Tensor-CEM Closeout And Bottleneck Lessons - 2026-06-27

Scope:

- Accepted forward control: L1+L2+L4, variant `compile_cost`.
- L5b historical control: `lewm_compile_cost_l5_no_inner_control_seed0_ep50_l40s`.
- L5b historical nested candidate: `lewm_tensor_cem_nested_seed0_ep50_l40s`.
- L5b historical no-inner candidate: `lewm_tensor_cem_no_inner_seed0_ep50_l40s`.
- L5b pinned control: `lewm_compile_cost_l5_pinned_control_seed0_ep50_l40s`.
- L5b pinned nested candidate: `lewm_tensor_cem_nested_pinned_seed0_ep50_l40s`.
- L5b pinned no-inner candidate: `lewm_tensor_cem_no_inner_pinned_seed0_ep50_l40s`.
- L5c focused control: `lewm_profile_compile_cost_l5c_b1_control_seed0_l40s`.
- L5c focused candidate: `lewm_profile_tensor_cem_static_b1_seed0_l40s`.
- The historical L5b seed-0 closeout manifests validate and share base commit
  `8a943e89d47d803897dba0ac3129e0ca1adca83c`, diff hash
  `5671b6bd4c661ad0a4175e542747145408eb4ce82fe2501f1e875c8f52518aa3`,
  weights hash `a7f1ae0cfbfad8aca613f737d66d12220fa2a8e345c5b46de8b89496c44ced62`,
  dataset hash `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`,
  seed `0`, `50` episodes, `cem_exact`, CUDA-event CEM telemetry, and compile
  diagnostics.
- The pinned L5b closeout and focused L5c profile use the later pinned-resource
  contract and are recorded as mechanism evidence, not acceptance sweep evidence.

L5b branch flags:

| Branch | Variant | L4 Cost Compile | Tensor CEM Compile | Inner Cost Compile |
|---|---|---:|---:|---:|
| Control | `compile_cost` | true | false | n/a |
| Nested L5b | `tensor_cem_compile` | true | true | true |
| No-inner L5b | `tensor_cem_compile_no_inner_compile` | false | true | false |

L5b post-warm timing:

| Metric | L4 Control | Nested L5b | No-inner L5b |
|---|---:|---:|---:|
| Eval wall | `43.51 s` | `91.13 s` | `114.34 s` |
| Warm CEM CUDA | `15.83 s` | `55.75 s` | `79.00 s` |
| Post-warm CEM CUDA | `3.0307 s` | `3.0588 s` | `3.0230 s` |
| Post-warm ms/output | `275.52 ms` | `278.07 ms` | `274.82 ms` |
| Success | `46/50` | `46/50` | `45/50` |

Mechanism:

| Region | L4 Control | Nested L5b | No-inner L5b |
|---|---:|---:|---:|
| Visible non-final CEM regions | `216 ms` | `70 ms` | `77 ms` |
| `cem.cpu_final_cost` | `2789 ms` | `2987 ms` | `2944 ms` |

Conclusions:

- L5b is implemented and behaviorally close enough for diagnostics, but it is not
  promoted. Nested L5b is slightly slower than L4 post-warm, and no-inner L5b is
  only `1.0025x` faster by post-warm CEM CUDA while recording one extra failed
  episode.
- The nested-compile hypothesis is not supported as the main blocker. Removing
  the separate inner `get_cost_from_embeddings` compile did not materially
  change the captured graph or produce a useful speedup.
- L5b reduces visible Python/CEM glue, but the final CPU materialization point
  absorbs more queued GPU work. The result is consistent with work being
  repackaged rather than removed.
- A pinned-resource closeout later confirmed the same classification:
  L4 `276.09 ms/output`, nested L5b `279.81 ms/output`, no-inner L5b
  `277.18 ms/output`, with quality `46/50` for all three runs.
- L5b is closed as not promoted. Active runtime, CLI, exactness, and Modal
  wiring has been removed; the labels remain only as historical diagnostics.

L5c focused timing:

| Metric | L4 Control | L5c `tensor_cem_static` | Interpretation |
|---|---:|---:|---|
| Post-warm CEM CUDA | `862.29 ms` | `899.54 ms` | L5c is `0.96x`, about `4.3%` slower. |
| Focused success | `8/10` | `8/10` | No behavioral signal in the focused pair. |
| CUDA kernel count | `165060` | `165073` | Kernel work count is effectively unchanged. |
| CUDA kernel total | `3586.37 ms` | `3584.69 ms` | GPU math work is effectively unchanged. |
| CUDA memcpy total | `3050.85 ms` | `488.84 ms` | Transfer/wait attribution moved. |
| CUDA runtime total | `335.03 ms` | `3044.71 ms` | Runtime/API wait attribution grew. |

The focused CEM rows explain the slowdown:

| Region | L4 Control | L5c | Interpretation |
|---|---:|---:|---|
| `cem.cpu_final_cost` | `704.57 ms` | n/a | L4 waits when each batch cost is materialized. |
| `cem.static_expand_info` | n/a | `498.72 ms` | L5c defers prior queued GPU work into the next staging range. |
| `cem.output_edge_cpu` | n/a | `241.83 ms` | L5c still pays an output-edge synchronization. |
| Combined wait-like ranges | `704.57 ms` | `740.55 ms` | The shifted wait grows by about `36 ms`, matching the `37 ms` total slowdown. |
| `cem.model_get_cost` / `cem.static_tensor_cem_solve` visible range | `112.99 ms` | `91.10 ms` | Visible pre-sync work shrinks, but the saved time reappears at synchronization points. |

Code-level cause:

- L4 synchronizes per solver batch when it executes
  `final_batch_cost_gpu.detach().cpu().tolist()` in
  `lewm/solvers/cem_exact.py`. The next batch starts after that wait.
- The historical L5c implementation deferred cost/action/variance materialization
  to one output edge, but the next staging operation still called
  `expand_info_batch(...).to(device=...)`. That later staging range can absorb
  queued CUDA work from the prior chunk.
- L5c adds outer workspace staging (`chunk_mean.copy_`, `chunk_var.copy_`,
  `final_costs.copy_`) and prefilled noise storage, but it does not preallocate
  or fuse the inner CEM tensors created every refinement step: `sampled`,
  `candidates`, `costs`, `topk_vals`, `topk_inds`, and `topk_candidates`.
- The inner model-cost call remains the same repeated LeWM path:
  candidate actions go through `get_cost_from_embeddings()`,
  `rollout_from_embeddings()`, action encoding, prediction, and final cost
  reduction.

L5 final decision:

- L5b and L5c are closed as not promoted. They were useful diagnostics, but the
  current implementations mostly repackage CEM runtime boundaries rather than
  reducing the repeated candidate-cost workload.
- Do not run broad L5 acceptance sweeps. The forward LeWM branch remains
  L1+L2+L4 (`compile_cost`). Reopening this line requires new active code and a
  same-stack control.
- Avoid overgeneralizing this result: a future tensor planner or custom kernel
  can still be useful if it reduces or fuses actual inner-loop work. The rejected
  claim is narrower: the tested L5 compile/static-boundary changes did not clear
  the speed gate.

What we learned about the bottleneck:

- Accepted cleanup through L4 removed repeated encodes, loop-internal CPU sync,
  and a real cost-helper dispatch boundary.
- The remaining high-value target is repeated LeWM model-cost evaluation inside
  CEM: candidate rollout through action encoding and predictor calls, repeated
  across CEM refinement steps.
- CEM glue is present but secondary in the latest profiles:
  `cem.sample_candidates`, `cem.topk`, and `cem.elite_stats` are measurable, but
  not yet proven large enough to justify kernel work before an inner-path profile.
- Low-yield areas are now clear: output-edge cleanup, static outer workspaces,
  and wider compile wrappers around the same candidate-cost work.
- Sensitive areas are also clear: synchronization placement can move apparent
  time between profiler ranges, and torch-profiler user ranges can hide queued
  GPU work until a later CPU/CUDA boundary. Use total post-warm CEM CUDA plus
  kernel/runtime/memcpy summaries for mechanism claims.

Bridge to Phase 3R:

- Phase 3R started from L1+L2+L4, not from L5.
- Phase 3R promoted `model_cost_tf32` and rejected fused CEM glue,
  compile/capture variants, and fused terminal scoring.
- Same-budget Phase 4 warm-start controls are now rejected. Shifted CEM
  mean/variance and action-tail-only receding-horizon controls both failed
  Push-T quality before any budget reduction.
- The next higher-value work is planner/model work that reduces model-cost
  evaluations without this failed receding-horizon warm-start branch: iCEM,
  elite retention, CEM-GD, MPPI, or Fast-LeWM. See
  `docs/research/lewm_phase3r_model_cost_report.md` and
  `docs/research/lewm_phase4_warm_start_controls_report.md`.
