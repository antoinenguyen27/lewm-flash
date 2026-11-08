# Appendix A. Data dictionary

%APPENDIX A
The solution pack's `data` folder holds five comma-separated tables, produced from the raw benchmark records by `lewm_flash.py prepare`.

%TABLE: Columns of the five datasets.
| file | column | meaning |
|---|---|---|
| calls.csv | config | reference, pytorch_optimised, megakernel, or one of four probes (probe_cached_cem, probe_cached_icem, probe_mk_cem, probe_mk_icem) |
| | role | system (compared in the evaluation) or probe (isolates one effect for the profile) |
| | seed, call_index | episode seed (1000 to 1060) and policy call within the episode (1 to 50) |
| | latency_ms | end-to-end latency of the call, CUDA event timers |
| episodes.csv | config, seed, episode | system, evaluation seed (0 to 2) and episode index (0 to 99); the same 300 episodes for every system |
| | success | 1 if the T-block reached the goal pose within the 50-call budget, else 0 |
| cost_scaling.csv | execution, candidates | execution method and number of candidates in one scoring call |
| | p50_ms, n | median latency and number of timed calls |
| components.csv | component, value, unit, source | component timings (encoders, searches, synchronisation primitives) and planner telemetry (encoder calls per plan, share of a planning call spent in the search) |
| parity.csv | check | scoring case (problem, number of candidates, sampling scale), encoder check |
| | pytorch_bf16_rel_max, megakernel_rel_max | maximum relative error against FP32 (relative L2 norm for the encoder row) |
| | pytorch_bf16_top30, megakernel_top30 | number of the FP32 top-30 candidates also in the method's top 30 |

# Appendix B. Solution pack

%APPENDIX B
The solution pack (`solution_pack/` in the project repository, also submitted as an archive) contains the datasets, the solution program and its outputs.

%TABLE: Contents of the solution pack.
| path | content |
|---|---|
| `lewm_flash.py` | the solution program: data preparation, preprocessing, compute-profile analysis, statistical evaluation, tables and figures |
| `data/*.csv` | the five datasets of Appendix A (460 KB) |
| `outputs/results.json`, `outputs/tables.md` | every statistic in this report, as generated |
| `outputs/figs/*.png` | the data figures of this report (Figures 3, 5, 6 and 7) |
| `kernel/` (archive only) | the GPU kernel and its Python runtime, packer and planner class, copied from `megakernel/` |
| `README.md` | how to run |

The program needs Python 3.10 or later with numpy and matplotlib, and runs on any computer. Without arguments it runs `analyse`, which regenerates `outputs/` in about five seconds; `prepare` rebuilds `data/` from the raw records inside the repository; and `reproduce` prints the commands that re-collect the data on a cloud H100.

```
pip install numpy matplotlib
python lewm_flash.py
python lewm_flash.py prepare
python lewm_flash.py reproduce
```

`analyse` performs, in order: loading and validating the datasets; warm-up removal, call classification and outlier flagging; the stage decomposition, scaling-model fits and per-layer costs; bootstrap intervals, speed-up intervals, Wilson intervals, paired McNemar tests, paired bootstrap intervals and the power calculation; and writes the tables, the JSON results and the figures. Re-collecting the raw data (`reproduce --yes`) launches the benchmark on rented H100 GPUs through Modal and requires a Modal account, the LeWM checkpoint and the kernel build; it costs roughly $0.20 per seed per system.

# Appendix C. Configuration and repeatability

%APPENDIX C
%TABLE: Planner configuration of LeWM-MK (identical to the optimised PyTorch planner; the reference uses CEM with 300 candidates in every iteration and no retained elites).
| setting | value |
|---|---|
| candidates per iteration | max(⌈300 · 1.25^−i^⌉, 60), rounded up to a multiple of 60: 300, 240, 240, 180, 180, 120, 120, 120, then 60 for 22 iterations |
| iterations, elites, retained elites | 30, top 30, 5 per iteration |
| rollouts per plan | 2,820 (reference: 9,000) |
| horizon, replanning interval, action block | 5, 5, 5 |
| initial sampling scale | 1.0 |
| coloured noise, momentum, clipping | not used (reduced success on Push-T in the earlier planner study) |

%TABLE: Kernel configuration on the H100.
| item | value |
|---|---|
| grid | 7 clusters of 16 blocks (one robot); 15 clusters of 8 for batched evaluation |
| block | 256 threads (two groups of 128), 255 registers per thread, 231,584 bytes of shared memory |
| weights | 1,472 blocks of 24 KB in BF16, pre-arranged in the tensor-core layout (964 for the predictor, action encoder and projector; 508 for the encoder) |
| random numbers | Philox 4×32-10, keyed by plan seed, iteration, robot, candidate and action element |
| measured primitives | cluster barrier 0.68 µs, grid barrier 1.56 µs, empty kernel launch 14 µs |

The latency benchmark was run twice. Table C3 compares the earlier pass with the pass used in this report for the configurations both contain. The PyTorch configurations, which depend on the CPU for every launch, moved by up to 18%; the configurations running the kernel moved by at most 2.2%.

%TABLE: Median planning latency in two passes of the benchmark (ms).
| configuration | earlier pass | this report | change |
|---|---|---|---|
| reference | 526.9 | 622.4 | +18.1% |
| optimised PyTorch | 196.3 | 182.1 | −7.2% |
| probe: encode once, CEM | 391.4 | 377.0 | −3.7% |
| probe: encode once, iCEM | 410.3 | 410.2 | 0.0% |
| probe: kernel search, CEM | 81.7 | 82.1 | +0.5% |
| probe: kernel search, iCEM | 58.0 | 59.3 | +2.2% |

# Appendix D. Compute and cost

%APPENDIX D
All GPU work ran on rented NVIDIA H100 80GB HBM3 instances on Modal, with kernel builds on CPU instances. The project used about $31.50 of GPU time against an approved budget of $35. Most of it (about $21) went on the two latency passes, because the benchmark harness starts a fresh process for each of the 61 seeds, rebuilding the simulator, reloading the model and, for the compiled baseline, recompiling, so that most of a pass's GPU time is setup rather than measurement. The remaining $10.40 covered kernel development, numerical checks, micro-benchmarks and the 300-episode evaluations. A harness that keeps one process across seeds would cut the cost of a latency pass by an order of magnitude.

# Appendix E. Code and data availability

%APPENDIX E
All code, raw records, datasets and analysis are in the project repository (`lewm-flash`, branch `antoinenguyen27/lewm-megakernel`). The kernel is `megakernel/lewm_mk.cu` with its packer (`pack.py`), runtime (`runtime.py`) and planner class (`solver.py`). The cloud entry points are in `modal/megakernel_app.py`, and the benchmark harness is `tools/benchmarks/lewm_plan_benchmark.py`. The raw per-call records are under `results/runs/`, and the micro-benchmark and numerical records under `results/megakernel/`. The solution program and datasets are in `solution_pack/`, and the report sources in `docs/paper/src/`. The model is LeWM's released Push-T checkpoint, evaluated with stable-worldmodel's `swm/PushT-v1` environment; ThunderKittens is used at commit `67845f5f`.
