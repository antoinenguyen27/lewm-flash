# 3. Results

We report the ladder in the order of its rungs, because each rung sets the terms of the next. Section 3.1 gives the end-to-end result, Sections 3.2 to 3.5 the contribution of each step and its mechanism, Section 3.6 the paired closed-loop evidence that no step changed task success, Section 3.7 the numerical agreement of the megakernel with the reference, and Section 3.8 how the kernel reached its final latency and what did not work. Every latency in this chapter is a batch-1 end-to-end plan latency on the same H100 in the same container image unless it is labelled otherwise, and every success rate is on the same 300 paired Push-T episodes.

## 3.1 The ladder end to end

Table 4 and Figure 4 give the result. The reference LeWM planner takes 622 ms to produce a plan for one robot on an H100. The final rung, one megakernel launch from the raw observation to the action distribution, takes 46.5 ms, 13.4 times less, with a p90 of 47.0 ms. The best PyTorch implementation, LeWM-Flash, takes 182 ms; the megakernel rungs are 3.1 and 3.9 times faster than it. Task success is 260 to 265 of 300 on every rung.

%TABLE: Batch-1 end-to-end plan latency on one H100 (final pass; 72 to 76 plans per rung, first seed dropped) and Push-T success on 300 paired episodes. Step gain is the ratio of the medians of a rung and the rung above it; R2b is compared with R2, R3 with R2, R3-CEM with R1.
| rung | plan p50 (ms) [95% CI] | p90 (ms) | step call p50 (ms) | step gain | total vs R0 | success (Wilson 95%) |
|---|---|---|---|---|---|---|
| R0 reference LeWM (CEM, FP32 eager) | 622.4 [605.5, 626.7] | 673.8 | 2.14 | — | 1.0× | 261/300 (82.7–90.3%) |
| R1 + cached embeddings, no host sync | 377.0 [375.8, 381.0] | 401.0 | 2.02 | 1.65× | 1.65× | 263/300 (83.5–90.9%) |
| R2 + iCEM, FP32 eager | 410.2 [406.2, 416.3] | 433.3 | 2.11 | 0.92× | 1.52× | 265/300 (84.2–91.5%) |
| R2b LeWM-Flash (compile, TF32, BF16) | 182.1 [177.3, 185.7] | 204.3 | 1.51 | 2.25× | 3.42× | 261/300 (82.7–90.3%) |
| R3 + megakernel (predictor + iCEM) | 59.3 [59.1, 59.4] | 61.1 | 2.35 | 6.92× | 10.5× | 262/300 (83.1–90.6%) |
| **R4 + fused encoder, one launch** | **46.5 [46.4, 46.6]** | **47.0** | **0.82** | **1.28×** | **13.4×** | 260/300 (82.4–90.1%) |
| R3-CEM megakernel with CEM 300 × 30 | 82.1 [81.7, 82.5] | 83.6 | 1.59 | 4.59× | 7.58× | 261/300 (82.7–90.3%) |

![Batch-1 end-to-end plan latency per rung on one H100, log scale. Small points are individual plans (72 to 76 per rung); large markers are medians, also given at the right. Orange rungs are PyTorch, blue rungs are the megakernel.](figs/fig_ladder.png)

The shape of the distributions matters as much as their medians for a controller. The PyTorch rungs spread over tens to hundreds of milliseconds: R0 from 556 to 759 ms, R2b from 112 to 259 ms with two distinct clusters. The megakernel rungs are tight: R3 spans 54 to 70 ms, and R4's interquartile range is 46.1 to 46.8 ms. R4 has one outlier, a single plan of 458 ms among 76, which raises its mean to 51.8 ms; it is a one-off in its seed's process and we did not isolate its cause. The step calls, which only pop a buffered action, fall from about 2 ms to 0.82 ms at R4 because the fused path no longer runs the policy's CPU image transform on every call.

## 3.2 Caching and synchronisation: 1.65×, exact

Encoding the observation and goal once per plan instead of in every one of the 30 cost calls, and keeping the CEM loop on the GPU, cuts the plan from 622 to 377 ms. The change is exact: on fixed candidates it reproduces the reference costs and rankings with zero difference (LeWM-Flash exactness check). On the 300 paired episodes it differs from the reference in two episodes, both successes for R1 (263 against 261; McNemar p = 0.50); because the costs are identical on fixed candidates, the two must come from inputs that check does not fix. The most plausible source is floating-point differences of the order of 10⁻⁷ between the two code paths' encoder and cost calls, whose tensors have different shapes and layouts, which can flip a near-tie in a ranking and send a sensitive episode down a different path.

The remaining 377 ms is almost entirely the rollout: 30 sequential cost calls, each a 5-step autoregressive pass through the predictor for 300 candidates, issued one PyTorch operator at a time.

## 3.3 iCEM: no gain in eager PyTorch, 1.38× in the megakernel

iCEM evaluates 3.19 times fewer candidate rollouts per plan than CEM. In eager PyTorch it makes the plan slower, 410 ms against 377 (0.92×), with success unchanged (265 against 263 of 300; 17 against 15 discordant episodes, p = 0.86). Figure 5 shows why. A single eager FP32 cost call takes 20 ms whether it evaluates 60 candidates or 300: the call is a sequence of a few hundred small kernels, each launched from Python, and at these sizes the GPU finishes each kernel before the next is issued, so the time is set by the number of launches and not by the number of candidates. Reducing the population therefore buys nothing, and iCEM's small extra work per iteration (retaining elites, a variable population) costs a little. In this micro-benchmark the eager solve takes 601 ms, 30 calls of about 20 ms; the same path inside the benchmark harness is faster in absolute terms (Table 4), a difference we did not isolate, but it is equally flat in the population.

![Latency of one cost call (a 5-step rollout of every candidate through the predictor) against the number of candidates, on one H100. Eager FP32 PyTorch is flat at about 20 ms; LeWM-Flash's compiled BF16 path grows from 2.5 to 4.9 ms; the megakernel, timed here as a launch per cost call, from 1.4 to 2.3 ms. Micro-benchmark with cached embeddings; medians of 30 to 50 calls.](figs/fig_cost_scaling.png)

Once execution stops being launch-bound, the population size matters. LeWM-Flash's compiled cost call grows from 2.5 ms at 60 candidates to 4.9 ms at 300, and the megakernel's from 1.4 to 2.3 ms. The clean measure of iCEM's value is the side rung R3-CEM, the same megakernel running the reference CEM schedule: 82.1 ms against R3's 59.3 ms, a 1.38 times gain, with success unchanged (262 against 261; 20 against 19 discordant, p = 1.00). The gain is smaller than the 3.19 times reduction in evaluations because the predictor chain of a plan has 30 iterations regardless of population, and at 60 candidates the 64-row tiles of the kernel are almost empty: the cost per iteration falls by less than the population does. iCEM is a planner-level change whose payoff depends on the execution model beneath it, which is the first finding of the ladder.

## 3.4 The megakernel: 6.9× over eager iCEM, 3.1× over LeWM-Flash

R3 runs exactly R2's planner, with the same population schedule, elites and horizon, as one kernel launch per plan, and takes 59.3 ms against R2's 410 ms (6.92×) and LeWM-Flash's 182 ms (3.07×). Its single-environment solve, the part of the plan inside the launch, takes 41.3 ms in isolation, against 82.3 ms for LeWM-Flash's compiled BF16 solve and 601 ms for eager FP32 in the same micro-benchmark. The remaining 18 ms of R3's plan is outside the kernel (Figure 6): the PyTorch FP32 encoder for the observation and goal (6.9 ms) and the policy's CPU image transform and glue (about 11 ms by difference).

The megakernel's advantage over LeWM-Flash is not arithmetic. Both use BF16 tensor cores with FP32 accumulation, and both do the same 2,820 rollouts. What the megakernel removes is everything between the arithmetic: the 30 host round trips of the iCEM loop, the per-operator launches and the graph replay of the compiled path, the materialisation of every intermediate activation in global memory, and the reloading of the weights for every operator. Inside the kernel a plan is a chain of 900 dependent layer evaluations, each of about twenty sub-microsecond steps, with weights streamed from L2 into shared memory ahead of use and activations never leaving the chip between layers.

## 3.5 The fused encoder: 1.28×, one launch per plan

R4 moves the encoder into the same launch, reads the raw uint8 frame, and caches the goal embedding for the episode. The plan falls from 59.3 to 46.5 ms (1.28×). In isolation, encoding the observation and goal takes 1.34 ms in the kernel (1.25 ms for the observation alone, when the goal is cached) against 6.9 ms for the FP32 PyTorch encoder. The larger part of the gain, however, comes from what fusion removes around the encoder: R4 skips the policy's CPU image transform, which also removes most of the step-call latency, and the glue between plan and kernel shrinks from about 11 ms to about 4 ms (Figure 6). Of R4's 46.5 ms, about 41.3 ms is the predictor and iCEM solve, 1.3 ms the encoder and 3.9 ms everything else.

![Composition of the R3 and R4 plans (medians). The solve and encoder times are isolated measurements of the same kernel and encoders on the same GPU; the remainder (grey) is the difference to the end-to-end median and contains the policy's CPU preprocessing, Python glue and the transfers to and from the host.](figs/fig_breakdown.png)

## 3.6 Task success is unchanged at every rung

The ladder's claim on accuracy is a null result, and Table 5 and Figures 7 and 8 give its evidence. Every rung's success rate lies between 260 and 265 of 300, every Wilson interval overlaps every other, and no paired comparison with the reference or between adjacent rungs comes close to significance (smallest McNemar p = 0.50). The final rung differs from the reference in 39 of 300 episodes, 19 won and 20 lost, for a paired difference of −0.3 points with a 95% interval of −4.3 to +3.7.

%TABLE: Paired comparisons on the 300 Push-T episodes. Discordant episodes are counted as wins for the first rung / wins for the second; the interval is a paired bootstrap over episodes (10,000 resamples); p is the exact McNemar test.
| comparison | what changes | difference (points) [95% CI] | discordant | p |
|---|---|---|---|---|
| R1 vs R0 | caching, no host sync | +0.7 [0.0, +1.7] | 2 / 0 | 0.50 |
| R2 vs R1 | CEM → iCEM | +0.7 [−3.0, +4.3] | 17 / 15 | 0.86 |
| R2b vs R2 | compile, TF32, BF16 | −1.3 [−4.3, +1.7] | 8 / 12 | 0.50 |
| R3 vs R2 | megakernel (BF16, Philox) | −1.0 [−4.7, +2.3] | 13 / 16 | 0.71 |
| R3 vs R2b | megakernel vs LeWM-Flash | +0.3 [−3.3, +4.0] | 16 / 15 | 1.00 |
| R3-CEM vs R1 | megakernel, CEM schedule | −0.7 [−4.3, +3.0] | 15 / 17 | 0.86 |
| R3 vs R3-CEM | iCEM inside the megakernel | +0.3 [−3.7, +4.3] | 20 / 19 | 1.00 |
| R4 vs R3 | fused encoder | −0.7 [−4.0, +2.7] | 12 / 14 | 0.85 |
| R4 vs R0 | the whole ladder | −0.3 [−4.3, +3.7] | 19 / 20 | 1.00 |

![Push-T success per rung on the 300 paired episodes, with Wilson 95% intervals. The dashed line is the reference.](figs/fig_success.png)

![Paired outcomes against the reference. Each row counts the episodes that only the reference solved (left, grey) and that only the rung solved (right). The annotation gives the paired difference, its bootstrap 95% interval and the exact McNemar p-value.](figs/fig_paired.png)

Two features of the paired data deserve comment. The exact rung, R1, differs from the reference in 2 episodes, while every rung that changes the optimiser, the numerics or the random stream differs from it in 32 to 39. That is the scale of the variation a change of sampling noise alone produces in this task: an episode near the success boundary is decided by the particular candidates drawn, and a new random stream redraws them. The discordance between R3 and R3-CEM (39 episodes) is as large as that between R4 and the reference, although the two share their kernel and numerics and differ only in the population schedule. The comparisons therefore resolve differences of about 6 points at conventional power (Section 2.6.3); within that resolution, no step of the ladder changed what the planner achieves.

## 3.7 Numerical agreement with the reference

The megakernel is a numeric change, and Table 6 places its error against the accepted BF16 autocast path of LeWM-Flash on identical inputs. On eight (problem, population, variance) cases, the maximum relative cost error against FP32 is 0.0005 to 0.0043 for the megakernel and 0.0005 to 0.0040 for BF16 autocast, and both recover 27 to 30 of the FP32 top-30 candidates. The fused encoder is closer to FP32 than PyTorch's own BF16 autocast: its embeddings deviate by 0.41% in relative L2 against 0.52%, with a minimum cosine similarity of 0.99999, and costs computed from its embeddings are within 0.32% of FP32 with identical top-30 sets.

%TABLE: Numerical agreement with the FP32 eager reference on identical inputs (H100). Cost error is the maximum relative error over all candidates of a case; top-30 is the overlap with the FP32 top-30 set.
| quantity | LeWM-Flash BF16 autocast | megakernel |
|---|---|---|
| predictor cost, max relative error (8 cases) | 0.0005 – 0.0040 | 0.0005 – 0.0043 |
| top-30 overlap with FP32 (8 cases) | 28 – 30 | 27 – 30 |
| batched launch, 6 environments: max relative error / top-30 | — | ≤ 0.0029 / 29 – 30 |
| encoder embedding, relative L2 / minimum cosine (32 frames) | 0.0052 / 0.999986 | 0.0041 / 0.999992 |
| cost from encoder embeddings, max relative error / top-30 (6 cases) | — | ≤ 0.0032 / 30 |

Planner-level agreement was tested separately: 16 (observation, goal) problems were solved by the megakernel in one batched launch and by the PyTorch iCEM solver, and each returned plan was re-scored by the FP32 model. The median plan cost was 189.23 for the megakernel and 189.31 for PyTorch, and the megakernel's plan was better or equal on 7 of 16. The plans differ draw for draw, because the noise streams differ, but they come from the same distribution.

## 3.8 How the kernel reached 41 ms, and what did not work

The single-environment solve fell from 76.9 ms for the first correct kernel to 41.3 ms in five steps (Figure 9, Table 7). Each step removed a latency term from the dependent chain rather than arithmetic.

![Single-environment solve time of the megakernel through its development (H100, CUDA events, 30 iCEM iterations with the bucketed schedule). The dashed line is LeWM-Flash's compiled BF16 solve in the same micro-benchmark.](figs/fig_kernel_history.png)

%TABLE: Megakernel tuning log (single-environment solve). Accepted steps in order, then reverted attempts.
| step | solve (ms) | what changed and why |
|---|---|---|
| first correct kernel | 76.9 | parity with FP32 at the first run; the compiler had serialised every WGMMA because a `printf` failure path was reachable from the GEMM loop |
| vectorised data movement | 55.3 | 16-byte DSMEM, shared-memory and AdaLN-table accesses; an inline trap instead of a function call on the failure path, so nothing interrupts the WGMMA chains |
| relaxed cluster barrier, fast erf | 43.8 | one release fence and relaxed remote arrives: cluster barrier 3.6 → 0.65 µs; the library `erff` and reciprocal replaced by the Abramowitz–Stegun form with a fast divide |
| no producer warp | 41.9 | each warpgroup refills its own weight ring; eight warps keep 255 registers per thread, a ninth caps them at 168 and spills |
| warpgroup split at 16-CTA clusters, parallel M-phases | 41.3 | at batch 1 the two warpgroups of a CTA split q, k and the output projection from v; action-encoder and AdaLN phases split over (row block, slice) units with atomic accumulation |
| reverted: peel the first accumulate of each GEMM chain | worse | register pressure rose and spills grew |
| reverted: prefetch cached K and V with the weights | worse | as above |
| reverted: prefetch row-owner operands across the cluster barrier | worse | as above |

The micro-benchmarks bound what remains. A cluster barrier costs 0.65 µs and a grid barrier 1.6 µs; a 64 × 64 epilogue 0.15 µs; an empty launch 14 µs. Each of the 900 layer evaluations of a plan passes four cluster barriers and two row-owner reduction rounds, and each iteration four grid barriers; those synchronisation costs alone account for several milliseconds, and the rest of the 41 ms is the latency, not the throughput, of the WGMMA chains and their epilogues at 64-row tiles. The weight ring almost never waits. Getting much below 40 ms at batch 1 would need fewer cross-SM hand-offs per layer, for example one layer per cluster with resident weights or an all-reduce that removes the owner round trip, which is a redesign rather than a tuning step.

The ladder was measured twice (Appendix C). The megakernel rungs reproduced to within 1.3 ms (R3 58.0 and 59.3 ms; R3-CEM 81.7 and 82.1 ms). The PyTorch rungs moved by up to 18%: the reference from 527 to 622 ms, R1 from 391 to 377, R2b from 196 to 182, with R2 unchanged at 410. Launch-bound execution inherits the variance of the host CPU and of the container it lands on; a single launch does not.
