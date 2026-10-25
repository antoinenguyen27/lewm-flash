# 4. Compute-profile analysis

This section reports what the data showed about the reference planner. Its findings fixed the design of Section 5.

## 4.1 Where a plan's time goes

The reference planner's median planning call takes 622.4 ms (95% CI 605.5 to 626.7) and is spread widely, from 556 to 759 ms. Planner telemetry from the 300-episode evaluation shows that 99.3% of a planning call is spent inside the search itself, so the policy code around it is not the problem. Table 4 breaks the call into stages.

%TABLE: Compute profile of one reference planning call (median, one robot, one H100).
| stage | ms | share | how it was measured |
|---|---|---|---|
| CPU preprocessing and policy code | 2.1 | 0.3% | median of the calls that only pop a planned action |
| image encoding | 207.4 | 33.3% | 60 encoder calls per plan (planner telemetry) × 3.46 ms per image (micro-benchmark: 6.91 ms for observation and goal) |
| predictor rollouts and distribution updates | 368.1 | 59.1% | probe that encodes once per plan: its planning call minus its policy cost minus one encoding |
| CPU synchronisation and other | 44.8 | 7.2% | remainder, by difference |
| total | 622.4 | 100% | end-to-end median |

Three findings follow from the profile and from the measurements behind it.

## 4.2 Finding 1: a third of the time re-encodes the same two images

A plan has two distinct images, the current observation and the goal, but the planner's telemetry counts 60 encoder calls per plan: the released planner treats the encoder as part of the scoring function and re-encodes both images in each of its 30 iterations. The images do not change during a plan, so 58 of the 60 calls are redundant, and they cost about 207 ms, a third of the plan. This is pure waste: removing it changes no computed value.

## 4.3 Finding 2: the search is bound by overhead, not computation

The search takes 368 ms for 30 iterations, 12.3 ms per iteration, or 409 µs per transformer-layer evaluation. The arithmetic of one layer evaluation for 300 candidates is a few billion floating-point operations: about 3 µs of the H100's BF16 tensor-core capacity, and still only tens of microseconds even at its FP32 rate without tensor cores. More than nine-tenths of the search time is therefore not computation.

The scaling model identifies what it is (Figure 3, Table 5). For eager PyTorch, one scoring call costs 20.0 ms of fixed overhead plus 0.10 ms per hundred candidates: 99% of a full-size call is fixed. The call is a sequence of a few hundred small kernels, each launched from Python, and each finishes before the next is issued, so its time is set by the number of launches, not by the number of candidates. Compiling the call with `torch.compile` and replaying it as a CUDA Graph cuts the fixed part to 1.9 ms, but the compiled planner still returns to the CPU every iteration, and its single-robot search still takes 82.3 ms for arithmetic worth a few milliseconds.

![Latency of one scoring call (a five-step rollout of every candidate) against the number of candidates, with the fitted line latency = a + b·N for each execution method. Eager PyTorch is almost entirely fixed overhead; the megakernel's call is shown for comparison (Section 6.5). Medians of 30 to 50 calls on one H100.](figs/fig_scaling.png)

%TABLE: Fitted scaling model of one scoring call, and what it predicts iCEM saves (30 iterations in both schedules: CEM 9,000 rollouts, iCEM 2,820).
| execution | fixed cost a (ms) | per 100 candidates b (ms) | R² | fixed share at 300 | predicted iCEM speed-up |
|---|---|---|---|---|---|
| PyTorch eager FP32 (reference) | 19.96 | 0.095 | 0.26 | 0.99 | 1.01× |
| PyTorch compiled BF16 (optimised baseline) | 1.95 | 0.971 | 1.00 | 0.40 | 1.70× |
| megakernel, one call per launch | 1.04 | 0.362 | 0.79 | 0.49 | 1.54× |

The low R² for eager execution is itself the finding: there is almost no variation with the number of candidates for a line to explain.

## 4.4 Finding 3: sample-efficient search only pays once overhead is gone

iCEM evaluates 3.19 times fewer candidates than CEM, and the scaling model predicts what that is worth. Under eager execution, where 99% of each call is fixed, it predicts a speed-up of 1.01, that is, nothing. The probes confirm it: encoding once per plan, the eager planner takes 377 ms with CEM and 410 ms with iCEM, 0.92 times as fast, because iCEM's bookkeeping costs slightly more than its smaller batches save. Only an execution method with a small fixed cost can turn fewer candidates into less time. A sample-efficient algorithm and an efficient execution model are therefore complements, not alternatives, and the design must provide both.

## 4.5 From findings to design

Table 6 maps each finding to the design response of Section 5.

%TABLE: Design responses to the profile.
| finding | evidence | design response |
|---|---|---|
| 58 of 60 encoder calls per plan are redundant | telemetry; 207 ms, 33% | encode each image once per plan; cache the goal for the whole episode |
| the search is fixed overhead: launches, Python, CPU round trips | 99% fixed share; 409 µs per layer for about 3 µs of arithmetic; 45 ms of synchronisation | run all 30 iterations, including sampling, scoring and distribution updates, inside one GPU program |
| fewer candidates only help without the overhead | predicted 1.01× and observed 0.92× for iCEM in eager PyTorch | adopt iCEM together with, not instead of, the fused execution |
| the work is a long dependent chain of tiny steps | 900 layer evaluations per plan, 64 to 300 rows each | minimise the cost of each step: keep data on chip, cheap synchronisation, weights prefetched ahead of use |
| encoding becomes a visible share once the search is fast | 6.9 ms of PyTorch encoding against a 41 ms search | move the encoder into the same GPU program |
