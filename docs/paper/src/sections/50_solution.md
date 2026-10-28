# 5. Solution design and development

The solution, LeWM-MK, combines four changes, each a response to a finding of Section 4. The trained model is used unchanged, and the planner keeps the reference's horizon, iteration count and elite count.

## 5.1 Encode each image once

The observation and the goal are encoded once per plan and passed to the search as embeddings, and the search keeps its ranking and distribution updates on the GPU, so that the CPU waits only for the final result. Within an episode the goal image never changes, so its embedding is computed on the first plan and reused, keyed by a hash of the image's bytes. This change is exact: on fixed candidates it reproduces the reference's scores with zero difference.

## 5.2 Search with iCEM

The search uses iCEM with the schedule that an earlier study of this planner validated on Push-T (Appendix C): 300 candidates in the first iteration, shrinking by a factor of 1.25 per iteration to a floor of 60 (twice the elite count), rounded up to multiples of 60 so that only five batch shapes occur; the top 30 candidates as elites; and the best 5 elites of each iteration carried into the next. A plan evaluates 2,820 rollouts instead of 9,000. Coloured noise, momentum and action clipping, which iCEM also offers, reduced success on Push-T and are not used. As Finding 3 showed, this change saves time only in combination with the next.

## 5.3 One GPU program per plan

The core of the solution is a persistent kernel that runs a whole plan: all 30 iterations of sampling candidates, encoding their actions, computing the AdaLN modulation, rolling every candidate through the predictor for five steps, scoring, selecting the elites and refitting the distribution. The CPU launches it once and receives the final action distribution. Figure 4 shows its structure. It is written with ThunderKittens for the H100 (Spector, Arora, et al., 2025).

![Structure of LeWM-MK. (a) One launch per plan, beginning with the encoder (Section 5.4). (b) The phases of one iCEM iteration, separated by barriers across the whole GPU. (c) One transformer layer inside a cluster of 8 or 16 processors: each computes its share of the attention heads and MLP, owner processors combine the partial results through distributed shared memory, apply the residual connection, normalisation and AdaLN modulation, and broadcast the input of the next operation.](figs/fig_megakernel.png)

Four design decisions carry its speed, and each attacks a cost identified by the profile (Table 7).

**The whole GPU stays resident.** The kernel starts one block of 256 threads on each of 112 of the H100's 132 processors (streaming multiprocessors), as seven clusters of 16, and keeps them running for the whole plan, so no kernel is ever launched mid-plan. Candidates are drawn inside the kernel from a counter-based random number generator (Salmon et al., 2011), and the elites are selected inside the kernel, so the CPU is not consulted between iterations. Phases are separated by a barrier across the whole GPU, which costs 1.6 µs.

**One small model is spread across a cluster.** At one robot the matrices are only 64 to 300 rows tall, too small to occupy a whole GPU one operation at a time. The kernel therefore groups processors into clusters of 16 and splits every layer across a cluster: each processor owns one of the 16 attention heads and a sixteenth of the MLP, computes its share with the H100's tensor-core instructions, and exchanges partial results with its neighbours through distributed shared memory rather than main memory. The cluster barrier that orders those exchanges is on the critical path four times per layer, 3,600 times per plan. Implemented naively, with full memory ordering on every signal, it cost 3.6 µs; with one memory fence and relaxed signals it costs 0.65 µs. Together with a faster GELU, this took about 11 ms off the search.

**Weights stream ahead of use.** Because the planner's sequence of operations is fixed, the order in which weights are needed is known before the plan starts. The weights are folded and repacked once on the CPU into 1,472 blocks of 24 KB, already laid out exactly as the tensor-core instructions read them, and each group of threads keeps a two-slot buffer that it refills with an asynchronous bulk copy as soon as it releases a slot. Prefetching therefore continues across operation, layer and iteration boundaries, and the computation almost never waits for weights. No separate thread group is spent on copying: on the H100 a ninth group of threads would cap every thread's registers at 168 instead of 255 and force spills to slower memory.

**No redundant work in the rollout.** In the first three of the five rollout steps the predictor's history window is still growing, so the attention keys and values of earlier steps are exactly those already computed; the kernel caches them and computes 9 token rows per candidate instead of 12, with no approximation.

%TABLE: Key design decisions of the kernel and their measured effect.
| decision | cost it removes | measured effect |
|---|---|---|
| persistent kernel, in-kernel sampling and elite selection | kernel launches and CPU round trips between and within iterations | one launch per plan; grid barrier 1.6 µs, empty launch 14 µs |
| tensor parallelism across a 16-processor cluster | idle processors at small matrix sizes; traffic through main memory | partial results exchanged on chip through distributed shared memory |
| relaxed-ordering cluster barrier | serialised signals on the critical path (3,600 per plan) | 3.6 → 0.65 µs per barrier; search 55.3 → 43.8 ms together with a faster GELU |
| static weight stream without a copy thread group | waiting for weights; register spills | 255 registers per thread instead of 168; search 43.8 → 41.9 ms |
| exact reuse of cached attention keys and values | recomputing earlier rollout steps | 9 instead of 12 token rows per candidate |

The kernel reached its final speed through a development loop that re-profiled each change. Three further attempts (removing the first accumulation of each matrix chain, prefetching cached keys and values, and prefetching owner operands across barriers) all made it slower by raising register pressure and were discarded.

## 5.4 The encoder inside the same program

Once the search took about 41 ms, the 6.9 ms PyTorch encoder and the CPU image transform before it became a visible share of a plan. The encoder therefore runs as the first phase of the same kernel. The kernel reads the raw 8-bit camera frame, normalises it and computes the patch embedding as one matrix multiplication. It then runs the 12 ViT layers with an attention computation that never stores the full attention matrix (Dao et al., 2022), and applies the projector. Encoding the observation now takes 1.25 ms (1.34 ms with the goal on an episode's first plan). A plan is one GPU launch from camera frame to action distribution.

## 5.5 Numerical design

The kernel computes matrix products with 16-bit brain-float (BF16) inputs and 32-bit (FP32) accumulation, keeps the residual stream, normalisation statistics, scores and search statistics in FP32, and uses an accurate approximation of the error function in the GELU activation (Abramowitz & Stegun, 1964). This is the same numerical class as the BF16 arithmetic of the optimised PyTorch baseline, which mixed-precision practice treats as safe for inference (Micikevicius et al., 2018; Kalamkar et al., 2019). The kernel is mathematically equivalent to the reference but not bit-identical, so its acceptance rests on measured error and on closed-loop behaviour (Section 6).

## 5.6 Deployment

LeWM-MK is deployed as a planner class (`MegakernelICEMSolver`) that plugs into the stable-worldmodel evaluation harness through its configuration system, so a user switches planners by changing one configuration name. The kernel is compiled once for the H100 (`sm_90a`), and builds are content-addressed by the hash of the source so that a stale build can never run. The analysis is deployed as the solution program described in Appendix B.
