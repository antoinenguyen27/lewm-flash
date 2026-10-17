# 2. Methodology

## 2.1 Research design

The research is quantitative and experimental. We take one released JEPA world model and its released planner, change the inference stack one step at a time, and measure each step with two instruments: the end-to-end latency of a replanning control call for a single robot on one GPU, and closed-loop task success on a fixed, paired set of Push-T episodes. Four design principles were fixed before the H100 programme spent any GPU time, and they are what separate a measured ladder from a collection of speed-ups.

First, every rung adds exactly one change to the rung below it, and every rung runs on the same hardware in the same container image. The reference is the stock LeWM planner as released, not a straw man and not a result transferred from another GPU. The steps of the ladder are then attributable: a rung's gain is the ratio of two measurements that differ in one change.

Second, the primary metric is batch-1 end-to-end plan latency. A deployed controller serves one robot, and it waits for the whole call, not for the planner kernel inside it. We therefore time the replanning `policy.get_action()` call from observation in to action out, including image preprocessing, encoding, the solve and the copy back to the host, and we do it with one environment per process so that no batching amortises any cost. Kernel-level timings are reported only to explain the end-to-end numbers.

Third, accuracy is measured in the loop, and comparisons are paired. A change that alters the numerics (BF16 operands) or the optimiser (iCEM) could change what the robot does even when every cost it computes is close to the reference. Every rung is evaluated on the same 300 Push-T episodes, whose start states are a deterministic function of the seed, so that two rungs differ only in the planner that acts on identical initial conditions. Evaluation episodes run 100 environments at a time, which is an evaluation convenience only: each environment's planner mathematics is the same whether one or a hundred are solved together.

Fourth, the change classes are kept separate. We distinguish exact changes, which compute the same function in a different order (caching, removing host synchronisation); planner changes, which change the optimisation algorithm (iCEM); and numeric changes, which compute the same function at lower precision or in a different arithmetic order (BF16 operands, fused kernels). Each class has its own acceptance test: exact changes must reproduce the reference costs and rankings on fixed candidates, numeric changes must stay within the cost and ranking error of the accepted BF16 autocast path, and every change must leave paired closed-loop success within noise of the reference.

## 2.2 System under test

We study LeWorldModel (LeWM; Maes, Le Lidec, Scieur, et al., 2026), a joint-embedding predictive world model trained end to end from pixels, in its released Push-T configuration, planned by the cross-entropy method of stable-worldmodel (Maes, Le Lidec, Facury, et al., 2026) exactly as the authors release it. The facts of the model and planner that the optimisations depend on are listed in Table 1; each was read from the released code and checkpoint rather than from the paper.

%TABLE: The system under test as released. Facts from the LeWM and stable-worldmodel repositories and the Push-T evaluation configuration.
| item | value |
|---|---|
| encoder | ViT-tiny: 12 layers, width 192, 3 heads, patch 14 on 224 × 224 images (256 patches + CLS = 257 tokens), LayerNorm ε = 10⁻¹²; CLS output through a BatchNorm projector to a 192-dimensional embedding |
| predictor | 6-layer transformer, width 192, 16 heads × 64, MLP 2,048, AdaLN-zero conditioning on the action embedding (Peebles & Xie, 2023), history window of 3 latent frames, causal attention; output through a BatchNorm projector (`pred_proj`) |
| action encoder | one-dimensional convolution with kernel 1 and a two-layer MLP: 10 → 768 → 192 per action block |
| action space | 2-dimensional end-effector displacement, grouped into blocks of 5 (frame-skip 5), so each plan step is a 10-dimensional action block |
| planning objective | squared Euclidean distance between the predicted embedding after the horizon and the goal embedding |
| planner (reference) | CEM: 300 candidate sequences, 30 iterations, top-30 elites, horizon 5 blocks, receding horizon 5, Gaussian initialisation with variance scale 1 |
| control loop | 50 policy calls per episode; a plan is computed on call 1 and on call 26 and its 5 action blocks are executed over the following calls; the other 48 calls pop buffered actions |
| evaluation | Push-T from the expert dataset: start state from the dataset, goal 25 steps later, budget 50 policy calls, success by the environment's coverage predicate |
| precision as released | FP32 parameters and arithmetic, eager PyTorch |

![LeWM planning on Push-T. The encoder maps the current observation and the goal image to 192-dimensional embeddings; the planner samples candidate action sequences, rolls each through the predictor for five autoregressive steps, scores the final embedding against the goal and refits the sampling distribution to the best candidates. The reference planner runs 30 iterations of 300 candidates per plan.](figs/fig_system.png)

Two properties of this system define the optimisation problem. The model is small: the whole model holds about 15 million parameters, about 5 million of them in the encoder (Maes, Le Lidec, Scieur, et al., 2026), so one cost evaluation of 300 candidates is under 100 GFLOP, less than a tenth of a millisecond of H100 tensor-core time, and all the weights fit in the GPU's 50 MB L2 cache. And the planner is deep: one plan is 30 sequential iterations of a 5-step autoregressive rollout through 6 layers, a chain of 900 dependent transformer-layer evaluations per plan whose individual matrix products are 64 to 300 rows tall. Such a workload is bound neither by arithmetic nor by memory bandwidth on a datacentre GPU; it is bound by latency, by how fast the hardware can walk a long chain of small dependent steps. Every rung of the ladder is an attack on a different term of that latency.

## 2.3 Task and episodes

Push-T requires the agent to push a T-shaped block into a target pose on a table, and it has become the common manipulation benchmark for planning with learned world models (Florence et al., 2022; Chi et al., 2023; Zhou et al., 2025; Maes, Le Lidec, Scieur, et al., 2026). We use stable-worldmodel's `swm/PushT-v1` environment with episodes drawn from the expert training set as in the LeWM evaluation: the environment is set to a recorded state, the goal is the recorded state 25 steps later, and the agent has 50 policy calls to reach it. Two episode sets are used and both are fixed by seed.

The **quality set** is 300 episodes, 100 for each of seeds 0, 1 and 2, the protocol of the LeWM-Flash study that preceded this one. The reference planner solves 87% of these episodes, below the 96% that LeWM reports for CEM on Push-T; we use the evaluation configuration released with the code (start states from the expert set, goal offset 25, budget 50) and did not investigate the difference, which is common to every rung. Episode start states and goals are a deterministic function of the seed, so every rung sees the same 300 episodes and every comparison between rungs is paired by episode.

The **latency set** is 61 single-environment episodes, seeds 1000 to 1060, run one environment per process. Each episode replans at calls 1 and 26 unless it succeeds before call 26, which gives one or two measured plans per episode. The first seed is dropped as process warmup, leaving 72 to 76 measured plans per rung. Calls that do not replan are recorded separately as step latency.

## 2.4 The ladder

The ladder has five main rungs and two side rungs (Table 2). R0 is the reference as released. R1 and R2 are the planner-level changes that the LeWM-Flash study accepted on an L40S, R1 exact and R2 changing the optimiser. R2b is LeWM-Flash itself, which adds the kernel- and precision-level changes of that study (`torch.compile`, TF32, BF16 autocast) to R2; it is the strongest PyTorch implementation we have and the baseline the megakernel has to beat. R3 replaces R2b's kernel and precision stack with one persistent megakernel that runs the predictor and the whole iCEM loop. R4 moves the encoder into the same launch. R3-CEM runs the megakernel with the reference CEM schedule, to price iCEM in an execution model where it can pay.

%TABLE: The H100 ladder. Each main rung adds one change to the rung above it; kind is the change class of Section 2.1.
| rung | change | kind | numerics of the planner |
|---|---|---|---|
| R0 | reference LeWM: stock stable-worldmodel CEM 300 × 30, re-encodes observation and goal inside every cost call | — | FP32 eager |
| R1 | encode observation and goal once per plan; keep CEM bookkeeping on the GPU with no host synchronisation inside the loop | exact | FP32 eager |
| R2 | bucketed iCEM: 300 → 60 candidates, top-30, 5 retained elites (2,820 evaluations per plan instead of 9,000) | planner | FP32 eager |
| R2b | LeWM-Flash: R2 + `torch.compile` of the cost function + TF32 matmuls + BF16 autocast of the rollout | numeric | BF16 operands, FP32 accumulation |
| R3 | megakernel: predictor, action encoder, AdaLN, cost, top-k and the elite update for all 30 iterations in one launch | numeric | BF16 operands, FP32 accumulation and residual |
| R4 | R3 + the ViT encoder and projector fused into the same launch on raw uint8 frames; goal embedding cached per episode | numeric | as R3; encoder BF16 operands, FP32 accumulation |
| R3-CEM | R3's megakernel running the reference CEM schedule (300 × 30, no retained elites) | side rung | as R3 |

### 2.4.1 Exact caching and synchronisation (R1)

The released planner computes the cost of a batch of candidates by re-encoding the observation and the goal for every call, 60 image encodings per plan for 2 distinct images, and it reads candidate costs back to the CPU on every iteration to rank them. R1 encodes both images once per plan, passes the embeddings into a tensor-only cost function, and keeps the ranking, elite selection and distribution update on the GPU, so that the only host synchronisation of the plan is the final action. On fixed candidates the change reproduces the reference costs with zero difference and identical rankings (LeWM-Flash exactness check, L40S).

### 2.4.2 Improved cross-entropy method (R2)

iCEM (Pinneri et al., 2021) adds to CEM a decaying population, elite retention across iterations, and optionally coloured noise, momentum and clipping. The LeWM-Flash study tested each feature on Push-T and accepted three: a population that decays geometrically by a factor of 1.25 per iteration from 300 to a floor of 60 (twice the elite count), rounded up to a multiple of 60 so that only five distinct batch shapes occur; the top 30 candidates as elites; and the 5 best elites of each iteration retained into the next population. Coloured noise, clipping and momentum lowered success or wall time on Push-T and were rejected, as were shorter schedules of 20 and 25 iterations. The resulting schedule (Figure 2) evaluates 2,820 candidate rollouts per plan against CEM's 9,000, 3.19 times fewer, at the same horizon, iteration count and elite count.

![Candidate rollouts per optimiser iteration. CEM evaluates 300 candidates in every one of 30 iterations; bucketed iCEM starts at 300 and decays by 1.25 per iteration to a floor of 60, rounded to multiples of 60.](figs/fig_icem_schedule.png)

### 2.4.3 The megakernel (R3)

R3 keeps every planner-level decision of R2 and discards the PyTorch execution of it. One persistent CUDA kernel, written with ThunderKittens (Spector, Arora, et al., 2025) for the Hopper architecture (Andersch et al., 2022), runs a whole plan: all 30 iterations of sampling, rollout, cost and distribution update, with the host seeing only the final mean and standard deviation. Figure 3 shows its structure; the design follows from the latency-bound character of the workload rather than from FLOPs.

![Structure of the megakernel. (a) One launch per plan; R4 adds the encoder phase at the front. (b) The phases of one iCEM iteration, separated by grid barriers. (c) The tensor-parallel schedule of one transformer layer inside a thread-block cluster: each CTA computes its own heads and MLP slices, row-owner CTAs reduce the partial results over distributed shared memory, apply the residual, LayerNorm and AdaLN modulation, and broadcast the next operand.](figs/fig_megakernel.png)

**Persistent grid and phases.** The kernel launches one 256-thread CTA (two warpgroups) per streaming multiprocessor, all co-resident, grouped into thread-block clusters of 8 or 16 CTAs (15 or 7 clusters on a 132-SM H100). Each iteration runs four phases separated by grid barriers: M1 draws candidates in the kernel from a counter-based Philox stream (Salmon et al., 2011) as mean plus standard deviation times noise, with slot 0 set to the mean and the next 5 slots to the retained elites, and runs the action encoder; M2 computes the AdaLN modulation table for every candidate, timestep and layer once; the rollout phase runs the five autoregressive predictor passes; and T computes the costs, a per-environment top-k and the elite mean and unbiased standard deviation, which it publishes for the next iteration.

**Tensor parallelism inside a cluster.** A cluster owns a 64-row chunk of candidate tokens. Each CTA owns 16/CL of the attention heads and 2,048/CL of the MLP and `pred_proj` hidden units, computes its share of every GEMM with Hopper warpgroup matrix-multiply-accumulate instructions (WGMMA), and writes a BF16 partial into its own shared memory. Row-owner CTAs then reduce the partials across the cluster through distributed shared memory (DSMEM), add the FP32 gated residual, apply LayerNorm and the AdaLN modulation, and broadcast the next GEMM operand into every CTA's shared memory. Each half-layer costs two cluster barriers. We implemented the cluster barrier with mbarriers, one release fence and relaxed remote arrives, which measured 0.65 µs against 3.6 µs for a barrier with a release on every arrive; the grid barrier measured 1.6 µs.

**Weights as a static stream.** All weights are folded on the host (inner LayerNorm affines, the BatchNorm of `pred_proj` and the kernel-1 convolution go into adjacent GEMMs) and packed into 964 panels of 24 KB (BF16 tiles of 192 × 64 or 64 × 192), each pre-swizzled in the exact ThunderKittens shared-tile layout so that a single bulk asynchronous copy (`cp.async.bulk`) lands it ready for WGMMA. Because the sample schedule is fixed, the sequence of panels a CTA needs over the whole plan is a static function of the schedule. Each warpgroup replays that sequence with a resumable iterator and its thread 0 refills a two-slot ring the moment the warpgroup releases a slot, so weight prefetch runs through operation, layer, pass and barrier boundaries without a dedicated producer warp. That choice is forced by the register file: a ninth warp caps every thread at 168 registers, eight warps keep 255.

**Exact KV reuse.** The predictor's history window grows from one to three frames over passes 0 to 2 and then slides. In the growing passes the keys and values of earlier positions are those of the previous pass, so the kernel caches them and computes 9 token rows per candidate over the five passes instead of 12, with no approximation.

**Numerics.** The kernel follows the BF16 autocast path of R2b: BF16 GEMM operands with FP32 accumulation, an FP32 residual stream and FP32 LayerNorm statistics, BF16 AdaLN modulation, queries, keys and values, FP32 costs and iCEM statistics. GELU uses the exact error function through the approximation 7.1.26 of Abramowitz and Stegun (1964) (absolute error ≤ 1.5 × 10⁻⁷) rather than the tanh form. The kernel is therefore mathematically equivalent to the reference and not bit-exact, and its noise comes from its own Philox stream, so its plans differ from the PyTorch solver's draw for draw while following the same distribution.

### 2.4.4 The fused encoder (R4)

In R3 the encoder still runs in PyTorch at FP32, preceded by the CPU image transform of the policy. R4 adds a 62-sub-phase encoder at the front of the same launch: the kernel reads the raw uint8 frame, normalises it, computes the patch embedding as a GEMM, runs the 12 ViT layers with a FlashAttention-style online softmax (Dao et al., 2022) over the 257 tokens, and applies the projector. The goal image of an episode does not change, so its embedding is cached under a hash of its bytes and only the observation is encoded on later plans. The encoder weights join the same panel stream, which grows from 964 to 1,472 panels (36 MB). A plan is then one kernel launch from image to action distribution.

## 2.5 Numerical acceptance

Before any closed-loop evaluation, every numeric rung was compared against the FP32 eager reference on fixed inputs. For the predictor and planner (R2b, R3, R4) we evaluate the cost of identical candidate sets drawn at the iCEM population sizes and at full and reduced sampling variance, eight (problem, population) cases in total, and record the maximum relative cost error and the overlap of the top-30 sets with the FP32 top-30. For the encoder (R4) we encode 32 Push-T frames and record the relative L2 error and minimum cosine similarity of the embeddings against FP32, and the cost error and top-30 overlap that result from planning with the fused embeddings. The acceptance criterion was set by the accepted BF16 autocast path of LeWM-Flash: a megakernel error at or below that path's on the same cases.

## 2.6 Evaluation

### 2.6.1 Latency protocol

Table 3 gives the latency protocol. The timer is a pair of CUDA events around `policy.get_action()`, recorded by the benchmark harness for every call and classified as a plan or a step by whether the solver ran. All rungs share one container image and the same process structure; each latency seed starts a fresh process, builds the world and loads the model, and for R2b compiles and warms the cost function at every bucketed shape before the first timed call.

%TABLE: Batch-1 latency protocol.
| item | setting |
|---|---|
| hardware | one NVIDIA H100 80GB HBM3 (SXM, 132 SMs), Modal |
| software | PyTorch 2.11.0 with CUDA 12.8, the same container image for every rung; ThunderKittens at commit `67845f5f`; kernel built with `nvcc` for `sm_90a` |
| episodes | 61 one-environment Push-T episodes, seeds 1000 to 1060; first seed dropped as warmup |
| timed quantity | end-to-end replanning `policy.get_action()` call (preprocessing, encoding, solve, action to host), CUDA events |
| plans per rung | 72 to 76 (calls 1 and 26 of each remaining episode, while it runs) |
| statistics | median (p50) with a bootstrap 95% interval over plans (10,000 resamples), p90, mean, minimum and maximum; step-call median |
| repeats | two complete passes of the ladder (Phase A without R4, and the final pass with all seven rungs) |

### 2.6.2 Closed-loop protocol

Each rung was evaluated on the 300 episodes of the quality set, 100 environments at a time, with the planner configuration of Table 2 and nothing else changed. The reference and LeWM-Flash quality runs were made on the same H100 configuration before the ladder (Section 2.7) with exactly the ladder's flags. Success is the environment's predicate at the end of the 50-call budget.

### 2.6.3 Statistics

The primary accuracy quantity is the success rate over the 300 paired episodes with a Wilson 95% interval (Wilson, 1927), about ±4 points at 87% success. Comparisons between rungs are paired by episode: we report the discordant-episode counts, an exact McNemar test (McNemar, 1947) and a paired bootstrap 95% interval on the difference in success rate with 10,000 resamples over episodes (Efron & Tibshirani, 1993). The ladder's claim on accuracy is a claim of no detectable change, so we state its resolution: at the observed discordance of 12 to 13% of episodes, 300 pairs detect a paired difference of about 6 points at 80% power and 5% two-sided significance, and a 2-point difference would need about 2,400 pairs (Connor, 1987). For latency we report the median of the plan distribution and a bootstrap interval on it, because the distributions of the PyTorch rungs are skewed and in one case bimodal. Gains are ratios of medians.

## 2.7 Experimental programme

The programme ran in three phases, of which the first predates this study.

- **Phase 0, LeWM-Flash on an L40S (June 2026).** A ledger of PyTorch-level changes to the released planner, each run as a paired 3 × 100 Push-T sweep against the reference: embedding caching and synchronisation removal (accepted), rollout preallocation and an action-embedding cache (rejected), compilation of the cost function (accepted), compiled and static whole-loop CEM variants (rejected), TF32 matmuls (accepted), receding-horizon warm starts of the CEM distribution (rejected: success fell to 34–42%), iCEM with its feature ablations (accepted in the form of Section 2.4.2), iCEM with gradient refinement of the elites (rejected: better model cost, no success gain, slower) and BF16 autocast of the rollout (accepted). The accepted stack is R2b. Its rejected paths are the reason R1 and R2 have the form they do.
- **Phase 1, the megakernel (16 October 2026).** The kernel was developed on H100 against the FP32 reference and the BF16 path, with the tuning steps of Section 3.6, and evaluated on the 300-episode quality set against the reference and LeWM-Flash on the same GPU. The H100 reference and LeWM-Flash quality runs used in the ladder were made in this phase.
- **Phase 2, the ladder (17 October 2026).** A first batch-1 pass of R0 to R3 and R3-CEM (Phase A) and quality runs of R1 and R2; then the fused encoder, its parity check, and a final batch-1 pass of all seven rungs with quality runs of R3, R3-CEM and R4.

Three deviations from the plan are material. The study was originally budgeted at $10 of GPU time for a megakernel on its own; the user extended it to $35 when the equal-hardware ladder was added, and the programme used about $31.50 (Appendix D). The batch-1 harness starts a new process per seed and pays the world construction, model load and, for R2b, the compile warmup 61 times, which made each latency pass cost $10 to $12 and limited the ladder to two passes. And the quality runs of the seven rungs were made in three batches over two days rather than in one, which is harmless to the pairing because the episodes are fixed by seed, but means the 300-episode results of R0 and R2b come from the Phase 1 runs with the same flags rather than from the ladder's own invocation.

## 2.8 Implementation and infrastructure

The planner code is the LeWM and stable-worldmodel source with the LeWM-Flash solvers (`lewm/solvers/`) and the benchmark harness (`tools/benchmarks/lewm_plan_benchmark.py`). The megakernel is a single CUDA source (`megakernel/lewm_mk.cu`, about 1,600 lines) with PyTorch bindings, a weight folder and panel packer (`pack.py`), a runtime wrapper (`runtime.py`) and a solver class that plugs into the harness (`solver.py`). Builds are content-addressed by the hash of the source, so a container never runs a stale extension. All GPU work ran on Modal: kernel builds on CPU containers with `nvcc` 12.8, and parity checks, micro-benchmarks, ladder passes and quality runs on single H100 containers, with each run's metrics, manifest and logs written to a results volume and committed to the repository. The analysis script `docs/paper/src/compute_stats.py` regenerates every latency and success statistic in this paper from the committed per-call records (Appendix E).
