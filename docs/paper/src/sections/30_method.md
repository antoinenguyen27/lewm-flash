# 3. Methodology

## 3.1 Research design

The research is quantitative and follows an analytics pipeline (Figure 1). Data about the planner's behaviour are collected with timing instruments, prepared into tidy tables, and analysed to discover where the time goes and why. The patterns found determine the design of the solution, and the solution is evaluated with the same instruments and statistics, then packaged for deployment. Any design change that does not improve the measured profile without changing behaviour is discarded.

![The analytics pipeline of the project. Steps 1 to 3 produce the compute profile (Section 4); step 4 is the solution (Section 5); step 5 its evaluation (Section 6); step 6 the deliverables (Appendix B). The orange arrow is the loop by which every design change was re-profiled before it was kept.](figs/fig_pipeline.png)

Three design principles were fixed at the outset, and each answers a specific threat to validity.

- **Measure what a deployment would feel.** The primary metric is the end-to-end latency of one planning call for one robot, from camera image in to action out, including image preprocessing, encoding, the search and the copy back to the CPU. A kernel timed in isolation can hide costs that the robot still pays, so isolated timings are used only to explain the end-to-end number.
- **Equal hardware, strong baseline.** Every configuration runs on the same GPU model in the same software container. The new pipeline is compared with the released reference planner and with the best PyTorch implementation available, which already uses the standard remedies of Section 2.3 (compilation, CUDA Graph replay, TF32 and BF16 arithmetic). Beating only an unoptimised reference would not show that the design matters.
- **Behaviour is measured in the loop, on paired episodes.** A faster planner that makes the robot fail more often is worse, and small numerical changes can alter which candidate wins a close race. Every configuration is evaluated on the same 300 Push-T episodes, whose start states are fixed by the random seed, so differences are measured episode by episode.

## 3.2 System under study

Table 1 lists the facts of the model and planner that the analysis depends on, taken from the released code and checkpoint. Figure 2 shows how one plan is computed.

%TABLE: The reference system: LeWM's released Push-T checkpoint and planner.
| item | value |
|---|---|
| encoder | ViT-tiny: 12 layers, width 192, 3 attention heads, 14 × 14 patches of a 224 × 224 image (257 tokens), projected to a 192-number embedding |
| predictor | 6-layer transformer, width 192, 16 heads, MLP width 2,048, conditioned on actions by AdaLN; looks back over up to 3 past steps |
| action | a block of 5 two-dimensional pushes (10 numbers) per planning step; plans cover 5 steps |
| planner | CEM: 300 candidates, top 30 elites, 30 iterations (9,000 imagined rollouts per plan) |
| cost | squared distance between the predicted final embedding and the goal image's embedding |
| control loop | a plan is computed at policy calls 1 and 26 of a 50-call episode; the calls in between execute the planned actions |
| arithmetic | 32-bit floating point (FP32), PyTorch operations launched one at a time ("eager" mode) |

![How LeWM plans. Both images are encoded to embeddings; the planner samples candidate action sequences, rolls each through the predictor for five steps, scores the final embedding against the goal, refits its sampling distribution to the best candidates and repeats 30 times.](figs/fig_system.png)

Two properties of this system shape the problem. The model is small, about 15 million parameters, so scoring 300 candidates is under 100 billion floating-point operations, less than a tenth of a millisecond of the H100's tensor-core capacity, and all its weights fit in the GPU's 50 MB on-chip cache. And the computation is deep: one plan is 30 sequential iterations of a 5-step rollout through 6 layers, a chain of 900 transformer-layer evaluations, each of which depends on the one before.

## 3.3 Data collection

Four instruments produced the data (Table 2). All ran on NVIDIA H100 80GB HBM3 GPUs rented from Modal, with PyTorch 2.11.0 and CUDA 12.8 in one container image.

%TABLE: Datasets collected. All are in the solution pack's data folder; Appendix A gives the columns.
| dataset | rows | instrument | content |
|---|---|---|---|
| calls.csv | 11,942 | CUDA event timers around every policy call, inside the benchmark harness | latency of every call of 61 one-robot episodes for each of three systems and four probe configurations |
| episodes.csv | 900 | Push-T evaluation, 3 seeds × 100 episodes per system | success (0 or 1) of every episode |
| cost_scaling.csv | 15 | micro-benchmark of one scoring call at 60 to 300 candidates | median latency per execution method |
| components.csv | 11 | component micro-benchmarks and planner telemetry | encoder and search timings, synchronisation costs, encoder calls per plan |
| parity.csv | 15 | comparison with FP32 on identical inputs | maximum relative cost error and agreement of the top-30 candidates |

The latency episodes use seeds 1000 to 1060, one robot per process, so nothing is shared between robots. The seven configurations in calls.csv are the three **systems** compared in the evaluation (the reference planner, the optimised PyTorch planner and the new pipeline) and four **probes**: configurations that differ from a system in exactly one respect, run only to isolate an effect for the profile. Two probes encode the images once per plan, with CEM and with iCEM. The other two run the new kernel with CEM and with iCEM, using the PyTorch encoder.

## 3.4 Data preparation and preprocessing

The raw records are line-delimited JSON files written by the harness, one per run. The program's `prepare` step extracts them into the five tidy CSV tables, one row per observation. The `analyse` step then applies three preprocessing rules before any statistic is computed.

- **Warm-up removal.** The first seed of every configuration is discarded, because the first calls of a fresh process include one-off costs (memory allocation, kernel loading, and for the compiled baseline the compilation itself). This leaves 72 to 76 planning calls per configuration.
- **Call classification.** Policy calls 1 and 26 of each episode are planning calls; all others only pop an already-planned action and are kept as a separate step-call distribution, which measures the fixed cost of a policy call.
- **Outlier flagging.** Planning calls outside 1.5 interquartile ranges of their configuration's quartiles are counted and reported but kept. Medians are used as the location statistic because the latency distributions are skewed.

## 3.5 Analysis methods

**Stage decomposition.** A plan's time is divided into stages: CPU preprocessing, image encoding, the search (rollouts and distribution updates), and synchronisation with the CPU. Each stage is estimated from a direct measurement where one exists (the encoder's time from its micro-benchmark, multiplied by the number of encoder calls counted in the planner's telemetry; the search from the probe that removes the redundant encoding), and the remainder is attributed by difference and labelled as such.

**Scaling model.** To tell computation apart from overhead, the latency of one scoring call is modelled as a straight line in the number of candidates N, latency = a + b·N, fitted by least squares for each execution method. The intercept a is the fixed cost of a call, paid whatever its size; the slope b is the cost of each additional candidate. The share a / (a + 300b) is the fraction of a full-size call that is fixed overhead. A share near 1 identifies a launch- or latency-bound regime, in which reducing the number of candidates cannot help. The fitted model also predicts what iCEM should save: with 30 calls in both schedules, the predicted speed-up is (30a + 9,000b) / (30a + 2,820b).

**Per-layer cost.** Dividing the search time by the 900 layer evaluations of a plan gives the average cost of one layer evaluation, which can be compared with its arithmetic content and with the measured cost of the GPU's synchronisation primitives.

## 3.6 Evaluation design

Table 3 summarises the evaluation. Latency is summarised by the median with a bootstrap 95% confidence interval (10,000 resamples over plans) and by the 90th percentile, which matters for a controller with a deadline. Speed-ups are ratios of medians with bootstrap intervals. Task success is reported with Wilson 95% intervals (Wilson, 1927). Systems are compared episode by episode: the counts of episodes won by each side, an exact McNemar test (McNemar, 1947) and a paired bootstrap interval on the difference in success rate (Efron & Tibshirani, 1993). Because the claim is that behaviour does not change, the evaluation's resolution is stated: with the observed 13% of episodes differing between systems, 300 pairs detect a difference of about 6 percentage points at 80% power, and a 2-point difference would need about 2,500 pairs (Connor, 1987). Before any closed-loop run, the new pipeline's scoring was compared with FP32 on identical candidates, and accepted only if its error was no larger than that of the optimised PyTorch planner's BF16 arithmetic, whose closed-loop behaviour had already been accepted.

%TABLE: Evaluation protocol.
| item | setting |
|---|---|
| systems | reference planner; optimised PyTorch planner (iCEM, compiled, TF32/BF16); new pipeline (LeWM-MK) |
| latency | 61 one-robot episodes per system (first dropped), 72 to 76 plans; median, bootstrap 95% CI, 90th percentile |
| task success | 300 Push-T episodes (seeds 0 to 2 × 100), identical for every system; Wilson intervals; paired McNemar and bootstrap |
| numerical fidelity | 8 scoring cases against FP32: maximum relative error and top-30 agreement; encoder embedding error |
| hardware and software | one NVIDIA H100 80GB HBM3; PyTorch 2.11.0, CUDA 12.8; ThunderKittens commit `67845f5f` |

## 3.7 Tools and implementation

The planner and harness are the released LeWM and stable-worldmodel code with an extended benchmark harness. The new kernel is one CUDA source file (`megakernel/lewm_mk.cu`, about 1,600 lines) with a Python packer, runtime and planner class. GPU work ran on Modal. The analysis is one Python program (`solution_pack/lewm_flash.py`) that needs only numpy and matplotlib; it regenerates every table, statistic and data figure in this report from the CSV datasets in about five seconds on a laptop.
