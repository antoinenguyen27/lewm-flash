# 6. Evaluation

## 6.1 Latency

Table 8 and Figure 5 compare the three systems on the same H100. LeWM-MK computes a plan for one robot in 46.5 ms, 13.4 times faster than the reference planner and 3.9 times faster than the optimised PyTorch planner, which already uses compilation, CUDA Graph replay and BF16 arithmetic. All three confidence intervals are narrow and far apart.

%TABLE: End-to-end planning latency for one robot on one H100 (ms). Speed-ups are ratios of medians with bootstrap 95% intervals.
| system | plans | median [95% CI] | 90th percentile | interquartile range | step call | speed-up of LeWM-MK |
|---|---|---|---|---|---|---|
| reference planner | 73 | 622.4 [605.5, 626.7] | 673.8 | 595.9 – 641.7 | 2.14 | 13.4× [13.0, 13.5] |
| optimised PyTorch planner | 72 | 182.1 [177.3, 185.7] | 204.3 | 175.3 – 195.4 | 1.51 | 3.9× [3.8, 4.0] |
| **LeWM-MK** | 76 | **46.5 [46.4, 46.6]** | **47.0** | **46.1 – 46.8** | **0.82** | — |

![End-to-end planning latency of the three systems, log scale. Boxes span the interquartile range with the median as a vertical line; whiskers extend to the most extreme plan within 1.5 interquartile ranges; points beyond them are plotted individually.](figs/fig_latency.png)

For a controller, predictability matters as much as speed. The reference planner's plans spread over 200 ms and the optimised PyTorch planner's over 150 ms, including a separate group of plans near 115 ms that appear as outliers in Figure 5. LeWM-MK's interquartile range is 0.7 ms wide, and its 90th percentile is within 0.6 ms of its median. One of its 76 plans took 458 ms, a one-off in a single process that the data do not explain; it is reported and kept. At the median, a LeWM-MK plan fits inside the 50 ms period of a 20 Hz control loop, and comfortably inside the 100 ms of a 10 Hz loop. The calls that only execute a planned action also became cheaper (0.82 ms), because the fused pipeline no longer runs the CPU image transform on every call.

## 6.2 The new compute profile

Figure 6 shows the profile before and after. Of LeWM-MK's 46.5 ms, the search takes 41.3 ms (89%), the encoder 1.3 ms (3%) and everything outside the kernel 3.9 ms (8%). The redundant encoding is gone, the CPU synchronisation is gone, and the remaining search costs 46 µs per layer evaluation against 409 µs in the reference, nine times less.

![Compute profile of one plan before (reference planner) and after (LeWM-MK), by stage. Medians; stages without a direct measurement are attributed by difference (Section 3.5).](figs/fig_profile.png)

What limits the search now is the depth of the chain itself. The measured synchronisation primitives account for about 2.8 ms of the 41.3 ms (four cluster barriers per layer at 0.68 µs, and the grid barriers between phases at 1.6 µs), or 7%. The rest is the latency of the tensor-core instruction chains and the work between them at small matrix sizes. The weight stream almost never waits. LeWM-MK has therefore moved the planner from an overhead-bound regime into one bounded by the latency of its own dependent arithmetic, which is the regime a single robot's planning problem imposes.

## 6.3 Task success

Faster planning must not change what the robot achieves. Figure 7 and Table 9 give the evidence on the 300 paired Push-T episodes. The three systems solve 261, 261 and 260 episodes, and their Wilson intervals overlap almost completely. LeWM-MK and the reference planner disagree on 39 of the 300 episodes: LeWM-MK solves 19 that the reference fails, and fails 20 that the reference solves. The paired difference is −0.3 percentage points with a 95% interval of −4.3 to +3.7, and the exact McNemar test gives p = 1.00. The reference's success rate of 87% is below the 96% that LeWM's authors report for Push-T; we used the evaluation configuration released with the code and did not investigate the difference, which applies equally to every system.

![Push-T success of the three systems on the same 300 episodes, with Wilson 95% intervals. The dashed line is the reference planner.](figs/fig_success.png)

%TABLE: Paired comparison of task success on the 300 episodes. Wins and losses count the episodes solved by only one of the two systems; the interval is a paired bootstrap (10,000 resamples); p is the exact McNemar test.
| comparison | successes | difference (points) [95% CI] | wins / losses | p |
|---|---|---|---|---|
| LeWM-MK vs reference | 260 vs 261 | −0.3 [−4.3, +3.7] | 19 / 20 | 1.00 |
| optimised PyTorch vs reference | 261 vs 261 | 0.0 [−4.0, +4.0] | 18 / 18 | 1.00 |
| LeWM-MK vs optimised PyTorch | 260 vs 261 | −0.3 [−4.0, +3.3] | 16 / 17 | 1.00 |

The 39 disagreements are expected. Planning is a randomised search, and an episode that sits near the boundary between success and failure is decided by which candidates happen to be drawn; LeWM-MK draws its candidates from its own random number generator, so it redraws those episodes. The optimised PyTorch planner, which also differs from the reference only in its search and arithmetic, disagrees with it on 36. The evaluation detects differences of about 6 percentage points at 80% power. Within that resolution, LeWM-MK does not change task success.

## 6.4 Numerical fidelity

Table 10 compares LeWM-MK's arithmetic with FP32 on identical inputs. Across eight scoring cases its largest relative error is 0.06% to 0.32%, against 0.05% to 0.40% for the optimised PyTorch planner's BF16 arithmetic, and the two agree with FP32 on 27 to 30 and 28 to 30 of the 30 best candidates respectively. The fused encoder is closer to FP32 than PyTorch's own BF16 encoder: its embeddings differ by 0.41% (relative L2 norm) against 0.52%, and plans scored from its embeddings stay within 0.32% of FP32 with identical top-30 sets.

%TABLE: Numerical agreement with FP32 on identical inputs (one H100).
| check | optimised PyTorch (BF16) | LeWM-MK |
|---|---|---|
| scoring, maximum relative error (8 cases) | 0.05% – 0.40% | 0.06% – 0.32% |
| scoring, agreement of the 30 best candidates | 28 – 30 of 30 | 27 – 30 of 30 |
| encoder embedding, relative L2 error (32 frames) | 0.52% | 0.41% |
| scoring from encoder embeddings, maximum relative error / top-30 (6 cases) | — | 0.32% / 30 of 30 |

## 6.5 Validating the scaling model

The scaling model of Section 4.3 made a prediction that the probes can test. Table 11 compares the predicted and observed value of iCEM in the two execution regimes.

%TABLE: Predicted and observed speed-up of iCEM over CEM (end-to-end medians of the probes, encoder in PyTorch in both kernel probes).
| execution | fixed share of a scoring call | predicted | observed |
|---|---|---|---|
| PyTorch eager (377.0 vs 410.2 ms) | 99% | 1.01× | 0.92× |
| LeWM-MK search (82.1 vs 59.3 ms) | 49% | 1.54× | 1.39× |

The model gets the qualitative answer right in both regimes: no gain where the cost is fixed, a substantial gain where it is not. It overestimates both, by about 0.1, because it ignores costs that do not depend on the number of candidates: iCEM's elite bookkeeping, and inside the kernel the 30-iteration chain and the 64-row tiles that are partly empty at 60 candidates. As an analytic tool it did what was needed. It explained why the sample-efficient algorithm had appeared useless, and it predicted that it would pay once the overhead was removed.
