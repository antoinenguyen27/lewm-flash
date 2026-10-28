# 7. Discussion

## 7.1 Were the objectives met?

**O1 (profile).** Yes. The profile located the reference planner's time precisely: a third in redundant image encoding, most of the rest in a search whose cost is 99% fixed overhead per call, and 7% in synchronisation with the CPU. The arithmetic the planner exists to perform accounts for a small fraction of its time.

**O2 (design and implement).** Yes. LeWM-MK removes each bottleneck the profile identified, with the trained model unchanged and the planner's horizon, iteration count and elite count unchanged. The one algorithmic change, iCEM's shrinking population, was adopted because the profile showed when it would pay.

**O3 (evaluate).** Yes. One plan for one robot takes 46.5 ms against 622.4 ms for the reference (13.4×) and 182.1 ms for the optimised PyTorch planner (3.9×), with a 90th percentile of 47.0 ms. On 300 paired episodes task success does not change detectably (−0.3 points, 95% CI −4.3 to +3.7), and the kernel's numerical error matches that of the accepted BF16 arithmetic.

**O4 (deploy).** Yes. The planner plugs into the existing harness by configuration name, and the solution program regenerates every result in this report from the collected data (Appendix B).

## 7.2 Why the solution works

The planner of a small world model is an unusual workload. Its model is tiny and its computation is deep: a few hundred billion floating-point operations per plan, spread across 900 dependent steps of 64 to 300 rows each. Neither of the quantities by which AI inference is usually optimised, the amount of arithmetic and the amount of memory traffic, limits it. What limits it is the fixed cost paid at every step: a kernel launch, a line of Python, a trip to the CPU, a round trip through main memory. Lower precision and bigger batches, the usual remedies, do little at one robot, because they shrink costs that were not binding. The remedy that works is to remove the steps between the arithmetic, and the planner's structure allows it. Once the search schedule is fixed, the planner's sequence of operations, the weights it needs and the synchronisation it requires are all known before it starts, which is what makes a single precomputed GPU program possible.

The analysis also showed that an algorithmic improvement and an execution improvement can depend on each other. iCEM was worth nothing under eager execution and a 1.39-times speed-up under fused execution. An evaluation that measured the algorithm in only one execution model would have reached the wrong conclusion either way. A planner's speed is a property of the algorithm and its execution together, and the profiling data are what reveal the interaction.

## 7.3 Position relative to other approaches

Other routes to faster world-model planning change the model or remove the search. Fast LeWorldModel retrains the predictor to roll out a whole candidate in one pass and halves planning time (Gao & Xu, 2026). An inverse-dynamics policy trained over LeWM replaces search with a single network evaluation, at 100 to 130 times lower cost per decision (Nguyen et al., 2026). LeWM-MK keeps the released weights and the explicit search and changes only how they are executed, so it composes with both: a faster model would shorten the chain the kernel walks, and a learned policy could give the search a better starting point. Megakernels for language models fuse one forward pass of a model per launch (Spector, Juravsky, Sul, Dugan, et al., 2025; Cheng et al., 2026). LeWM-MK fuses an entire optimisation loop of 2,820 model evaluations, 30 rounds of elite selection and their updates, which to our knowledge has not been done before for a learned world model.

## 7.4 Limitations

**Statistical resolution.** Three hundred paired episodes detect differences in task success of about 6 percentage points. The claim that LeWM-MK preserves success is a claim of no detectable change at that resolution; a difference of 2 or 3 points either way would need about 2,500 paired episodes to resolve.

**Attribution by difference.** Two stages of the reference profile (the search and the synchronisation) and one of the new profile (the time outside the kernel) are estimated from combinations of measurements rather than measured directly. The micro-benchmark also measures an eager scoring call at 20 ms where the harness implies about 12 ms per iteration. The analysis relies on the shape of the scaling curve, which both sources agree on, not on its absolute level.

**Scope.** All results are for one model, one task and one GPU type. The kernel is written for LeWM's dimensions and for features specific to the H100, and does not run on other GPUs without porting. The mechanism, a latency-bound chain of small dependent steps, is shared by sampling-based planning with any small world model, so the analysis should transfer even where the numbers do not.

**Simulation and the planning call only.** Push-T is simulated, and a 46.5 ms plan is a statement about computation, not about control on a physical robot, which adds sensing and actuation delays. Only the planning call was optimised.

**Measurement variance.** Latency was measured in one pass of 61 episodes per system, in a rented cloud container. An earlier pass of the same benchmark gave PyTorch medians that differed by up to 18% (Appendix C), while the configurations running the kernel reproduced to within about 2%. The PyTorch medians are therefore best read as typical values, and so are the speed-ups that use them.
