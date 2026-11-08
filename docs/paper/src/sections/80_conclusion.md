# 8. Conclusion and future work

## 8.1 Conclusion

This project set out to make planning with a learned world model fast enough for robot control, without retraining the model or making the robot less successful, and to do it by analysing data rather than by guesswork. Profiling LeWM's released planner on an H100 showed that its 622 ms per plan was mostly not computation. A third was spent re-encoding two images 60 times. Most of the rest was a search whose cost per iteration is 99% fixed overhead, which also explains why a sample-efficient search algorithm had appeared useless. The arithmetic the planner exists to perform took a small fraction of its time.

The solution follows from that profile. Each image is encoded once. The search uses iCEM, which evaluates 3.2 times fewer candidates. The whole plan, from camera frame to action distribution, runs as one persistent GPU program that keeps its data on chip, spreads the small model across a cluster of processors and streams its weights ahead of use. One plan for one robot now takes 46.5 ms with a 90th percentile of 47.0 ms: 13.4 times faster than the reference and 3.9 times faster than an optimised PyTorch implementation on the same GPU. Task success on 300 paired episodes is unchanged within the evaluation's resolution, and the numerical error matches the accepted half-precision arithmetic. Planning with this world model has moved from the time scale of deliberation to the time scale of control.

The broader lesson is analytic. When an AI system is slow, the remedy depends on what kind of slow it is. A small model planning for one robot is limited neither by arithmetic nor by memory bandwidth but by fixed costs repeated thousands of times. A simple fitted model of how latency scales with workload is enough to tell the cases apart, and it predicted correctly which optimisation would pay.

## 8.2 Future work

**Spend the saved time on better plans.** The search budget was held fixed to measure execution alone. At 46.5 ms a plan with more iterations or a larger population still fits a 100 ms control period. A sweep of these settings, now cheap to run, would map the trade-off between latency and success and test whether more search closes the gap between the 87% success measured here and the 96% that LeWM's authors report.

**Shorten the remaining chain.** The search is now limited by 900 dependent layer evaluations with several cross-processor exchanges each. Combining the kernel with Fast LeWorldModel's one-pass rollout (Gao & Xu, 2026), or redesigning the kernel so that each cluster holds one layer and candidates stream through, would attack the chain directly.

**Other models, other hardware.** LeWM's single-vector representation is the most favourable case for this design. Larger world models such as DINO-WM and V-JEPA 2-AC have larger matrices and may return to a regime where throughput matters, and onboard robot computers have different costs again. Automatic megakernel compilers (Cheng et al., 2026; Jin et al., 2026) could make porting a matter of compilation rather than rewriting.

**Evaluate at the resolution a deployment needs, and on a robot.** Resolving differences in success of 2 points needs about 2,500 paired episodes, which at the new speed is a matter of minutes of planning rather than hours. The final step is a physical robot, where the question becomes how much faster replanning improves what the robot achieves.
