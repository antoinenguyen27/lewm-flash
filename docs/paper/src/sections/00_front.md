# Executive summary

%ABSTRACT
Robots that plan with a learned world model imagine the outcome of hundreds of candidate actions before they move, and that imagination takes time. This project studies LeWorldModel (LeWM), a compact AI world model that plans by searching over action sequences in its own learned representation, and asks why one plan for one robot still takes 622 milliseconds on an NVIDIA H100, one of the fastest GPUs available, and how much of that time can be removed without retraining the model or making the robot less successful. We treat the question as a data analytics problem. We collected per-call latency records, planner telemetry and component timings, prepared them into tidy datasets, and used them to build a stage-by-stage compute profile of the planner. The profile showed three things. A third of the time (207 ms) is spent re-encoding the same two camera images 60 times. Most of the rest (368 ms) is the planner's 30 optimisation iterations, each of which costs the same 20 ms in PyTorch whether it evaluates 60 candidate plans or 300: a fitted latency model attributes 99% of the cost to fixed per-launch overhead rather than to computation. And the arithmetic itself is negligible, because the model is tiny and the work is a long chain of small dependent steps. We designed the solution against that profile. It encodes each image once, uses a sample-efficient optimiser (iCEM) that evaluates 3.2 times fewer candidates, and runs the whole plan, from the raw camera frame to the chosen action distribution, as one persistent GPU program (a "megakernel") written with the ThunderKittens library. One plan now takes 46.5 ms (13.4 times faster; 95% CI 13.0 to 13.5), and 3.9 times faster than a carefully optimised PyTorch implementation, with a 90th percentile of 47.0 ms. Task success on 300 paired Push-T episodes is unchanged: 260 against 261 successes, a paired difference of −0.3 percentage points (95% CI −4.3 to +3.7; exact McNemar p = 1.00). The kernel's numerical error matches the accepted half-precision path. The solution pack contains the datasets and a single Python program that regenerates every number and chart in this report.
%ENDABSTRACT

# Acknowledgements

I thank my supervisor for their guidance, the authors of LeWorldModel and stable-worldmodel for releasing the model, the planner and the benchmark harness, and the Hazy Research group for ThunderKittens. GPU time was rented on Modal.

# Declaration

The assignment cover sheet is attached to this submission. I declare that this report is my own work, that it has not been submitted for any other degree or diploma, and that all sources of information and assistance have been acknowledged. Generative AI tools were used as a coding assistant, as a literature search aid and as a writing aid; every source cited was checked against the primary document, and every result reported was produced by the code and data in the solution pack (Appendix B). No ethics approval was required: the research involved no human or animal participants.

%PAGEBREAK
%TOC
