# Abstract

%ABSTRACT
Joint-embedding predictive world models plan by searching over action sequences in a learned latent space, and their planning latency has been the obstacle to using them for control. LeWorldModel (LeWM), the smallest model of the line, made planning fast enough to discuss for control by shrinking the representation to a single token, yet its released planner still takes 622 ms per plan for one robot on an H100. This paper asks how much of that latency can be removed without changing the model, the planner's semantics or its task success, and which kinds of change account for it. We build a seven-rung ladder on one H100, adding one change per rung to the released planner and measuring each rung's end-to-end latency for a single robot and its closed-loop success on the same 300 paired Push-T episodes, with exact McNemar tests and bootstrap intervals. Encoding the observation and goal once per plan and keeping the optimiser on the GPU is exact and removes 40% of the latency (377 ms). The improved cross-entropy method (iCEM), which evaluates 3.19 times fewer candidate rollouts, is 8% slower than CEM in eager PyTorch, where every cost call is launch-bound and costs about 20 ms at 60 candidates as at 300, and 1.38 times faster inside a megakernel; its value depends on the execution model beneath it. We then replace the PyTorch execution with LeWM-MK, one persistent ThunderKittens kernel per plan that runs the ViT encoder, the AdaLN predictor rollout with an exact KV cache, the cost and all 30 iCEM iterations with in-kernel sampling and elite selection, using thread-block-cluster tensor parallelism over distributed shared memory and a static weight stream. A plan takes 46.5 ms (p90 47.0 ms), 13.4 times faster than the reference and 3.9 times faster than the best PyTorch implementation with compilation, TF32 and BF16 autocast (182 ms). The kernel's BF16-operand, FP32-accumulate numerics match the accepted autocast path in cost error, and task success is unchanged at every rung: the final rung differs from the reference by −0.3 points (95% CI −4.3 to +3.7) on 300 paired episodes. A JEPA planner at batch 1 is bound neither by arithmetic nor by memory bandwidth but by the latency of a chain of 900 dependent layer evaluations, and fusing the whole optimiser loop into one launch is the lever that moves it.
%ENDABSTRACT

%PAGEBREAK
# Acknowledgements

We thank the authors of LeWorldModel and stable-worldmodel for releasing the model, its planner and its evaluation harness, which made a same-hardware comparison against the reference implementation possible, and the Hazy Research group for ThunderKittens and for the public account of their Llama megakernels, which this work follows in spirit. The GPU time was provided by Modal on rented H100 instances.

# Declaration

I declare that this paper is my own work, that it has not been submitted for any other degree or diploma, and that all sources of information and assistance have been acknowledged. Generative AI tools were used as a coding assistant, as a literature search aid and as a writing aid; every source cited was checked against the primary document, and every result reported was produced by the code and evaluation records described in Appendix E. No ethics approval was required: the research involved no human or animal participants.

%PAGEBREAK
%TOC
