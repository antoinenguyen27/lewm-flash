# Literature review: "LeWM on H100"

Verified literature for a paper on accelerating LeWorldModel (LeWM) planning: exact planner-level changes, CEM to iCEM, and a single persistent ThunderKittens megakernel on H100 (622 ms to 46.5 ms per plan at batch 1, with the same task success).

Searched up to 2026-10-17. Each entry was checked against a primary source: arXiv abstract, HTML or API record; the publisher or proceedings page (PMLR, NeurIPS or Curran, CVF, USENIX, ACM or Crossref, RSS, ICLR proceedings); or an official blog or docs page. Where a published venue exists, the published version is cited. Numbers quoted under "Relevance" come from the paper's own text or tables unless marked otherwise.

Note on APA years: PMLR volumes are dated by publication. iCEM (CoRL 2020) appears in PMLR vol. 155, dated 2021. STORM and Implicit BC (CoRL 2021) appear in PMLR vol. 164, dated 2022. The citations below use the PMLR dates. If the thesis prefers conference years, change them consistently.

---

## Theme A: JEPA and latent world models used for planning

**A1.** Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 15619–15629). IEEE. https://doi.org/10.1109/CVPR52729.2023.01499
- In-text: (Assran et al., 2023)
- Relevance: Introduces I-JEPA, which predicts the *representations* of masked target blocks from a context block instead of reconstructing pixels. This is the origin of the joint-embedding predictive architecture that LeWM inherits. It is background for why LeWM's cost function is a distance in embedding space and not in pixel space.

**A2.** Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024). Revisiting feature prediction for learning visual representations from video. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=QaCCuDfBk2
- In-text: (Bardes et al., 2024)
- Relevance: Introduces V-JEPA, which extends feature prediction to video. The largest model (ViT-H/16) reaches 81.9% on Kinetics-400 with a frozen backbone. It links I-JEPA to the action-conditioned V-JEPA 2-AC planner (A3), and is background only.

**A3.** Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., . . . Ballas, N. (2025). *V-JEPA 2: Self-supervised video models enable understanding, prediction and planning* (arXiv:2506.09985). arXiv. https://doi.org/10.48550/arXiv.2506.09985
- In-text: (Assran et al., 2025)
- Relevance: Pre-trains on over 1M hours of video, then post-trains an action-conditioned world model (V-JEPA 2-AC) on under 62 h of DROID robot video. V-JEPA 2-AC is deployed zero-shot on Franka arms with image-goal planning. Planning uses CEM with 800 samples, 10 refinement iterations and horizon 1. The paper reports **16 s per action** on a single RTX 4090, against **4 min per action** for a Cosmos baseline (80 samples).
- How we use it: it is the main precedent showing that JEPA planning latency is a real deployment bottleneck. Its latency is still about 340x our 46.5 ms per plan, although the task and model scale differ.

**A4.** Balestriero, R., & LeCun, Y. (2025). *LeJEPA: Provable and scalable self-supervised learning without the heuristics* (arXiv:2511.08544). arXiv. https://doi.org/10.48550/arXiv.2511.08544
- In-text: (Balestriero & LeCun, 2025)
- Relevance: Argues that JEPA embeddings should follow an isotropic Gaussian, and introduces SIGReg (Sketched Isotropic Gaussian Regularization). SIGReg has one trade-off hyperparameter, linear time and memory, and no stop-gradient or teacher–student heuristics. It reports 79% ImageNet-1k linear-probe accuracy with a frozen ViT-H/14. SIGReg is the regulariser LeWM uses. It acts only at training time, so it does not change the inference graph we accelerate.

**A5.** Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). *LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels* (arXiv:2603.19312). arXiv. https://doi.org/10.48550/arXiv.2603.19312
- In-text: (Maes, Le Lidec, Scieur, et al., 2026). With only one Maes 2026 entry, use (Maes et al., 2026).
- Relevance: The model we accelerate. Facts from the paper:
  - About 15M trainable parameters, trainable on one GPU in a few hours.
  - Encoder: ViT-Tiny (about 5M parameters, patch 14, 12 layers, 3 heads, dimension 192), using the [CLS] token through a 1-layer MLP projector with BatchNorm.
  - Predictor: 6-layer transformer, 16 heads, about 10M parameters, AdaLN action conditioning. The appendix instead says a ViT-S backbone, which is internally inconsistent.
  - Loss: MSE plus λ·SIGReg, with λ = 0.1 and M = 1024 projections.
  - Planner: CEM with 300 samples per iteration, top-30 elites, initial variance 1, horizon 5 with frame-skip 5. It uses **30 iterations on Push-T** and 10 elsewhere, with receding-horizon MPC.
  - Speed: plans "up to 48× faster" than foundation-model world models (Fig. 3 caption: "up to ∼50× faster than DINO-WM", attributed to "∼200× fewer tokens"). Full planning completes "in under one second", with speed comparable to PLDM.
  - Solvers (Table 10): CEM 96.0 ± 2.83% Push-T success, against Adam at 84%.
- How we use it: the baseline we start from. Our no-optimisation 622 ms per plan on H100 is consistent with the paper's "under one second". Our work picks up exactly where LeWM's speed claim stops.

**A6.** Zhou, G., Pan, H., LeCun, Y., & Pinto, L. (2025). DINO-WM: World models on pre-trained visual features enable zero-shot planning. In *Proceedings of the 42nd International Conference on Machine Learning* (Vol. 267, pp. 79115–79135). PMLR. https://proceedings.mlr.press/v267/zhou25t.html
- In-text: (Zhou et al., 2025)
- Relevance: Learns a predictor over frozen DINOv2 *patch* features and plans with CEM-based MPC by minimising latent distance to a goal. It includes Push-T. MPC success is 0.90 on Push-T, against 0.86 for open-loop CEM and 0.28 for gradient descent. Table 10 (A6000 GPU) reports **53 s for one CEM 100×10 planning call**, against 0.014 s for a batch-32 forward step.
- How we use it: the main contrast. Its per-plan cost is dominated by patch-token count, which LeWM removes by using a single [CLS] latent. We then remove the remaining systems overhead.

**A7.** Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., . . . Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=a68SUt6zFt
- In-text: (Oquab et al., 2024)
- Relevance: The frozen encoder behind DINO-WM (A6). Cite it only to explain why foundation-model world models carry a large per-observation token budget, in contrast to LeWM's ViT-Tiny.

**A8.** Sobal, U., Zhang, W., Cho, K., Balestriero, R., Rudner, T. G. J., & LeCun, Y. (2025). Learning from reward-free offline data: A case for planning with latent dynamics models. In *Advances in Neural Information Processing Systems 38* (pp. 49126–49162). Curran Associates. https://doi.org/10.52202/085713-1465
- In-text: (Sobal et al., 2025)
- Note: the arXiv version lists the first author as "Vlad Sobal". The proceedings list "Uladzislau Sobal".
- Relevance: PLDM trains a JEPA-style latent dynamics model on reward-free offline data and plans with it. It shows better generalisation to unseen layouts and better data efficiency than model-free goal-conditioned RL. LeWM reports planning speed comparable to PLDM and higher Push-T success. PLDM is a sibling baseline in the same latent-MPC family that our optimisations would also apply to.

**A9.** Bar, A., Zhou, G., Tran, D., Darrell, T., & LeCun, Y. (2025). Navigation world models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 15791–15801). IEEE. https://doi.org/10.1109/CVPR52734.2025.01472
- In-text: (Bar et al., 2025)
- Relevance: A 1B-parameter conditional diffusion transformer world model that plans navigation with CEM (N = 120 candidates, 3 rollouts each, a single CEM iteration). Table 8 (RTX 6000 Ada) reports 30.3 s to simulate *one* trajectory, falling to 0.4 s with distillation.
- How we use it: shows that generative world models put latency into the model itself and pay for it with heavy approximation. LeWM plus our megakernel keeps the full CEM or iCEM budget and runs it in milliseconds.

**A10.** Terver, B., Yang, T.-Y., Ponce, J., Bardes, A., & LeCun, Y. (2026). What drives success in physical planning with joint-embedding predictive world models? *Transactions on Machine Learning Research*. https://doi.org/10.48550/arXiv.2512.24497
- In-text: (Terver et al., 2026)
- Note: arXiv v1 is dated 30 Dec 2025. The v3 comment states TMLR acceptance. Code is at github.com/facebookresearch/jepa-wms.
- Relevance: A systematic study of JEPA world models (JEPA-WMs) covering architecture, objective and planner. Planners compared are CEM, Nevergrad, Adam and GD. "CEM L2 performs best overall", and gradient methods do poorly on Push-T, Maze and Wall. Push-T settings are N = 300, H = 6, top-K = 10 (reduced from DINO-WM's 30) and J = 30 iterations. The authors note high sensitivity to these settings. **No planning wall-clock time is reported.**
- How we use it: confirms that CEM-family solvers are the right target to accelerate. It also shows the JEPA-planning literature does not treat latency as a variable.

**A11.** Gao, Y., & Xu, X. (2026). *Fast LeWorldModel* (arXiv:2606.26217). arXiv. https://doi.org/10.48550/arXiv.2606.26217
- In-text: (Gao & Xu, 2026)
- Relevance: The closest prior work on *speeding up LeWM planning*. It replaces LeWM's autoregressive one-step rollout with parallel action-prefix prediction, reducing model calls per candidate from 5 to 1. CEM uses 300 samples, 30 elites and H = 5.
- Reported timing (Appendix E, Table 2, RTX 4090, summed over the whole evaluation workload): CEM time falls from 33.7 s to 16.1 s, and dynamics time from 19.7 s to 4.1 s. The paper attributes the speedup to the algorithm change and does not discuss kernel-level or compiler optimisation.
- How we use it: a contrast. It changes the model (retraining required), while we keep LeWM's weights and outputs exact. The two approaches are complementary.

**A12.** Hu, J., Zheng, Y., Bo, S., Wang, T., Dubost, F., Mottini, A., Liu, J., Srinivasan, A., Zhong, K., Qian, K., Gao, S., & Cui, Q. (2026). *Representation geometry matters for planning with JEPA world models* (arXiv:2608.16287). arXiv. https://doi.org/10.48550/arXiv.2608.16287
- In-text: (Hu et al., 2026)
- Note: the arXiv v3 title is misspelled "Reperesentation". v1 was titled "SCALE: State-calibrated latent embeddings for JEPA planning in the right geometry".
- Relevance: The only work found that runs **iCEM with LeWM** (and DINO-WM), alongside CEM, across five compute tiers from 300 candidates × 30 iterations down to 10 × 3. Elites are max(round(0.1N), 2), and no sample-decay schedule is described.
  - Push-T iCEM success for LeWM: 86.0% at the top tier, 63.9% pooled.
  - Timing (Table 5, Cube, top tier): LeWM about 1.17 s per plan with iCEM and 1.21 s with CEM. DINO-WM about 91.7 s per plan with iCEM (about 77x slower).
- How we use it: direct precedent that iCEM is viable on LeWM. It does not use bucketed sample decay or elite retention as we do, and it does not optimise wall-clock time.

**A13.** Nguyen, H., Xu, X., & Huang, X. (2026). *Latent geometry beyond search: Amortizing planning in world models* (arXiv:2605.08732). arXiv. https://doi.org/10.48550/arXiv.2605.08732
- In-text: (Nguyen et al., 2026)
- Relevance: Replaces online search over a pretrained LeWM with a goal-conditioned inverse dynamics model. It matches or beats CEM in 7 of 8 settings with **100–130× lower per-decision cost**, after sweeping CEM, MPPI, iCEM and gradient planners.
- How we use it: the amortisation alternative to our approach. It trades exact search for a learned policy, while we keep the search and make it fast.

**A14.** Nam, H., Le Lidec, Q., Maes, L., LeCun, Y., & Balestriero, R. (2026). *Causal-JEPA: Learning world models through object-level latent masking* (arXiv:2602.11389). arXiv. https://doi.org/10.48550/arXiv.2602.11389
- In-text: (Nam et al., 2026)
- Note: the arXiv comment states ICML 2026 acceptance. Update to the proceedings reference when available.
- Relevance: An object-centric JEPA that claims "substantially more efficient planning" using about 1% of the latent input features of patch-based world models, with performance comparable to DINO-WM on Push-T. Like LeWM, it gets efficiency by shrinking the representation, which is complementary to our systems-level work.

**A15.** Wang, Y., Bounou, O., Zhou, G., Balestriero, R., Rudner, T. G. J., LeCun, Y., & Ren, M. (2026). *Temporal straightening for latent planning* (arXiv:2603.12231). arXiv. https://doi.org/10.48550/arXiv.2603.12231
- In-text: (Wang et al., 2026)
- Note: the arXiv comment states ICML 2026 camera-ready.
- Relevance: A curvature regulariser makes JEPA latent trajectories straighter so that gradient-based planning becomes stable. It is an example of the "replace sampling with gradients" route to cheaper planning, which we contrast with keeping a sampling-based planner and accelerating it.

**A16.** Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019). Learning latent dynamics for planning from pixels. In *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 2555–2565). PMLR. https://proceedings.mlr.press/v97/hafner19a.html
- In-text: (Hafner et al., 2019)
- Relevance: PlaNet is the canonical precedent for CEM planning in a learned latent space from pixels. It uses H = 12, I = 10 iterations, J = 1000 candidates and K = 100 elites (appendix). iCEM (B3) later reported real-time planning with PlaNet's models, so PlaNet is the historical template for LeWM-style latent MPC.

**A17.** Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In *Advances in Neural Information Processing Systems 31* (pp. 4754–4765). Curran Associates. https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
- In-text: (Chua et al., 2018)
- Relevance: PETS (probabilistic ensembles with trajectory sampling) plans with CEM instead of random shooting. The main text states it uses CEM. It established CEM-MPC over learned models as a standard, and is the baseline iCEM was designed to improve.

**A18.** Hansen, N. A., Su, H., & Wang, X. (2022). Temporal difference learning for model predictive control. In *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 8387–8406). PMLR. https://proceedings.mlr.press/v162/hansen22a.html
- In-text: (Hansen et al., 2022)
- Note: arXiv lists the authors as Hansen, Wang and Su. The PMLR order (Hansen, Su, Wang) is used here.
- Relevance: TD-MPC does short-horizon trajectory optimisation in the latent space of a decoder-free world model, with a terminal value function. It is a precedent for sampling-based planning in a learned latent space.

**A19.** Hansen, N., Su, H., & Wang, X. (2024). TD-MPC2: Scalable, robust world models for continuous control. In *The Twelfth International Conference on Learning Representations*. https://openreview.net/forum?id=Oxh5CstDJU
- In-text: (Hansen et al., 2024)
- Relevance: Plans with MPPI in latent space: 512 samples, 64 elites, 6 iterations, horizon 3, 24 policy-prior samples. The authors report about 2× planning throughput from "code-level optimizations". It is the closest precedent for treating planner throughput as an engineering target in latent-MPC, though only incrementally and without kernel work.

**A20.** Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion policy: Visuomotor policy learning via action diffusion. In *Proceedings of Robotics: Science and Systems XIX*. https://doi.org/10.15607/RSS.2023.XIX.026
- In-text: (Chi et al., 2023)
- Relevance: Popularised the Push-T benchmark (pushing a T-shaped block to a target pose with 2D end-effector control) used by DINO-WM, LeWM and this paper. Cite it as the source of the task setting. An extended journal version also exists (arXiv:2303.04137).

**A21.** Florence, P., Lynch, C., Zeng, A., Ramirez, O. A., Wahid, A., Downs, L., Wong, A., Lee, J., Mordatch, I., & Tompson, J. (2022). Implicit behavioral cloning. In *Proceedings of the 5th Conference on Robot Learning* (Vol. 164, pp. 158–168). PMLR. https://proceedings.mlr.press/v164/florence22a.html
- In-text: (Florence et al., 2022)
- Relevance: The original source of the planar T-pushing task that Diffusion Policy adapted into Push-T. Diffusion Policy says Push-T is adapted from IBC; this is confirmed by secondary sources, see Unverified item U9. Cite it with Chi et al. (2023) for the provenance of Push-T.

**A22.** Maes, L., Le Lidec, Q., Facury, L., Massaudi, N., Chaurasia, A., Capuano, F., Gao, R., Gillin, T., Haramati, D., Scieur, D., LeCun, Y., & Balestriero, R. (2026). *stable-worldmodel: A platform for reproducible world modeling research and evaluation* (arXiv:2605.21800). arXiv. https://doi.org/10.48550/arXiv.2605.21800
- In-text: (Maes, Le Lidec, Facury, et al., 2026). This form is needed only if A5 is also cited.
- Note: listed as a NeurIPS 2026 poster (neurips.cc/virtual/2026/poster/139680); the track is not stated. An earlier version is arXiv:2602.08968 ("stable-worldmodel-v1").
- Relevance: The evaluation harness we use for Push-T. Its README lists CEM, iCEM, MPPI, predictive sampling, GD/Adam, PGD and augmented-Lagrangian solvers. **No planning-speed figures are given.** Cite it as the tool and as the source of the reference CEM implementation our baseline uses.

**A23.** Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*. https://openreview.net/forum?id=YicbFdNTTy
- In-text: (Dosovitskiy et al., 2021)
- Relevance: The ViT architecture behind LeWM's ViT-Tiny encoder, which our megakernel fuses. Cite it for patch embedding and the [CLS] token.

**A24.** Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 4172–4182). IEEE. https://doi.org/10.1109/ICCV51070.2023.00387
- In-text: (Peebles & Xie, 2023)
- Relevance: The source of adaLN / adaLN-Zero conditioning in transformer blocks. LeWM's predictor uses AdaLN action conditioning, and Terver et al. (2026) also adopt AdaLN-zero. Cite it when describing the predictor blocks the megakernel implements.

---

## Theme B: Sampling-based trajectory optimisation

**B1.** Rubinstein, R. (1999). The cross-entropy method for combinatorial and continuous optimization. *Methodology and Computing in Applied Probability, 1*(2), 127–190. https://doi.org/10.1023/A:1010091220143
- In-text: (Rubinstein, 1999)
- Relevance: The original CE method for optimisation. Cite it for the CEM planner LeWM uses: sample, select elites, refit the Gaussian.

**B2.** de Boer, P.-T., Kroese, D. P., Mannor, S., & Rubinstein, R. Y. (2005). A tutorial on the cross-entropy method. *Annals of Operations Research, 134*(1), 19–67. https://doi.org/10.1007/s10479-005-5724-z
- In-text: (de Boer et al., 2005)
- Relevance: The standard tutorial reference for CEM's update rule and smoothing. Cite it with B1 when formalising the 300 × 30 CEM baseline.

**B3.** Pinneri, C., Sawant, S., Blaes, S., Achterhold, J., Stueckler, J., Rolinek, M., & Martius, G. (2021). Sample-efficient cross-entropy method for real-time planning. In *Proceedings of the 2020 Conference on Robot Learning* (Vol. 155, pp. 1049–1065). PMLR. https://proceedings.mlr.press/v155/pinneri21a.html
- In-text: (Pinneri et al., 2021). Use (Pinneri et al., 2020) only if the thesis cites by conference year.
- Note: PMLR spells the names "Stückler" and "Rolínek"; arXiv uses "Stueckler" and "Rolinek". arXiv:2008.06389.
- Relevance: iCEM adds:
  - colored-noise (temporally correlated) action sampling with exponent β;
  - keep-elites and shift-elites, carrying a fraction 0.3 of elites forward;
  - executing the best action;
  - an exponential population decay **N_i = max(N·γ^(−i), 2K)**.

  It reports 2.7–22× fewer samples and a 1.2–10× performance increase, and reaches real-time planning with PlaNet models in PyTorch.
- How we use it: the direct source of our contribution (2). Our bucketed 300 → 60 decay with elite retention is a discretised form of iCEM's decay and elite memory, chosen so that each bucket maps to a fixed GEMM shape inside the megakernel.

**B4.** Williams, G., Aldrich, A., & Theodorou, E. A. (2017). Model predictive path integral control: From theory to parallel computation. *Journal of Guidance, Control, and Dynamics, 40*(2), 344–357. https://doi.org/10.2514/1.G001921
- In-text: (Williams, Aldrich, & Theodorou, 2017)
- Relevance: The MPPI paper, with sampling done in parallel on a GPU. It is the founding precedent for treating sampling-based MPC as a GPU-parallel workload. MPPI is the main alternative to CEM in latent planners (TD-MPC2) and is among the stable-worldmodel solvers.

**B5.** Williams, G., Wagener, N., Goldfain, B., Drews, P., Rehg, J. M., Boots, B., & Theodorou, E. A. (2017). Information theoretic MPC for model-based reinforcement learning. In *2017 IEEE International Conference on Robotics and Automation (ICRA)* (pp. 1714–1721). IEEE. https://doi.org/10.1109/ICRA.2017.7989202
- In-text: (Williams, Wagener, et al., 2017)
- Relevance: MPPI with neural-network dynamics, run in real time on hardware for aggressive driving. It is a precedent for real-time sampling MPC through a *learned* model, which is the regime LeWM planning sits in.

**B6.** Bharadhwaj, H., Xie, K., & Shkurti, F. (2020). Model-predictive control via cross-entropy and gradient-based optimization. In *Proceedings of the 2nd Conference on Learning for Dynamics and Control* (Vol. 120, pp. 277–286). PMLR. https://proceedings.mlr.press/v120/bharadhwaj20a.html
- In-text: (Bharadhwaj et al., 2020)
- Relevance: Interleaves CEM sampling with gradient updates on the action sequences. It converges faster and avoids local minima, matching or beating CEM. It is a hybrid alternative to iCEM for cutting samples, which we did not adopt because it needs backpropagation through the predictor, outside our forward-only megakernel.

**B7.** Huang, K., Lale, S., Rosolia, U., Shi, Y., & Anandkumar, A. (2021). *CEM-GD: Cross-entropy method with gradient descent planner for model-based reinforcement learning* (arXiv:2112.07746). arXiv. https://doi.org/10.48550/arXiv.2112.07746
- In-text: (Huang et al., 2021)
- Relevance: Seeds with 15,000 CEM samples at t = 0, then refines with 50 samples per step plus gradient descent. This is about 100× fewer samples per step than 5,000-sample CEM and about 25% lower runtime than CEM-5000. It is still slower than CEM with 50–500 samples (about 2.2–2.6 s against 0.36–0.41 s). It is a useful contrast: reducing samples does not reduce wall-clock time unless the per-iteration cost is also cut, which is the gap our megakernel addresses.

**B8.** S V, J., Jalagam, S., LeCun, Y., & Sobal, V. (2023). *Gradient-based planning with world models* (arXiv:2312.17227). arXiv. https://doi.org/10.48550/arXiv.2312.17227
- In-text: (S V et al., 2023)
- Note: arXiv lists the first author as "Jyothir S V". Check the preferred surname form with the author if possible.
- Relevance: Notes that most visual world-model MPC uses population methods (CEM, MPPI), and proposes gradient-based MPC exploiting world-model differentiability. It matches or beats alternatives in sample-efficient settings. Together with Terver et al. (2026) and LeWM's Table 10, it frames the CEM-versus-gradient choice. CEM remains best on Push-T, which justifies accelerating CEM-family solvers.

**B9.** Bhardwaj, M., Sundaralingam, B., Mousavian, A., Ratliff, N. D., Fox, D., Ramos, F., & Boots, B. (2022). STORM: An integrated framework for fast joint-space model-predictive control for reactive manipulation. In *Proceedings of the 5th Conference on Robot Learning* (Vol. 164, pp. 750–759). PMLR. https://proceedings.mlr.press/v164/bhardwaj22a.html
- In-text: (Bhardwaj et al., 2022)
- Relevance: GPU-parallel, MPPI-style sampling MPC for a Franka arm, computing each control command in under 8 ms (125 Hz). It is a precedent for millisecond sampling MPC on GPU, but with analytic kinematics and costs, not a learned transformer world model.

**B10.** Vlahov, B., Gibson, J., Gandhi, M., & Theodorou, E. A. (2024). *MPPI-Generic: A CUDA library for stochastic trajectory optimization* (arXiv:2409.07563). arXiv. https://doi.org/10.48550/arXiv.2409.07563
- In-text: (Vlahov et al., 2024)
- Note: latest version v4, 24 Feb 2026.
- Relevance: A C++/CUDA MPPI library offering a **combined kernel**, with dynamics and cost fused in one kernel and intermediates kept in shared memory, alongside split kernels. It picks between them by timing. This is the closest robotics precedent for fusing the sampling-MPC rollout loop into one kernel, though for hand-written dynamics and not a neural world model.
- How we use it: it fuses one rollout. We fuse the whole 30-iteration solve, including the neural predictor and the encoder.

**B11.** Sundaralingam, B., Hari, S. K. S., Fishman, A., Garrett, C., Van Wyk, K., Blukis, V., Millane, A., Oleynikova, H., Handa, A., Ramos, F., Ratliff, N., & Fox, D. (2023). *cuRobo: Parallelized collision-free minimum-jerk robot motion generation* (arXiv:2310.17274). arXiv. https://doi.org/10.48550/arXiv.2310.17274
- In-text: (Sundaralingam et al., 2023)
- Relevance: GPU-parallel trajectory optimisation that solves hard motion-generation problems in about 50 ms on average (about 60× faster than prior trajectory optimisation). Its geometric planner runs within 20 ms. It is evidence that robotics planning reaches tens-of-milliseconds latency only through custom GPU kernels and parallelism, which is the same lesson we apply to world-model planning.

**B12.** Wang, T., & Ba, J. (2020). Exploring model-based planning with policy networks. In *International Conference on Learning Representations*. https://openreview.net/forum?id=H1exf64KwH
- In-text: (Wang & Ba, 2020)
- Relevance: POPLIN initialises or optimises planning through a policy network, about 3× more sample-efficient than PETS. A distilled policy can sometimes skip test-time planning. Mention it briefly as the "amortise the planner" alternative, alongside Nguyen et al. (2026). We keep explicit search.

---

## Theme C: Compute and kernel-level optimisation of small-batch transformer inference

**C1.** Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM, 52*(4), 65–76. https://doi.org/10.1145/1498765.1498785
- In-text: (Williams et al., 2009)
- Relevance: The roofline model, which bounds attainable throughput by arithmetic intensity against memory bandwidth. Use it to argue that LeWM planning at batch 1 is neither compute- nor bandwidth-bound but *latency- or launch-bound*, so fusion and persistence matter more than peak FLOPs.

**C2.** Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Heek, J., Xiao, K., Agrawal, S., & Dean, J. (2023). Efficiently scaling transformer inference. In *Proceedings of Machine Learning and Systems 5* (MLSys 2023). https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html
- In-text: (Pope et al., 2023)
- Note: the arXiv version adds A. Levskaya. The MLSys page lists the 9 authors above.
- Relevance: An analytical latency/MFU model for transformer inference, reporting 29 ms per token at low batch for PaLM-540B with int8 weights. It is the reference for small-batch inference being memory- and latency-dominated and for KV-cache reasoning. We use the same style of analysis at a much smaller scale.

**C3.** Gupta, K., Stuart, J. A., & Owens, J. D. (2012). A study of persistent threads style GPU programming for GPGPU workloads. In *2012 Innovative Parallel Computing (InPar)* (pp. 1–14). IEEE. https://doi.org/10.1109/InPar.2012.6339596
- In-text: (Gupta et al., 2012)
- Relevance: The canonical characterisation of "persistent threads": launch only as many blocks as the GPU can hold and loop over work. Our megakernel is a persistent-threads design with grid barriers between CEM or iCEM phases.

**C4.** Diamos, G., Sengupta, S., Catanzaro, B., Chrzanowski, M., Coates, A., Elsen, E., Engel, J., Hannun, A., & Satheesh, S. (2016). Persistent RNNs: Stashing recurrent weights on-chip. In *Proceedings of the 33rd International Conference on Machine Learning* (Vol. 48, pp. 2024–2033). PMLR. https://proceedings.mlr.press/v48/diamos16.html
- In-text: (Diamos et al., 2016)
- Relevance: Persistent kernels keep RNN weights in on-chip memory across timesteps, sustaining 2.8 TFLOP/s at mini-batch 4 on a TitanX. It is the closest neural-network precedent for our design, where the small LeWM predictor's weights are reused across 5 rollout steps × 30 solver iterations inside one launch.

**C5.** Gray, A. (2019, September 5). *Getting started with CUDA Graphs*. NVIDIA Technical Blog. https://developer.nvidia.com/blog/cuda-graphs/
- In-text: (Gray, 2019)
- Relevance: Quantifies launch overhead. For a 2.9 µs kernel on V100, it measures 9.6 µs per kernel with a sync after each launch, 3.8 µs with one sync per step, and 3.4 µs with a CUDA Graph. Use it to explain why our torch.compile/CUDA-Graph baseline still loses to a single persistent kernel: graphs remove CPU launch cost but not inter-kernel bubbles.

**C6.** NVIDIA. (2026). *CUDA programming guide* (Version 13.4.2). https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- In-text: (NVIDIA, 2026)
- Relevance: The official reference for CUDA Graphs (§4.2), Cooperative Groups (§4.4, grid-wide synchronisation and cooperative launch), asynchronous barriers (§4.10) and asynchronous data copies (§4.12). Cite it for the grid barriers and async copies our kernel relies on. Give the version and access date, since the guide is continuously updated.

**C7.** Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., . . . Chintala, S. (2024). PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2* (pp. 929–947). ACM. https://doi.org/10.1145/3620665.3640366
- In-text: (Ansel et al., 2024)
- Relevance: TorchDynamo and TorchInductor (torch.compile), which generate fused Triton kernels and support CUDA Graphs. This is the strongest "off-the-shelf" baseline in our optimisation ladder. Our megakernel replaces it, and we argue it is limited by per-operator kernel boundaries at batch 1.

**C8.** Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An intermediate language and compiler for tiled neural network computations. In *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages* (pp. 10–19). ACM. https://doi.org/10.1145/3315508.3329973
- In-text: (Tillet et al., 2019)
- Relevance: The tile-based kernel language that TorchInductor emits. It is the code-generation substrate of our torch.compile baseline and a point of contrast with ThunderKittens' explicit Hopper primitives.

**C9.** Ivanov, A., Dryden, N., Ben-Nun, T., Li, S., & Hoefler, T. (2021). Data movement is all you need: A case study on optimizing transformers. In *Proceedings of Machine Learning and Systems 3* (MLSys 2021). https://doi.org/10.48550/arXiv.2007.00072
- In-text: (Ivanov et al., 2021)
- Relevance: Shows transformer execution is dominated by data movement, not FLOPs. Global fusion and layout optimisation cut data movement by up to 22.91% and give 1.30× over frameworks on a BERT encoder layer. It is a precedent for whole-layer fusion arguments.

**C10.** Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems 35* (pp. 16344–16359). Curran Associates. https://doi.org/10.52202/068431-1189
- In-text: (Dao et al., 2022)
- Relevance: IO-aware tiled exact attention that keeps intermediates in SRAM. It is the conceptual basis for doing attention inside our fused kernel without materialising scores in HBM. LeWM's sequences are tiny, so the benefit for us is fusion, not long-sequence memory.

**C11.** Dao, T. (2024). FlashAttention-2: Faster attention with better parallelism and work partitioning. In *The Twelfth International Conference on Learning Representations*. https://openreview.net/forum?id=mZn2Xyh9Ec
- In-text: (Dao, 2024)
- Relevance: Better work partitioning across thread blocks and warps, which raises the original FlashAttention's 25–40% of peak FLOPs. It is background on warp-level work partitioning that our kernel also has to choose.

**C12.** Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In *Advances in Neural Information Processing Systems 37* (pp. 68658–68685). Curran Associates. https://doi.org/10.52202/079017-2193
- In-text: (Shah et al., 2024)
- Relevance: The Hopper-specific attention kernel. It uses warp specialisation and asynchrony of Tensor Cores (WGMMA) and TMA, reaching up to 840 TFLOP/s in BF16 (85% utilisation) on H100. It is the reference for the Hopper programming techniques (WGMMA, TMA, producer–consumer warps) our megakernel uses.

**C13.** Spector, B. F., Arora, S., Singhal, A., Parthasarathy, A., Fu, D. Y., & Ré, C. (2025). ThunderKittens: Simple, fast, and *adorable* kernels. In *The Thirteenth International Conference on Learning Representations*. https://openreview.net/forum?id=0fJfVOSUra
- In-text: (Spector, Arora, et al., 2025)
- Note: the ICLR version adds A. Parthasarathy and retitles the arXiv preprint "ThunderKittens: Simple, fast, and adorable AI kernels" (arXiv:2410.20399).
- Relevance: The framework our megakernel is written in. It provides 16×16 warp-level tiles, thread-block templates for asynchronous overlap, and grid-level abstractions. It matches cuBLAS on GEMM and FlashAttention-3 on attention inference, and wraps WGMMA and TMA.

**C14.** Spector, B., Juravsky, J., Sul, S., Dugan, O., Lim, D., Fu, D., Arora, S., & Ré, C. (2025, May 27). *Look ma, no bubbles! Designing a low-latency megakernel for Llama-1B*. Hazy Research Blog, Stanford University. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- In-text: (Spector, Juravsky, Sul, Dugan, et al., 2025)
- Relevance: The direct inspiration for our design. It fuses the whole Llama-3.2-1B batch-1 decode into a single ThunderKittens megakernel using:
  - an on-GPU instruction interpreter;
  - paged shared memory (13 pages of 16 KiB);
  - global-memory counters for fine-grained dependencies.

  It reports about 2.1 µs per dummy-kernel launch (about 1.3 µs with CUDA Graphs), about 78% of H100 bandwidth, forward passes under 1 ms on H100 (about 2.5× vLLM, over 1.5× SGLang), and under 680 µs on B200.
- How we use it: we apply the same "no bubbles" argument to a world-model *planner* rather than an LLM decoder.

**C15.** Spector, B., Juravsky, J., Sul, S., Lim, D., Dugan, O., Arora, S., & Ré, C. (2025, September 28). *We bought the whole GPU, so we're damn well going to use the whole GPU*. Hazy Research Blog, Stanford University. https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
- In-text: (Spector, Juravsky, Sul, Lim, et al., 2025)
- Relevance: The throughput-oriented follow-up: a tensor-parallel Llama-70B megakernel on 8×H100. It is more than 22% faster than SGLang end to end (19,170 to 23,468 tokens/s). Ablations credit about 14% to a global work queue and about 6% each to intra-SM pipelining and cross-SM interleaving. It shows megakernels also pay off at large batch, which is relevant if we extend to many robots per GPU.

**C16.** Wu, M., Cheng, X., Liu, S., Shi, C., Ji, J., Ao, M. K., Velliengiri, P., Miao, X., Padon, O., & Jia, Z. (2025). Mirage: A multi-level superoptimizer for tensor programs. In *19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)* (pp. 21–38). USENIX Association. https://www.usenix.org/conference/osdi25/presentation/wu-mengdi
- In-text: (Wu et al., 2025)
- Relevance: A superoptimiser that searches kernel, thread-block and thread-level program graphs to discover fused GPU kernels automatically. It is the compiler foundation of MPK (C17) and represents the automated alternative to our hand-written fusion.

**C17.** Cheng, X., Zhang, Z., Zhou, Y., Ji, J., Jiang, J., Zhao, Z., Xiao, Z., Ye, Z., Huang, Y., Lai, R., Jin, H., Hou, B., Wu, M., Dong, Y., Yip, A., Wang, S., Yang, W., Miao, X., Chen, T., & Jia, Z. (2026). MPK: A compiler and runtime for mega-kernelizing tensor programs. In *20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)* (pp. 1909–1926). USENIX Association. https://www.usenix.org/conference/osdi26/presentation/cheng
- In-text: (Cheng et al., 2026)
- Note: arXiv:2512.22219. The arXiv metadata lists "Zihao Ye" twice; the USENIX list (20 authors) is used here.
- Relevance: Mirage Persistent Kernel is the first compiler and runtime that automatically turns (multi-GPU) model inference into one persistent megakernel. It uses SM-level task graphs and decentralised in-kernel scheduling, and reports up to 1.7× lower end-to-end LLM inference latency.
- How we use it: the automated counterpart to our hand-written megakernel. It targets LLM serving, not an iterative planning loop with data-dependent elite selection.

**C18.** Nrusimha, A., Brandon, W., Mishra, M., Shen, Y., Panda, R., Ragan-Kelley, J., & Kim, Y. (2025). *FlashFormer: Whole-model kernels for efficient low-batch inference* (arXiv:2505.22758). arXiv. https://doi.org/10.48550/arXiv.2505.22758
- In-text: (Nrusimha et al., 2025)
- Relevance: Fuses the entire transformer forward pass into one kernel for low-batch LLM inference, where "memory bandwidth and kernel launch overheads are significant factors". It is an independent confirmation of the whole-model-kernel approach, again only for LLM decode.

**C19.** Jin, H., Hou, B., Wang, G., Lai, R., Chen, J., Ye, Z., Cai, Y., Dong, Y., Cheng, X., Zhang, Z., Zhao, Y., Huang, Y., Yang, L., Jiang, J., Oliaro, G., Ji, J., Miao, X., Grover, V., Mowry, T. C., . . . Chen, T. (2026). *Event Tensor: A unified abstraction for compiling dynamic megakernel* (arXiv:2604.13327). arXiv. https://doi.org/10.48550/arXiv.2604.13327
- In-text: (Jin et al., 2026)
- Note: the arXiv comment states acceptance at MLSys 2026. Update to the proceedings reference when available.
- Relevance: A compiler abstraction for megakernels with shape- and data-dependent dynamism. It is relevant because iCEM's shrinking sample counts and elite top-k are exactly this kind of dynamism, which we handle by bucketing to static shapes.

**C20.** Andersch, M., Palmer, G., Krashinsky, R., Stam, N., Mehta, V., Brito, G., & Ramaswamy, S. (2022, March 22). *NVIDIA Hopper architecture in-depth*. NVIDIA Technical Blog. https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- In-text: (Andersch et al., 2022)
- Relevance: The official description of the Hopper features our kernel uses:
  - H100 SXM5 has 132 SMs, up to 228 KB of shared memory per SM, and over 3 TB/s of HBM3.
  - Thread block clusters hold up to 16 blocks.
  - Distributed shared memory gives about 7× faster inter-block exchange than global memory.
  - TMA is single-thread-issued asynchronous bulk tensor copies.
  - Asynchronous transaction barriers are provided.

**C21.** NVIDIA. (2022). *NVIDIA H100 Tensor Core GPU architecture* [White paper, V1.02].
- In-text: (NVIDIA, 2022)
- Note: the contents were verified from a mirrored copy of the V1.02 PDF. The NVIDIA-hosted URL could not be fetched, because it redirected; see U2. Cite C20 as the accessible primary source if a URL is required.
- Relevance: The canonical architecture reference for fourth-generation Tensor Cores (WGMMA), the Tensor Memory Accelerator, thread block clusters and distributed shared memory.

**C22.** Luo, W., Fan, R., Li, Z., Du, D., Wang, Q., & Chu, X. (2024). Benchmarking and dissecting the Nvidia Hopper GPU architecture. In *2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)* (pp. 656–667). IEEE. https://doi.org/10.1109/IPDPS57955.2024.00064
- In-text: (Luo et al., 2024)
- Relevance: Independent microbenchmarks of Hopper's new units, including tensor-core instructions, DSM and asynchronous copy. Use it for latency and bandwidth figures when justifying cluster and DSM use. An extended version appears in *ACM Transactions on Computer Systems* (2026), https://doi.org/10.1145/3856813.

**C23.** Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed precision training. In *International Conference on Learning Representations*. https://openreview.net/forum?id=r1gs9JgRZ
- In-text: (Micikevicius et al., 2018)
- Relevance: The standard reference for reduced-precision arithmetic with FP32 master copies and accumulation. Cite it for our BF16/TF32 baseline rung and for keeping FP32 accumulation in the megakernel's WGMMA so that task success is unchanged.

**C24.** Kalamkar, D., Mudigere, D., Mellempudi, N., Das, D., Banerjee, K., Avancha, S., Vooturi, D. T., Jammalamadaka, N., Huang, J., Yuen, H., Yang, J., Park, J., Heinecke, A., Georganas, E., Srinivasan, S., Kundu, A., Smelyanskiy, M., Kaul, B., & Dubey, P. (2019). *A study of BFLOAT16 for deep learning training* (arXiv:1905.12322). arXiv. https://doi.org/10.48550/arXiv.1905.12322
- In-text: (Kalamkar et al., 2019)
- Relevance: Shows that BF16 keeps FP32's dynamic range and matches FP32 results without hyper-parameter changes. It justifies BF16 operands in our kernels; the planning-quality equivalence is shown empirically in our success-rate table.

---

## Synthesis and gap

1. JEPA world models are now a competitive route to goal-conditioned control. This line runs from I-JEPA and V-JEPA through DINO-WM, PLDM and V-JEPA 2-AC to LeWM and its 2026 descendants. Nearly all of them plan with CEM-family MPC in latent space (Zhou et al., 2025; Assran et al., 2025; Maes et al., 2026; Terver et al., 2026).
2. Where these papers report planning latency, it is a secondary, coarse metric:
   - DINO-WM: 53 s per CEM call on an A6000.
   - V-JEPA 2-AC: 16 s per action on an RTX 4090.
   - LeWM: "under one second", 48× faster than DINO-WM.
   - Hu et al. (2026): about 1.2 s per LeWM plan and about 92 s per DINO-WM plan.
   - Terver et al. (2026): no wall-clock time at all.
3. Every prior speedup comes from changing the *model or algorithm*:
   - fewer or no patch tokens (LeWM, C-JEPA);
   - parallel prefix prediction (Fast-LeWM, about 2× on CEM time);
   - gradient planners (Temporal Straightening; S V et al., 2023);
   - amortised inverse-dynamics policies (Nguyen et al., 2026, 100–130× per decision).

   We found no work that holds the model and planner semantics fixed and attacks the *systems* cost of planning, such as CPU syncs, redundant re-encoding and kernel-launch bubbles.
4. iCEM has been applied to LeWM only as an evaluated solver (Hu et al., 2026; Nguyen et al., 2026; stable-worldmodel). None of these uses iCEM's population-decay or elite-memory specifically to cut latency, or reports the latency/success trade-off of decay schedules on JEPA models.
5. In robotics, GPU-parallel sampling MPC is mature (Williams, Aldrich, & Theodorou, 2017; Bhardwaj et al., 2022; Sundaralingam et al., 2023). MPPI-Generic even fuses dynamics and cost into one kernel (Vlahov et al., 2024). These systems use analytic or small hand-written dynamics, not transformer world models with an image encoder.
6. On the ML-systems side, persistent kernels and megakernels are established for LLM decode:
   - persistent threads (Gupta et al., 2012) and Persistent RNNs (Diamos et al., 2016);
   - the Hazy Research Llama megakernels (Spector, Juravsky, et al., 2025);
   - FlashFormer, and the MPK and Event Tensor compilers.

   All of them target autoregressive language-model inference and not an iterative optimiser with data-dependent elite selection.
7. As far as we found, **no prior work fuses a world model's entire sampling-based planning loop** (encoder, multi-step predictor rollouts, cost, elite selection and distribution refit across all solver iterations) into a single persistent kernel. We also found none that uses Hopper-specific features (WGMMA, TMA, clusters with DSM, grid barriers) for robot planning.
8. Our contribution therefore sits at an unclaimed intersection: exact, model-preserving systems optimisation of JEPA planning, combining iCEM's sample-efficiency ideas (Pinneri et al., 2021) with megakernel techniques from LLM serving (Spector, Arora, et al., 2025; Cheng et al., 2026).
9. The resulting 46.5 ms per plan at batch 1 on H100 is, to our knowledge, the lowest reported plan latency for a JEPA world model at the full 300-sample, 30-iteration budget. It is about 13× below our measured 622 ms baseline, which matches the LeWM paper's "under one second", while keeping task success equal.
10. The literature also suggests two orthogonal next steps: combining our kernel with algorithmic speedups such as Fast-LeWM's prefix prediction, and an automated megakernel compiler that supports the dynamism iCEM needs (Jin et al., 2026).

---

## Unverified / not found

- **U1. LeWM absolute planning times (about 0.98 s against about 47 s for DINO-WM).** These figures appear only in press and secondary coverage (agendadigitale.eu, Medium). The paper text we could read gives only "under one second", "48×" and "∼50×"; the absolute numbers are probably in Fig. 3's plot. Do not quote them unless read from the figure. The paper also does not state the GPU used for timing.
- **U2. NVIDIA-hosted URL for the H100 whitepaper.** resources.nvidia.com redirected to nvidia.com, and only a third-party mirror (ece.lsu.edu) of V1.02 was readable. C20 (NVIDIA blog) is a fully verified substitute.
- **U3. Published venues not found for these arXiv preprints:** V-JEPA 2 (arXiv only), LeWM (no venue listed as of v3), CEM-GD (searched L4DC 2023 / PMLR 211 without a match), MPPI-Generic (no journal found), cuRobo (tech report), FlashFormer (no venue listed). They are cited as arXiv preprints. Re-check before submission.
- **U4. Exact track of stable-worldmodel at NeurIPS 2026.** The poster page does not state the track; it is cited as arXiv.
- **U5. Page numbers.** Not found for Pope et al. (2023, MLSys) and Ivanov et al. (2021, MLSys). The MLSys pages show no page range.
- **U6. MPK speedup range.** An earlier GitHub README states 1.2–6.7× latency reduction; the paper states "up to 1.7×". Only the paper figure is used.
- **U7. Hazy Research megakernel paper.** No archival paper (arXiv or venue) for the Llama-1B megakernel was found; only the blog posts are cited.
- **U8. Other 2026 megakernel preprints found but not reviewed in depth:** AutoMegaKernel (Jaber & Jaber, arXiv:2606.09682), Ada-MK (arXiv:2605.11581), ForgeMegakernel (arXiv:2609.12379) and ParallelKittens (Sul et al., arXiv:2511.13940). Titles and authors were checked against arXiv, but the content was not, so they are left out of the main list.
- **U9. Push-T provenance wording.** That Diffusion Policy adapts Push-T from Implicit BC (Florence et al., 2022) is confirmed by secondary sources; the exact sentence in Chi et al. (2023) was not read directly.
- **U10. Secondary-source figures for Fast-LeWM.** Press reports of "54.4 s → 28.3 s (48%)" and "85.8% → 90.5% success" were **not** found in the arXiv v2 HTML, whose Table 2 gives 33.7 s → 16.1 s for CEM time. Use only the paper's figures.
- **U11. TensorRT and a dedicated KV-cache reference.** Not added. Neither is load-bearing for a non-autoregressive JEPA planner; Pope et al. (2023) covers KV-cache reasoning if needed.
- **U12. Gradient-based planning author name (B8).** arXiv shows "Jyothir S V". The correct APA surname form could not be confirmed.
