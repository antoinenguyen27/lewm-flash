// Presentation deck for the paper "Planning at the Speed of Control: Fusing a JEPA World-Model Planner into One GPU Kernel".
// Usage: NODE_PATH=<node_modules with pptxgenjs> node make_deck.js   (run from this directory; figures from ../figs)
const pptxgen = require("pptxgenjs"); const fs = require("fs"); const path = require("path");
const pres = new pptxgen(); pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Antoine Nguyen"; pres.title = "Planning at the Speed of Control: Fusing a JEPA World-Model Planner into One GPU Kernel";
const NAVY = "14213D", INK = "1B1B1B", GREY = "5A5A5A", LIGHT = "F3F4F6", BLUE = "2A78D6", ORANGE = "EB6834", WHITE = "FFFFFF", PALE = "E9F1FC";
const HF = "Cambria", BF = "Calibri";
const W = 10, H = 5.625, M = 0.5;
const FIG = f => path.join(__dirname, "..", "figs", f);
const png = f => { const b = fs.readFileSync(f); return { data: "image/png;base64," + b.toString("base64"), w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
function fitImage(slide, file, x, y, maxW, maxH) { const im = png(FIG(file)); const s = Math.min(maxW / im.w, maxH / im.h); const w = im.w * s, h = im.h * s; slide.addImage({ data: im.data, x: x + (maxW - w) / 2, y: y + (maxH - h) / 2, w, h }); }
function title(slide, text, dark = false) { slide.addText(text, { x: M, y: 0.32, w: W - 2 * M, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: dark ? WHITE : NAVY, isTextBox: true, margin: 0, valign: "middle" }); }
function body(slide, items, x, y, w, h, opts = {}) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: opts.bullet === false ? false : { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 7 } }));
  slide.addText(runs, { x, y, w, h, fontFace: BF, fontSize: opts.size || 14, color: opts.color || INK, isTextBox: true, margin: 0, valign: "top", ...opts.extra });
}
function card(slide, x, y, w, h, big, small, col = BLUE, bigSize = 30) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  slide.addText(big, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h * 0.55, fontFace: HF, fontSize: bigSize, bold: true, color: col, isTextBox: true, margin: 0, valign: "middle", align: "center" });
  slide.addText(small, { x: x + 0.15, y: y + h * 0.6, w: w - 0.3, h: h * 0.38, fontFace: BF, fontSize: 10, color: GREY, isTextBox: true, margin: 0, valign: "top", align: "center" });
}
function label(slide, text, x, y, w, col = GREY, size = 11) { slide.addText(text, { x, y, w, h: 0.3, fontFace: BF, fontSize: size, color: col, isTextBox: true, margin: 0, italic: true }); }
function footer(slide, n) { slide.addText(`${n}`, { x: W - M - 0.5, y: H - 0.38, w: 0.5, h: 0.25, fontFace: BF, fontSize: 9, color: GREY, isTextBox: true, margin: 0, align: "right" }); }
function panels(slide, items, y, h, fill = LIGHT, gap = 3.1, wd = 2.9, size = 11.5) {
  items.forEach((c, i) => { const x = M + i * gap; slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: wd, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
    slide.addText(c[0], { x: x + 0.18, y: y + 0.12, w: wd - 0.35, h: 0.42, fontFace: HF, fontSize: 15, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    slide.addText(c[1], { x: x + 0.18, y: y + 0.58, w: wd - 0.35, h: h - 0.7, fontFace: BF, fontSize: size, color: INK, isTextBox: true, margin: 0, valign: "top" }); });
}
let n = 0;

// 1 Title (dark)
{ const s = pres.addSlide(); s.background = { color: NAVY }; n++;
  s.addText("Planning at the Speed of Control", { x: M, y: 1.0, w: W - 2 * M, h: 0.9, fontFace: HF, fontSize: 36, bold: true, color: WHITE, isTextBox: true, margin: 0, valign: "middle" });
  s.addText("Fusing a JEPA world-model planner into one GPU kernel: an equal-hardware optimisation ladder for LeWorldModel on H100, from exact caching and iCEM to a ThunderKittens megakernel", { x: M, y: 2.0, w: W - 2 * M, h: 1.0, fontFace: BF, fontSize: 16, color: "CADCFC", isTextBox: true, margin: 0 });
  s.addText([{ text: "Antoine Nguyen  ·  24433599", options: { breakLine: true, bold: true } }, { text: "Research presentation  ·  [Affiliation]", options: { breakLine: true } }, { text: "Supervisor: [supervisor name]  ·  October 2026", options: {} }],
    { x: M, y: 3.9, w: W - 2 * M, h: 1.0, fontFace: BF, fontSize: 13, color: WHITE, isTextBox: true, margin: 0, paraSpaceAfter: 4 });
  s.addNotes("Two things in fifteen minutes: why the planning latency of JEPA world models is a systems problem rather than a modelling problem, and what happens when you treat it as one. The title is the claim: a world-model planner fast enough for a control loop, with the model and the planner unchanged."); }

// 2 The project in one slide
{ const s = pres.addSlide(); n++; title(s, "The project in one slide");
  body(s, ["JEPA world models plan by searching over action sequences in latent space; LeWM is the smallest of them, and its released planner still takes 622 ms per plan for one robot on an H100.",
           "Question: how much of that latency can be removed without changing the model, the planner or its task success, and which kinds of change account for it?",
           "Method: a seven-rung ladder on one H100, one change per rung, each measured end to end at batch 1 and in closed loop on the same 300 Push-T episodes.",
           "Answer: exact caching gives 1.65x; iCEM is worth nothing in eager PyTorch and 1.38x in a megakernel; one persistent kernel per plan gives 46.5 ms, with success unchanged."],
    M, 1.15, 5.6, 3.9, { size: 14 });
  card(s, 6.5, 1.2, 1.5, 1.15, "622 ms", "reference LeWM,\nCEM, FP32", GREY, 24); card(s, 8.1, 1.2, 1.5, 1.15, "182 ms", "best PyTorch path\n(LeWM-Flash)", ORANGE, 24);
  card(s, 6.5, 2.55, 1.5, 1.15, "46.5 ms", "one megakernel\nlaunch per plan", BLUE, 24); card(s, 8.1, 2.55, 1.5, 1.15, "−0.3 pts", "success vs reference,\n300 paired episodes", NAVY, 22);
  label(s, "One H100, batch 1, end-to-end plan latency (median); 300 paired episodes", 6.5, 3.85, 3.2, GREY, 10);
  s.addNotes("This is the abstract compressed. Keep four numbers in mind: 622 milliseconds for the released planner, 182 for the best PyTorch implementation I had, 46.5 for one kernel launch, and task success that does not move. Everything that follows explains how the 622 becomes 46.5, and why the steps are not the ones you might expect."); footer(s, n); }

// 3 Motivation
{ const s = pres.addSlide(); n++; title(s, "Why this topic: a latency problem");
  body(s, ["A controller replans every few steps and must answer within the period it plans for: a 10 Hz loop has 100 ms for sensing, planning and actuation together.",
           "LeWM's model is tiny: a cost call for 300 candidates is under 0.1 ms of tensor-core arithmetic, and all its weights fit in the H100's L2 cache.",
           "Its plan is slow for other reasons: re-encoding inside every cost call, a host round trip per optimiser iteration, every operator launched from Python, and a chain of 900 dependent layer evaluations per plan.",
           "Neither FLOPs nor memory bandwidth binds. The latency of each link of the chain does."],
    M, 1.15, 5.4, 3.9, { size: 14 });
  card(s, 6.3, 1.2, 1.55, 1.1, "15M", "parameters in\nthe whole model", NAVY); card(s, 8.0, 1.2, 1.55, 1.1, "2,820", "rollouts per plan\n(iCEM; CEM: 9,000)", NAVY);
  card(s, 6.3, 2.5, 1.55, 1.1, "900", "dependent layer\nevaluations per plan", NAVY); card(s, 8.0, 2.5, 1.55, 1.1, "20 ms", "one eager cost call,\nat 60 or 300 candidates", NAVY);
  s.addText("The released planner is 30 iterations x 5 predictor passes x 6 layers, each link a handful of small kernels. Batch 1 is what a robot runs.", { x: 6.3, y: 3.8, w: 3.25, h: 0.9, fontFace: BF, fontSize: 11, color: GREY, italic: true, isTextBox: true, margin: 0 });
  s.addNotes("The motivation is control. LeWM was designed to make planning cheap by shrinking the representation, and it succeeded: the arithmetic of a plan is trivial for a datacentre GPU. Yet a plan still takes over half a second, because of how the planner is executed. That makes it a systems problem, and that is what this project attacks."); footer(s, n); }

// 4 Reflection: how I got here
{ const s = pres.addSlide(); n++; title(s, "How I got here (reflection)");
  const steps = [["LeWM-Flash on an L40S", "A ledger of PyTorch-level changes to the released planner, each run as a paired 300-episode sweep: caching, no host syncs, compilation, TF32, iCEM and BF16 accepted; warm starts, whole-loop compile, CEM-GD and static BF16 rejected."],
                 ["The wall", "The best PyTorch planner still spent most of its time between kernels: 82 ms per solve on an H100 for a few milliseconds of arithmetic. Compilers fuse operators, not an optimiser loop."],
                 ["The megakernel", "Rewrite the whole iCEM solve as one persistent ThunderKittens kernel on Hopper, following the LLM-decoding megakernels: 76.9 ms for the first correct kernel, 41.3 ms after five tuning steps."],
                 ["The ladder", "Measure every step on the same H100 against the reference, at batch 1 end to end, with paired closed-loop success; then fuse the encoder so that a plan is one launch."]];
  steps.forEach((st, i) => { const y = 1.15 + i * 0.98;
    s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: i === 3 ? BLUE : NAVY }, line: { color: WHITE } });
    s.addText(String(i + 1), { x: M, y: y + 0.05, w: 0.42, h: 0.42, fontFace: HF, fontSize: 14, bold: true, color: WHITE, isTextBox: true, margin: 0, align: "center", valign: "middle" });
    s.addText([{ text: st[0], options: { bold: true, breakLine: true, color: NAVY } }, { text: st[1], options: { color: INK } }], { x: 1.1, y, w: W - 1.1 - M, h: 0.92, fontFace: BF, fontSize: 12, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 2 }); });
  s.addNotes("Reflection slide. The project started as a conventional PyTorch optimisation study and ran into a wall: once compilation and BF16 were in, the planner was still dominated by what happens between kernels. That is what pushed me to a megakernel, and the megakernel is what forced the evaluation discipline, because a numeric rewrite of the whole planner needs a paired closed-loop check and an equal-hardware reference."); footer(s, n); }

// 5 Literature review
{ const s = pres.addSlide(); n++; title(s, "Literature review: three lineages, one gap");
  panels(s, [["JEPA world models", "I-JEPA, V-JEPA, DINO-WM, PLDM, V-JEPA 2-AC, LeWM: all plan with CEM-family MPC in latent space. Latency is reported coarsely, if at all (53 s per DINO-WM call, 16 s per V-JEPA 2-AC action, LeWM 'under one second'), and every speed-up changes the model or the planner."],
             ["CEM to iCEM", "CEM (Rubinstein) and PETS/PlaNet made latent CEM-MPC the standard. iCEM adds a decaying population and elite memory: 2.7-22x fewer samples. CEM-GD shows the catch: fewer samples do not mean less time unless the per-iteration cost falls too."],
             ["Kernels and fusion", "Small-batch inference is launch- and latency-bound. FlashAttention fuses an operator; torch.compile and CUDA graphs fuse regions; megakernels (Hazy Llama-1B, FlashFormer, MPK) fuse a whole LLM forward pass into one launch. None fuses an optimiser loop."]], 1.15, 2.9);
  s.addText([{ text: "The gap. ", options: { bold: true, color: NAVY } }, { text: "No work holds a JEPA world model and its planner fixed and measures what the execution of the planner costs; iCEM has run on LeWM only as an evaluated solver; and no kernel fuses a whole sampling-based planning loop with a neural encoder, a transformer rollout, elite selection and distribution updates. GPU MPC (STORM, cuRobo, MPPI-Generic) is fast only with analytic dynamics.", options: {} }],
    { x: M, y: 4.2, w: W - 2 * M, h: 1.1, fontFace: BF, fontSize: 12, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Three lineages. JEPA world models, where planning latency has always been addressed by changing the model. Sampling-based planners, where iCEM cut the sample count, and CEM-GD already showed that sample count is not wall time. And small-batch inference, where the megakernel is the newest answer but only for language models. The gap is the intersection: an exact, model-preserving systems treatment of world-model planning."); footer(s, n); }

// 6 Research questions
{ const s = pres.addSlide(); n++; title(s, "Research questions");
  const rqs = [["RQ1", "How much of the reference planner's latency is redundant work that an exact change removes?"],
               ["RQ2", "Does iCEM make planning faster, by how much, and does it change task success?"],
               ["RQ3", "How much does executing the whole planner as one persistent kernel gain over the best PyTorch implementation of the same planner?"],
               ["RQ4", "Do the megakernel's numerics and the fused encoder preserve accuracy, against FP32 on identical inputs and in closed loop?"]];
  rqs.forEach((r, i) => { const y = 1.2 + i * 0.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: 1.0, h: 0.7, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
    s.addText(r[0], { x: M, y, w: 1.0, h: 0.7, fontFace: HF, fontSize: 16, bold: true, color: WHITE, isTextBox: true, margin: 0, align: "center", valign: "middle" });
    s.addText(r[1], { x: 1.75, y, w: W - 1.75 - M, h: 0.7, fontFace: BF, fontSize: 14, color: INK, isTextBox: true, margin: 0, valign: "middle" }); });
  label(s, "One change per rung, one H100, one container image; primary metric: batch-1 end-to-end plan latency.", M, 5.0, W - 2 * M, GREY, 11);
  s.addNotes("Four questions, one per kind of change: exact, planner-level, execution, numeric. The fourth is the guard rail: none of the speed-ups counts if the robot does worse."); footer(s, n); }

// 7 Method (1): system and ladder
{ const s = pres.addSlide(); n++; title(s, "Method (1): the system and the ladder");
  fitImage(s, "fig_system.png", M, 1.1, 5.6, 3.2);
  body(s, ["R0 reference: CEM 300 x 30, FP32, re-encodes in every cost call.",
           "R1 + encode once per plan, no host sync (exact).",
           "R2 + bucketed iCEM: 300 -> 60 candidates, 5 retained elites (planner).",
           "R2b LeWM-Flash: + compile, TF32, BF16 (best PyTorch).",
           "R3 megakernel: predictor + all 30 iterations in one launch.",
           "R4 + ViT encoder fused: one launch per plan.",
           "Side rung R3-CEM: the megakernel running plain CEM."], 6.35, 1.15, 3.2, 4.0, { size: 11.5 });
  label(s, "LeWM on Push-T: ViT-tiny encoder, 6-layer AdaLN predictor, 5-step rollout, latent goal cost.", M, 4.45, 5.6, GREY, 10);
  s.addNotes("The system on the left: encode the observation and the goal, sample candidates, roll them through the predictor, score, refit, thirty times. The ladder on the right: each rung adds exactly one change, and the side rung prices iCEM inside the kernel."); footer(s, n); }

// 8 Method (2): the megakernel
{ const s = pres.addSlide(); n++; title(s, "Method (2): one kernel per plan");
  fitImage(s, "fig_megakernel.png", M, 1.05, 5.9, 3.9);
  body(s, ["Persistent grid: one CTA per SM, clusters of 8 or 16, grid barriers between iCEM phases; sampling (Philox), top-k and elite updates in the kernel.",
           "Cluster tensor parallelism: each CTA owns heads and MLP slices; row owners reduce and broadcast over distributed shared memory.",
           "Weights as a static stream: pre-swizzled 24 KB panels, cp.async.bulk into a self-refilling ring, no producer warp (keeps 255 registers).",
           "Exact KV reuse; BF16 operands with FP32 accumulation, residual and costs."], 6.6, 1.15, 3.0, 4.0, { size: 11.5 });
  s.addNotes("The kernel. The planner's loop has a static shape once the population schedule is fixed, so the weights it needs and the barriers it crosses are known before it starts. That is what allows a weight stream with no producer warp and a plan with no host involvement."); footer(s, n); }

// 9 Method (3): evaluation
{ const s = pres.addSlide(); n++; title(s, "Method (3): measuring what a robot would see");
  panels(s, [["Batch-1 latency", "The whole replanning call (preprocess, encode, solve, action to host) with CUDA events; 61 one-robot episodes per rung, first dropped; 72-76 plans; median with bootstrap CI and p90."],
             ["300 paired episodes", "Push-T seeds 0-2 x 100 episodes; start states fixed by seed, so every rung sees the same episodes. Wilson intervals, exact McNemar, paired bootstrap of the difference."],
             ["Numerical acceptance", "Before any closed loop: cost error and top-30 overlap against FP32 on identical candidates; the bar is the accepted BF16 autocast path. Encoder: embedding error and cost error."]], 1.2, 2.5, PALE);
  s.addText([{ text: "What 300 pairs can see. ", options: { bold: true, color: NAVY } }, { text: "At the observed 12-13% discordance, a paired difference of about 6 points is detectable at 80% power; 2 points would need about 2,400 pairs. Every accuracy claim in this work is a claim of no detectable change at that resolution.", options: {} }],
    { x: M, y: 3.95, w: W - 2 * M, h: 1.2, fontFace: BF, fontSize: 12, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Evaluation design. Latency is measured where a deployment feels it, end to end, for one robot. Accuracy is measured on identical episodes so that differences are paired, and the power calculation tells you what the claim of 'unchanged' actually means: no change of six points or more."); footer(s, n); }

// 10 Plan as executed
{ const s = pres.addSlide(); n++; title(s, "The plan, and how it ran");
  const ph = [["Phase 0", "LeWM-Flash (L40S)", "Paired 300-episode sweeps of PyTorch-level changes. Accepted: caching, no syncs, compile, TF32, iCEM, BF16. Rejected: warm starts (success fell to 34-42%), whole-loop compile, CEM-GD, static BF16."],
              ["Phase 1", "Megakernel (H100)", "Kernel built against FP32 and BF16 references; five tuning steps, 76.9 -> 41.3 ms solve; 300-episode runs of the reference, LeWM-Flash and the kernel on the same GPU."],
              ["Phase A", "First ladder pass", "Batch-1 latency of R0-R3 and R3-CEM; quality runs of R1 and R2."],
              ["Final", "Fused encoder + ladder", "Encoder in the kernel, parity checked; second batch-1 pass of all seven rungs; quality runs of R3, R3-CEM and R4."]];
  ph.forEach((p, i) => { const x = M + i * 2.3; s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.2, w: 2.15, h: 3.3, fill: { color: i >= 2 ? PALE : LIGHT }, line: { color: i >= 2 ? PALE : LIGHT }, rectRadius: 0.08 });
    s.addText(p[0], { x: x + 0.15, y: 1.3, w: 1.9, h: 0.35, fontFace: HF, fontSize: 15, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(p[1], { x: x + 0.15, y: 1.65, w: 1.9, h: 0.35, fontFace: BF, fontSize: 12, bold: true, color: BLUE, isTextBox: true, margin: 0 });
    s.addText(p[2], { x: x + 0.15, y: 2.05, w: 1.9, h: 2.4, fontFace: BF, fontSize: 10.5, color: INK, isTextBox: true, margin: 0, valign: "top" }); });
  label(s, "Compute: about $31.50 of H100 time against a $35 budget (extended from $10 when the ladder was added). Each batch-1 pass cost $10-12 because the harness rebuilds the simulator and model for every seed.", M, 4.7, W - 2 * M, GREY, 10.5);
  s.addNotes("Four phases. The first predates the H100 work and is why R1 and R2 look the way they do. Say plainly that the budget was extended and nearly used, and that the batch-1 harness is what made it expensive."); footer(s, n); }

// 11 Result 1: the ladder
{ const s = pres.addSlide(); n++; title(s, "Result 1: 622 ms to 46.5 ms on the same GPU");
  fitImage(s, "fig_ladder.png", M, 1.05, 6.3, 3.4);
  body(s, ["Caching: 1.65x, exact.",
           "iCEM in eager PyTorch: 0.92x.",
           "LeWM-Flash: 3.4x over the reference.",
           "Megakernel: 6.9x over eager iCEM, 3.1x over LeWM-Flash.",
           "Fused encoder: 1.28x more; 13.4x in total, p90 47.0 ms.",
           "Megakernel rungs are tight: R4's interquartile range is 46.1-46.8 ms."], 7.0, 1.15, 2.6, 3.9, { size: 11.5 });
  label(s, "Batch-1 end-to-end plan latency, log scale; small points are individual plans, markers are medians.", M, 4.55, 6.3, GREY, 10);
  s.addNotes("Read the chart top to bottom. Caching takes 40 per cent off. iCEM in eager PyTorch makes it slightly worse. Compilation and BF16 bring it to 182. The megakernel takes it to 59, and fusing the encoder to 46.5. Note also the spread: the PyTorch rungs spread over a hundred milliseconds, the megakernel rungs over one."); footer(s, n); }

// 12 Result 2: iCEM
{ const s = pres.addSlide(); n++; title(s, "Result 2: iCEM pays only when execution does");
  fitImage(s, "fig_cost_scaling.png", M, 1.05, 6.0, 2.6);
  card(s, 6.8, 1.15, 1.35, 1.15, "0.92x", "iCEM vs CEM,\neager PyTorch", ORANGE, 24); card(s, 8.25, 1.15, 1.35, 1.15, "1.38x", "iCEM vs CEM,\ninside the kernel", BLUE, 24);
  body(s, ["An eager cost call costs 20 ms at 60 candidates and at 300: it is a few hundred launches, not arithmetic.",
           "Inside the kernel the call scales with the population, so iCEM's 3.19x fewer rollouts become 1.38x (R3-CEM 82.1 -> R3 59.3 ms).",
           "Success unchanged both ways (265 vs 263; 262 vs 261 of 300)."], 6.8, 2.5, 2.8, 2.6, { size: 11 });
  label(s, "One cost call (5-step rollout of every candidate) against the number of candidates, H100.", M, 3.75, 6.0, GREY, 10);
  s.addText("A sample-efficient planner is only as fast as its smallest batch. Measured only in eager PyTorch, iCEM would have been rejected; measured only on the kernel, it would look a third as useful as its evaluation count suggests.", { x: M, y: 4.15, w: 6.0, h: 0.9, fontFace: BF, fontSize: 11.5, color: NAVY, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("This is the result I did not expect. iCEM evaluates three times fewer candidates, and in eager PyTorch it is slightly slower, because each cost call is launch-bound and costs the same at any population size. Only once the execution stops being launch-bound does the population matter. The order of the ladder matters for the accounting."); footer(s, n); }

// 13 Result 3: where the time goes
{ const s = pres.addSlide(); n++; title(s, "Result 3: where the time goes, and how it fell");
  fitImage(s, "fig_breakdown.png", M, 1.05, 4.6, 1.9);
  fitImage(s, "fig_kernel_history.png", M, 3.0, 4.6, 2.1);
  card(s, 5.4, 1.15, 1.35, 1.1, "41.3 ms", "kernel solve\n(Flash: 82.3)", BLUE, 20); card(s, 6.85, 1.15, 1.35, 1.1, "1.3 ms", "fused encoder\n(FP32 torch: 6.9)", BLUE, 20); card(s, 8.3, 1.15, 1.3, 1.1, "0.65 µs", "cluster barrier\n(was 3.6 µs)", NAVY, 20);
  body(s, ["Each tuning step removed latency, not arithmetic: no calls between WGMMAs, relaxed cluster arrives, fast erf, no producer warp.",
           "Every attempt that raised register pressure (peeling, K/V prefetch, owner prefetch) made the kernel slower.",
           "What remains is a 900-layer dependent chain; going far below 40 ms needs fewer cross-SM hand-offs per layer, a redesign."], 5.4, 2.5, 4.2, 2.7, { size: 11 });
  s.addNotes("Where R4's 46.5 milliseconds go: 41 in the solve, 1.3 in the encoder, about 4 in host glue. And how the solve got to 41: the biggest single win was the memory ordering of one barrier, not a GEMM. That is the signature of a latency-bound workload."); footer(s, n); }

// 14 Result 4: accuracy
{ const s = pres.addSlide(); n++; title(s, "Result 4: task success does not move");
  fitImage(s, "fig_paired.png", M, 1.05, 6.3, 3.0);
  body(s, ["Every rung: 260-265 of 300; no paired test near significance (smallest p = 0.50).",
           "Whole ladder, R4 vs R0: 19 won / 20 lost, -0.3 points [-4.3, +3.7].",
           "The exact rung differs in 2 episodes; any new random stream in 32-39: the size of sampling noise.",
           "Kernel cost error vs FP32 matches BF16 autocast (max rel. 0.0005-0.0043); fused encoder closer to FP32 than autocast."], 7.0, 1.15, 2.6, 4.0, { size: 11 });
  label(s, "Grey: episodes only the reference solved; coloured: episodes only the rung solved. Annotation: paired difference, bootstrap 95% CI, exact McNemar p.", M, 4.2, 6.3, GREY, 10);
  s.addNotes("The guard rail held. No step changed success detectably, and the discordance between rungs is the size you get from redrawing the random candidates alone. The honest framing is 'no detectable change at six points', not 'identical'."); footer(s, n); }

// 15 Conclusions and issues
{ const s = pres.addSlide(); n++; title(s, "Conclusions, and what I cannot claim");
  const ans = [["RQ1", "About 40% of the reference latency is redundant work: caching and no host syncs, 622 -> 377 ms, exact."],
               ["RQ2", "It depends on execution: 0.92x in eager PyTorch, 1.38x in the megakernel; success unchanged either way."],
               ["RQ3", "3.1x over the best PyTorch path end to end (2.0x on the solve), 6.9x over eager iCEM, at the same numerics class."],
               ["RQ4", "Yes, within resolution: cost error matches BF16 autocast; -0.3 points [-4.3, +3.7] on 300 paired episodes."]];
  ans.forEach((a, i) => { const y = 1.15 + i * 0.82;
    s.addText(a[0], { x: M, y, w: 0.7, h: 0.55, fontFace: HF, fontSize: 14, bold: true, color: BLUE, isTextBox: true, margin: 0, valign: "top" });
    s.addText(a[1], { x: 1.25, y, w: 4.55, h: 0.78, fontFace: BF, fontSize: 11, color: INK, isTextBox: true, margin: 0, valign: "top" }); });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.1, y: 1.15, w: 3.4, h: 3.9, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  s.addText("Issues and limits", { x: 6.3, y: 1.27, w: 3.0, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  body(s, ["300 pairs resolve about 6 points of success, no less.",
           "PyTorch rungs moved up to 18% between two passes; the kernel rungs under 2%.",
           "One model, one task, one GPU generation; Hopper-only kernel.",
           "Simulation only; only the planning call was optimised.",
           "R2b is bimodal and R4 has one 458 ms outlier, causes not isolated.",
           "Budget: $31.50 of $35, two latency passes, one quality run per rung."], 6.3, 1.72, 3.05, 3.3, { size: 10.5 });
  s.addNotes("Answers to the four questions, and the limits in the same breath. The speed-up is robust; the accuracy claim is bounded by the sample size; the factors are specific to this model and this GPU, even if the mechanism is not."); footer(s, n); }

// 16 Future work
{ const s = pres.addSlide(); n++; title(s, "Future work: towards a research programme");
  const fw = [["Spend the saved latency on planning", "Sweep iterations, population and elites at a few ms per plan: map the latency-success frontier, not one point."],
              ["Fewer hand-offs per layer", "A single-barrier all-reduce, or one layer per cluster with resident (FP8) weights."],
              ["Compose with model-level speed-ups", "Fast-LeWM's parallel prefix prediction would cut the chain from 5 passes per iteration to 1."],
              ["Other world models, other GPUs", "DINO-WM and V-JEPA 2-AC push back towards throughput; Blackwell and Jetson Thor change the primitives; automate with megakernel compilers."],
              ["Evaluate at deployment resolution", "About 2,400 paired episodes for 2 points, now minutes of planning; then a real robot."]];
  fw.forEach((f, i) => { const col = i % 2, row = Math.floor(i / 2); const x = M + col * 4.6, y = 1.15 + row * 1.3;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.05, w: 0.4, h: 0.4, fill: { color: BLUE }, line: { color: WHITE } });
    s.addText(String(i + 1), { x, y: y + 0.05, w: 0.4, h: 0.4, fontFace: HF, fontSize: 13, bold: true, color: WHITE, isTextBox: true, margin: 0, align: "center", valign: "middle" });
    s.addText([{ text: f[0], options: { bold: true, color: NAVY, breakLine: true } }, { text: f[1], options: { color: INK } }], { x: x + 0.55, y, w: 3.9, h: 1.2, fontFace: BF, fontSize: 11.5, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 2 }); });
  s.addText("Together: world-model planning treated as a systems problem, with the evaluation methodology to show that speed did not cost behaviour.", { x: 5.1, y: 3.85, w: 4.4, h: 1.2, fontFace: BF, fontSize: 12, italic: true, color: GREY, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Five directions in the order I would take them. The first is the most immediate payoff: a fast planner can afford a bigger search. The second and third push latency further. The fourth asks whether the result generalises beyond the most favourable model. The fifth turns compute into control."); footer(s, n); }

// 17 Close (dark)
{ const s = pres.addSlide(); s.background = { color: NAVY }; n++;
  s.addText("Take-home", { x: M, y: 0.6, w: W - 2 * M, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText([{ text: "A JEPA planner at batch 1 is bound by the latency of a long chain of tiny steps, not by FLOPs or bandwidth.", options: { bullet: { indent: 16 }, breakLine: true, paraSpaceAfter: 12 } },
             { text: "Sample-efficient planning (iCEM) only pays once execution stops being launch-bound: 0.92x in eager PyTorch, 1.38x in a kernel.", options: { bullet: { indent: 16 }, breakLine: true, paraSpaceAfter: 12 } },
             { text: "One persistent kernel per plan: 622 ms -> 46.5 ms on the same H100, same model, same planner, same task success.", options: { bullet: { indent: 16 } } }],
    { x: M, y: 1.5, w: W - 2 * M, h: 3.0, fontFace: BF, fontSize: 17, color: WHITE, isTextBox: true, margin: 0, valign: "top" });
  s.addText("Paper, kernel, per-call latency records and per-episode outcomes: docs/paper and megakernel/ in the project repository.", { x: M, y: 4.7, w: W - 2 * M, h: 0.5, fontFace: BF, fontSize: 12, color: "CADCFC", isTextBox: true, margin: 0 });
  s.addNotes("Three sentences to end on, then questions."); }

pres.writeFile({ fileName: path.join(__dirname, "..", "..", "presentation.pptx") }).then(f => console.log("wrote", f, n, "slides"));
