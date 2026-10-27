// Presentation deck for the report "Planning at the Speed of Control".
// Usage: NODE_PATH=<node_modules with pptxgenjs> node make_deck.js   (figures from ../figs)
const pptxgen = require("pptxgenjs"); const fs = require("fs"); const path = require("path");
const pres = new pptxgen(); pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Antoine Nguyen"; pres.title = "Planning at the Speed of Control";
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
  s.addText("Profiling and fusing world-model robot planning into a single GPU kernel", { x: M, y: 2.0, w: W - 2 * M, h: 0.8, fontFace: BF, fontSize: 18, color: "CADCFC", isTextBox: true, margin: 0 });
  s.addText([{ text: "Antoine Nguyen  ·  24433599", options: { breakLine: true, bold: true } }, { text: "Project presentation  ·  [Faculty / School], University of Technology Sydney", options: { breakLine: true } }, { text: "Supervisor: [supervisor name]  ·  October 2026", options: {} }],
    { x: M, y: 3.9, w: W - 2 * M, h: 1.0, fontFace: BF, fontSize: 13, color: WHITE, isTextBox: true, margin: 0, paraSpaceAfter: 4 });
  s.addNotes("This study profiles the inference latency of a world-model planner, identifies its bottlenecks from measured data, and evaluates a fused GPU implementation that removes them without retraining the model."); }

// 2 The project in one slide
{ const s = pres.addSlide(); n++; title(s, "Overview");
  body(s, ["LeWM is a 15M-parameter world model that plans by evaluating thousands of predicted action outcomes. One plan takes 622 ms on an H100 GPU.",
           "Research question: where is the planning latency spent, and how much can be removed without retraining the model or reducing task success?",
           "Method: compute profiling of 11,942 timed calls, a fitted latency-scaling model, solution design, and paired statistical evaluation on 900 episode outcomes.",
           "Result: latency was dominated by execution overhead rather than computation. A single persistent GPU kernel per plan reduces latency to 46.5 ms with no detectable change in task success."],
    M, 1.15, 5.6, 3.9, { size: 14 });
  card(s, 6.5, 1.2, 1.5, 1.15, "622 ms", "reference planner\n(one robot, H100)", GREY, 24); card(s, 8.1, 1.2, 1.5, 1.15, "46.5 ms", "LeWM-MK,\none GPU launch", BLUE, 24);
  card(s, 6.5, 2.55, 1.5, 1.15, "13.4×", "faster than the\nreference", BLUE, 26); card(s, 8.1, 2.55, 1.5, 1.15, "−0.3 pts", "success vs reference,\n300 paired episodes", NAVY, 22);
  label(s, "Medians of end-to-end planning latency; 95% CI of the speed-up 13.0–13.5×", 6.5, 3.85, 3.2, GREY, 10);
  s.addNotes("Summary of the study: reference latency 622 ms, final latency 46.5 ms, a 13.4-fold reduction, and no detectable change in task success. The following slides present the profile, the design derived from it, and the evaluation."); footer(s, n); }

// 3 Motivation
{ const s = pres.addSlide(); n++; title(s, "Motivation");
  body(s, ["A world model predicts future states; a planner evaluates many candidate action sequences with it and executes the best.",
           "A controller must finish each plan within its control period: 100 ms at 10 Hz, 50 ms at 20 Hz, for sensing, planning and acting together.",
           "LeWM made world-model planning cheap by shrinking the model to 15M parameters. Its released planner still takes 622 ms per plan.",
           "The arithmetic content of a plan corresponds to milliseconds of GPU time; the remaining latency is not attributable to computation."],
    M, 1.15, 5.4, 3.9, { size: 14 });
  card(s, 6.3, 1.2, 1.55, 1.1, "15M", "parameters in\nthe whole model", NAVY); card(s, 8.0, 1.2, 1.55, 1.1, "9,000", "rollouts per\nreference plan", NAVY);
  card(s, 6.3, 2.5, 1.55, 1.1, "900", "dependent layer\nevaluations per plan", NAVY); card(s, 8.0, 2.5, 1.55, 1.1, "16–53 s", "per plan for larger\nworld models", NAVY, 24);
  s.addNotes("Closed-loop control requires planning latency below the control period. Larger world models require tens of seconds per plan; LeWM requires 622 ms on an H100, although its arithmetic content corresponds to milliseconds."); footer(s, n); }

// 4 Aim, objectives, analytics link
{ const s = pres.addSlide(); n++; title(s, "Aim and objectives");
  const rqs = [["O1", "Profile: build a stage-by-stage compute profile of the reference planner from measured data, and identify the bottleneck."],
               ["O2", "Design and implement: remove the bottlenecks without retraining the model or shrinking the planner's search budget."],
               ["O3", "Evaluate: latency and task success against the reference and a strong optimised baseline, on the same GPU, with paired tests."],
               ["O4", "Deploy: a planner that plugs into the existing harness, and one program that regenerates every result from the data."]];
  rqs.forEach((r, i) => { const y = 1.15 + i * 0.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: 0.9, h: 0.65, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
    s.addText(r[0], { x: M, y, w: 0.9, h: 0.65, fontFace: HF, fontSize: 16, bold: true, color: WHITE, isTextBox: true, margin: 0, align: "center", valign: "middle" });
    s.addText(r[1], { x: 1.6, y, w: W - 1.6 - M, h: 0.65, fontFace: BF, fontSize: 13.5, color: INK, isTextBox: true, margin: 0, valign: "middle" }); });
  s.addText([{ text: "Data analytics and AI. ", options: { bold: true, color: NAVY } }, { text: "The object is an AI system (a self-supervised world model and its search algorithm); the method is an analytics pipeline over its measured behaviour: collection, preparation, pattern discovery with a fitted latency model, statistical evaluation and deployment.", options: {} }],
    { x: M, y: 4.6, w: W - 2 * M, h: 0.7, fontFace: BF, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Four objectives, one per stage of the study. The object of study is an AI system; the method is an analytics pipeline over its measured behaviour: data collection and preparation, pattern discovery with a fitted latency model, and statistical evaluation."); footer(s, n); }

// 5 Background
{ const s = pres.addSlide(); n++; title(s, "Background and research gap");
  panels(s, [["World models that plan", "JEPA models (I-JEPA, DINO-WM, V-JEPA 2, LeWM) predict in a learned representation and plan by search. Planning time is reported coarsely and only ever cut by changing the model."],
             ["Sampling-based search", "CEM samples, keeps the best, refits, repeats. iCEM shrinks the population and reuses the best: 2.7–22× fewer samples. But fewer samples need not mean less time (CEM-GD)."],
             ["Fast GPU inference", "Small models at batch 1 are bound by launch overhead and data movement. Remedies: CUDA Graphs, compilers, FlashAttention, and megakernels: one GPU program for a whole computation."]], 1.15, 2.85);
  s.addText([{ text: "Research gap. ", options: { bold: true, color: NAVY } }, { text: "No prior work has profiled where a world-model planner's latency is spent, measured whether sample-efficient search saves time in practice, or fused a whole planning loop, with a neural encoder, rollouts, scoring and elite selection, into one GPU program.", options: {} }],
    { x: M, y: 4.2, w: W - 2 * M, h: 1.0, fontFace: BF, fontSize: 12, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Three bodies of work are relevant: JEPA world models, in which planning latency has been reduced only by changing the model; sampling-based search, in which iCEM reduces sample counts but sample count does not determine latency; and GPU inference, in which megakernels have so far been applied only to language models."); footer(s, n); }

// 6 Method
{ const s = pres.addSlide(); n++; title(s, "Methodology");
  fitImage(s, "fig_pipeline.png", M, 1.05, W - 2 * M, 2.0);
  panels(s, [["Data", "11,942 timed policy calls (61 one-robot episodes × 3 systems + 4 probes), 900 episode outcomes, telemetry, component timings."],
             ["Analysis", "Warm-up removal and outlier flags; stage decomposition; a fitted model latency = a + b·N separating fixed overhead from work."],
             ["Evaluation", "Medians with bootstrap CIs; 300 paired episodes, McNemar and paired bootstrap; numerical error against FP32."]], 3.2, 1.95, PALE, 3.1, 2.9, 10.5);
  s.addNotes("Methodology. Timings and outcomes are collected, prepared into tidy tables and analysed to locate the latency; the design follows from the analysis and is evaluated with the same instruments. Probes are configurations that differ from a system in one respect and isolate a single effect. Each design change was re-profiled before adoption."); footer(s, n); }

// 7 Profile
{ const s = pres.addSlide(); n++; title(s, "Compute profile of the reference planner");
  fitImage(s, "fig_profile.png", M, 1.05, 6.0, 2.5);
  body(s, ["33%: re-encoding the same two images 60 times per plan (telemetry: 60 encoder calls, 2 distinct images).",
           "59%: the search, 12.3 ms per iteration, 409 µs per layer for about 3 µs of arithmetic.",
           "7%: CPU synchronisation between iterations.",
           "Under 1%: the policy code around the planner."], 6.7, 1.15, 2.9, 3.9, { size: 11.5 });
  label(s, "Top bar: reference planner, by stage. Bottom bar: LeWM-MK (Section 6). Stages without a direct measurement attributed by difference.", M, 3.75, 6.0, GREY, 10);
  s.addNotes("Compute profile. One third of the latency recomputes image embeddings that are constant within a plan. Most of the remainder is the search, at approximately 409 microseconds per layer evaluation against approximately 3 microseconds of arithmetic."); footer(s, n); }

// 8 Pattern
{ const s = pres.addSlide(); n++; title(s, "Latency scaling analysis");
  fitImage(s, "fig_scaling.png", M, 1.05, 6.0, 2.6);
  card(s, 6.8, 1.15, 1.35, 1.15, "99%", "of an eager call\nis fixed cost", ORANGE, 26); card(s, 8.25, 1.15, 1.35, 1.15, "1.01×", "predicted iCEM\ngain, eager", ORANGE, 24);
  body(s, ["Fitted model: 20.0 ms fixed + 0.10 ms per 100 candidates. Latency is effectively independent of workload.",
           "Reducing the candidate count (iCEM) therefore yields no gain: predicted 1.01×, observed 0.92×.",
           "A gain requires an execution model with low fixed cost per call."], 6.8, 2.5, 2.8, 2.6, { size: 11 });
  label(s, "Latency of one scoring call vs number of candidates, with fitted lines a + b·N.", M, 3.75, 6.0, GREY, 10);
  s.addNotes("Scaling analysis. A linear model of latency against workload shows that the eager planner's cost is almost entirely fixed, consistent with several hundred kernel launches from Python per call. This also explains the absence of a gain from iCEM under eager execution: it reduces work that is not the dominant cost."); footer(s, n); }

// 9 Solution
{ const s = pres.addSlide(); n++; title(s, "Solution design: LeWM-MK");
  fitImage(s, "fig_megakernel.png", M, 1.05, 5.9, 3.9);
  body(s, ["Encode each image once; cache the goal for the episode.",
           "iCEM search: 2,820 instead of 9,000 imagined futures.",
           "Whole plan in one persistent kernel: sampling, rollouts, scoring, elite selection, encoder; the CPU waits once.",
           "Small model spread across a 16-processor cluster; partial results exchanged on chip; barrier 3.6 → 0.65 µs.",
           "Weights streamed ahead of use; BF16 inputs, FP32 accumulation."], 6.6, 1.15, 3.0, 4.0, { size: 11.5 });
  s.addNotes("Each design element addresses one finding: single encoding removes the redundant third; iCEM reduces work once fixed overhead is removed; the persistent kernel removes launches and CPU round trips; cluster-level parallelism and weight streaming minimise the latency of each step of the dependent chain."); footer(s, n); }

// 10 Latency results
{ const s = pres.addSlide(); n++; title(s, "Results: planning latency");
  fitImage(s, "fig_latency.png", M, 1.1, 6.2, 2.2);
  card(s, 6.9, 1.15, 1.3, 1.1, "13.4×", "vs reference\n[13.0, 13.5]", BLUE, 24); card(s, 8.3, 1.15, 1.3, 1.1, "3.9×", "vs optimised\nPyTorch", BLUE, 24);
  body(s, ["90th percentile 47.0 ms; interquartile range 0.7 ms wide.",
           "Fits a 20 Hz control loop at the median.",
           "Search now 46 µs per layer (was 409); synchronisation about 7% of it."], 6.9, 2.45, 2.7, 2.6, { size: 11 });
  label(s, "End-to-end planning latency, log scale; points are plans, diamonds medians. Baseline already uses compile, CUDA Graphs and BF16.", M, 3.45, 6.2, GREY, 10);
  s.addNotes("Latency is reduced 13.4-fold relative to the reference and 3.9-fold relative to an optimised PyTorch implementation using compilation, CUDA Graphs and BF16. Latency variability is also reduced: the interquartile range is 0.7 ms."); footer(s, n); }

// 11 Success and fidelity
{ const s = pres.addSlide(); n++; title(s, "Results: task success and numerical fidelity");
  fitImage(s, "fig_success.png", M, 1.1, 6.2, 1.9);
  body(s, ["261 / 261 / 260 successes of 300 for reference, optimised PyTorch and LeWM-MK.",
           "LeWM-MK vs reference: 19 won, 20 lost; −0.3 points [−4.3, +3.7]; McNemar p = 1.00.",
           "Discordant episodes reflect stochastic search; the optimised PyTorch planner is discordant on 36.",
           "Numerical error vs FP32: 0.06–0.32%, matching accepted BF16 (0.05–0.40%)."], 6.9, 1.15, 2.7, 4.0, { size: 11 });
  label(s, "Push-T success on the same 300 episodes, Wilson 95% intervals. 300 pairs detect differences of about 6 points.", M, 3.2, 6.2, GREY, 10);
  s.addNotes("No detectable change in task success at the resolution of 300 paired episodes (approximately 6 percentage points at 80% power). Numerical error relative to FP32 is within the range of the accepted BF16 implementation."); footer(s, n); }

// 12 Conclusions and limits
{ const s = pres.addSlide(); n++; title(s, "Conclusions and limitations");
  const ans = [["O1", "The reference's time was overhead: 33% redundant encoding, a search 99% fixed cost per call, 7% CPU waits."],
               ["O2", "LeWM-MK removes each bottleneck with the model and search budget unchanged."],
               ["O3", "46.5 ms per plan: 13.4× the reference, 3.9× optimised PyTorch; success unchanged (−0.3 pts)."],
               ["O4", "Planner plug-in; one program regenerates every result from the data."]];
  ans.forEach((a, i) => { const y = 1.15 + i * 0.82;
    s.addText(a[0], { x: M, y, w: 0.7, h: 0.55, fontFace: HF, fontSize: 14, bold: true, color: BLUE, isTextBox: true, margin: 0, valign: "top" });
    s.addText(a[1], { x: 1.25, y, w: 4.55, h: 0.78, fontFace: BF, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" }); });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.1, y: 1.15, w: 3.4, h: 3.9, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  s.addText("Limitations", { x: 6.3, y: 1.27, w: 3.0, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  body(s, ["300 paired episodes resolve differences of about 6 percentage points.",
           "Some profile stages are attributed by difference.",
           "One model, one task, one GPU type; H100-only kernel.",
           "Simulation only; only the planning call was optimised.",
           "PyTorch latency varied by up to 18% between passes; kernel latency by about 2%."], 6.3, 1.72, 3.05, 3.3, { size: 11 });
  s.addNotes("All four objectives were met. The latency reduction is large and reproducible; the task-success result is bounded by sample size; the measured values are specific to this model and GPU, while the identified mechanism is general."); footer(s, n); }

// 13 Future work
{ const s = pres.addSlide(); n++; title(s, "Future work");
  const fw = [["Search budget versus task success", "Larger iteration and candidate budgets remain within a 100 ms control period; characterise the latency-success trade-off."],
              ["Reducing the dependent computation chain", "Combination with single-pass rollout models; layer-per-cluster pipelining."],
              ["Generalisation to other models and hardware", "Larger world models, embedded robot computers, automatic megakernel compilation."],
              ["Higher-resolution and physical evaluation", "Approximately 2,500 paired episodes to resolve 2 percentage points; evaluation on a physical robot."]];
  fw.forEach((f, i) => { const col = i % 2, row = Math.floor(i / 2); const x = M + col * 4.6, y = 1.2 + row * 1.6;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.05, w: 0.4, h: 0.4, fill: { color: BLUE }, line: { color: WHITE } });
    s.addText(String(i + 1), { x, y: y + 0.05, w: 0.4, h: 0.4, fontFace: HF, fontSize: 13, bold: true, color: WHITE, isTextBox: true, margin: 0, align: "center", valign: "middle" });
    s.addText([{ text: f[0], options: { bold: true, color: NAVY, breakLine: true } }, { text: f[1], options: { color: INK } }], { x: x + 0.55, y, w: 3.9, h: 1.3, fontFace: BF, fontSize: 12.5, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 2 }); });
  s.addNotes("Four directions: the latency-success trade-off of the search budget, further latency reduction, generalisation to other models and hardware, and higher-resolution and physical evaluation."); footer(s, n); }

// 14 Summary (dark)
{ const s = pres.addSlide(); s.background = { color: NAVY }; n++;
  s.addText("Summary", { x: M, y: 0.6, w: W - 2 * M, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText([{ text: "The reference planner's latency was dominated by execution overhead, not arithmetic.", options: { bullet: { indent: 16 }, breakLine: true, paraSpaceAfter: 12 } },
             { text: "A linear latency-scaling model identified the bottleneck regime and predicted when iCEM reduces latency.", options: { bullet: { indent: 16 }, breakLine: true, paraSpaceAfter: 12 } },
             { text: "A single persistent GPU kernel per plan reduces latency from 622 ms to 46.5 ms on the same H100, with the same model and no detectable change in task success.", options: { bullet: { indent: 16 } } }],
    { x: M, y: 1.5, w: W - 2 * M, h: 3.0, fontFace: BF, fontSize: 17, color: WHITE, isTextBox: true, margin: 0, valign: "top" });
  s.addText("Report, datasets and solution program: docs/paper and solution_pack/ in the project repository.", { x: M, y: 4.7, w: W - 2 * M, h: 0.5, fontFace: BF, fontSize: 12, color: "CADCFC", isTextBox: true, margin: 0 });
  s.addNotes("Summary of findings."); }

pres.writeFile({ fileName: path.join(__dirname, "..", "..", "presentation.pptx") }).then(f => console.log("wrote", f, n, "slides"));
