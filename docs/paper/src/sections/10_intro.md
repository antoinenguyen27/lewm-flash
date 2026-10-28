# 1. Introduction

## 1.1 Motivation

A world model is an AI model that predicts what will happen next. Given a camera image of a scene and a proposed action, it predicts the next state of the scene, and by chaining predictions it can imagine the consequences of a whole sequence of actions. A robot that owns such a model can plan: it imagines many candidate action sequences, scores each by how close its imagined outcome comes to a goal, and executes the best. This is how a growing family of robot-learning systems works, and the family built on the joint-embedding predictive architecture (JEPA) is among the most promising, because it learns to predict in a compact learned representation rather than in pixels (Assran et al., 2023; Zhou et al., 2025; Assran et al., 2025).

Planning by imagination is expensive. Every plan evaluates thousands of imagined futures, and for the large world models of the field a single plan has taken from 16 seconds to nearly a minute (Zhou et al., 2025; Assran et al., 2025). LeWorldModel (LeWM) was designed to fix that. It is a 15-million-parameter JEPA, small enough to train on one GPU in a few hours, that encodes each image into a single 192-number vector and plans up to 48 times faster than its predecessors, "in under one second" (Maes, Le Lidec, Scieur, et al., 2026).

Under one second is fast for a world model and slow for a robot. A controller that replans as it moves must finish each plan within its control period; a robot arm controlled ten times per second has 100 milliseconds for sensing, planning and acting together. Measured on one H100 GPU, LeWM's released planner takes 622 ms per plan. The model is so small that the arithmetic of a plan should take a few milliseconds, so the remaining time must be going somewhere else. Finding out where, and removing it, is the subject of this report.

## 1.2 Aim and objectives

The aim is to reduce the latency of one LeWM plan for one robot on one GPU to the time scale of control, without retraining the model and without reducing task success. Four objectives make the aim measurable.

- **O1. Profile.** Collect and analyse latency data to build a stage-by-stage compute profile of the reference planner and identify what limits its speed.
- **O2. Design and implement.** Design an inference pipeline that removes the bottlenecks found in O1, keeping the trained model's weights and the planner's search budget (horizon, iterations, elite count) unchanged.
- **O3. Evaluate.** Measure the new pipeline's end-to-end latency and closed-loop task success against the reference and against a strong optimised PyTorch implementation on the same hardware, with paired statistical tests.
- **O4. Deploy.** Package the solution as a planner that plugs into the existing evaluation harness, and the analysis as a single program that regenerates every result from the collected data.

## 1.3 Relevance to data analytics and AI

The project sits where AI systems meet data analytics. The object of study is an AI system: a self-supervised world model and the search algorithm that uses it to make decisions. The method is an analytics pipeline applied to that system's behaviour. Its data are measurements: 11,942 timed policy calls, 900 episode outcomes, planner telemetry and component micro-benchmarks. The pipeline follows the standard steps of an analytics project, from data collection, preparation and preprocessing, through pattern discovery and modelling, to evaluation and deployment (Section 3.1). The patterns it discovers drive the engineering. A fitted latency model distinguishes work that scales with the number of candidate plans from fixed overhead that does not, and that distinction decides which optimisations can pay. The evaluation is statistical, with bootstrap confidence intervals, paired tests and a power analysis, because a faster AI system is only useful if it is shown to behave the same. And the deployed artefact is an AI inference engine, the part of an AI system that runs in production.

## 1.4 Scope and structure

The study covers one model (LeWM's released Push-T checkpoint), one task (Push-T, pushing a T-shaped block into a target pose), one GPU type (NVIDIA H100) and one robot per GPU, which is the deployment case. The model is used as released; no training is involved. Section 2 reviews the background. Section 3 describes the methodology. Section 4 reports the compute-profile analysis, Section 5 the solution designed from it, and Section 6 its evaluation. Section 7 discusses the results and their limitations, and Section 8 concludes with future work. The appendices give the data dictionary, the solution pack, the configuration and the compute cost.
