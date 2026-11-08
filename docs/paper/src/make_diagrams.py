"""Report diagrams (system, analytics pipeline, kernel structure). Data figures come from solution_pack/lewm_flash.py."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

S = os.path.dirname(os.path.abspath(__file__)); F = f"{S}/figs"; os.makedirs(F, exist_ok=True)
ROOT = os.path.abspath(f"{S}/../../..")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_L, BLUE_D = "#7fb0e8", "#124a8f"
OR_L, OR_D = "#f5a888", "#a8431c"
INK, INK2, GRID, SURF, NAVY = "#0b0b0b", "#52514e", "#e6e6e2", "#ffffff", "#14213d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": SURF})
W = 6.3


def save(fig, name):
    fig.savefig(f"{F}/{name}.png"); plt.close(fig); print("wrote", name)


def box(ax, x, y, w, h, text, fc="#f3f4f6", ec=INK2, fs=8, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, col=INK2, rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=col, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def fig_system():
    fig, ax = plt.subplots(figsize=(W, 3.1)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.9); ax.axis("off")
    box(ax, 0.05, 3.3, 1.55, 0.9, "observation\n224x224 RGB", "#ffffff", fs=7.5); box(ax, 0.05, 2.0, 1.55, 0.9, "goal image\n224x224 RGB", "#ffffff", fs=7.5)
    box(ax, 2.0, 2.45, 1.95, 1.5, "ViT-tiny encoder\n+ projector\n12 layers, d = 192", "#e9f1fc", fs=7.5)
    arrow(ax, 1.6, 3.75, 2.0, 3.45); arrow(ax, 1.6, 2.45, 2.0, 2.85)
    box(ax, 4.35, 3.45, 1.35, 0.6, "z_obs (192)", "#ffffff", fs=7.5); box(ax, 4.35, 2.35, 1.35, 0.6, "z_goal (192)", "#ffffff", fs=7.5)
    arrow(ax, 3.95, 3.5, 4.35, 3.75); arrow(ax, 3.95, 2.9, 4.35, 2.65)
    box(ax, 5.95, 3.55, 2.75, 1.05, "sample N action sequences\n(5 blocks x 10 dims)", "#fdeee7", fs=7)
    box(ax, 5.95, 1.95, 2.75, 1.15, "predictor rollout:\n6-layer AdaLN transformer,\n5 autoregressive passes", "#e9f1fc", fs=7)
    box(ax, 5.95, 0.35, 2.75, 1.1, "cost = ||z_5 - z_goal||^2\ntop-k elites -> mean, std", "#fdeee7", fs=7)
    arrow(ax, 7.325, 3.55, 7.325, 3.1); arrow(ax, 7.325, 1.95, 7.325, 1.45)
    arrow(ax, 8.7, 0.9, 8.7, 4.05, rad=0.55, col=ORANGE, lw=1.1)
    ax.text(9.7, 2.5, "x 30\niters", ha="center", va="center", fontsize=8, color=OR_D, weight="bold")
    arrow(ax, 5.7, 3.75, 5.95, 2.5, rad=-0.15); arrow(ax, 5.7, 2.65, 5.95, 0.9, rad=0.1)
    ax.text(0.05, 1.45, "once per plan: encode 2 images", fontsize=7.5, color=INK2, va="top")
    ax.text(0.05, 0.95, "per plan: 30 iterations x N candidates x\n5 predictor passes; robot executes 5 blocks", fontsize=7.5, color=INK2, va="top")
    save(fig, "fig_system")


def fig_megakernel():
    fig, ax = plt.subplots(figsize=(W, 4.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(0.05, 6.8, "(a) One launch per plan: persistent kernel, one CTA per SM, grid = clusters x CTAs", fontsize=8, weight="bold", color=NAVY)
    segs = [("encoder\n(62 sub-phases)", 1.75, BLUE_D, "white"), ("iter 1", 1.05, BLUE, "white"), ("iter 2", 1.05, BLUE, "white"), ("...", 0.6, "#ffffff", INK),
            ("iter 30", 1.05, BLUE, "white"), ("mean, std\nto host", 1.3, INK2, "white")]
    x = 0.1
    for t, w, c, tc in segs:
        box(ax, x, 5.65, w, 0.85, t, c, ec=INK2, fs=7.5, tc=tc); x += w + 0.1
    ax.text(0.05, 5.15, "(b) One iCEM iteration (grid barrier between phases)", fontsize=8, weight="bold", color=NAVY)
    ph = [("M1: Philox sampling,\nelite injection,\naction encoder", 2.3, "#fdeee7"), ("M2: AdaLN\nmodulation table", 1.75, "#fdeee7"),
          ("rollout passes 0-4,\ntensor-parallel\nper cluster", 2.3, "#e9f1fc"), ("T: cost, per-env\ntop-k, mean / std", 2.3, "#fdeee7")]
    x = 0.1
    for i, (t, w, c) in enumerate(ph):
        box(ax, x, 3.75, w, 1.1, t, c, fs=7.3)
        if i < len(ph) - 1: arrow(ax, x + w, 4.3, x + w + 0.15, 4.3)
        x += w + 0.15
    ax.text(0.05, 3.25, "(c) Inside a cluster, per transformer layer (8 or 16 CTAs share one 64-row chunk)", fontsize=8, weight="bold", color=NAVY)
    steps = [("own heads:\nqkv WGMMA,\nattention,\nout-proj", "#e9f1fc"), ("bf16\npartial to\nDSMEM", "#ffffff"), ("row owners:\nreduce, +res,\nLN, AdaLN", "#fdeee7"),
             ("broadcast\nnext operand\n(DSMEM)", "#ffffff"), ("own MLP\nslices,\nGELU", "#e9f1fc"), ("reduce,\nbroadcast", "#fdeee7")]
    x = 0.1; w = 1.42
    for i, (t, c) in enumerate(steps):
        box(ax, x, 1.45, w, 1.45, t, c, fs=7)
        if i < len(steps) - 1: arrow(ax, x + w, 2.17, x + w + 0.2, 2.17)
        x += w + 0.2
    ax.text(0.1, 1.0, "Weights: 1,472 pre-swizzled 24 KB bf16 panels (with encoder). Each warpgroup's thread 0 refills its 2-slot shared-memory ring\n"
            "with cp.async.bulk from a static schedule, so weight prefetch runs through op, layer, pass and barrier boundaries.", fontsize=7, color=INK2, va="top")
    save(fig, "fig_megakernel")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W, 2.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.0); ax.axis("off")
    st = [("1 Collect", "latency records,\nepisodes,\ncounters", "#ffffff"),
          ("2 Prepare", "tidy tables,\nwarm-up removal,\noutlier flags", "#ffffff"),
          ("3 Discover", "stage profile,\nscaling model,\nbottleneck", "#e9f1fc"),
          ("4 Design", "caching, iCEM,\nmegakernel,\nfused encoder", "#fdeee7"),
          ("5 Evaluate", "latency CIs,\npaired success,\nnumerical parity", "#e9f1fc"),
          ("6 Deploy", "planner plug-in,\nsolution program", "#ffffff")]
    w, step = 1.4, 1.68
    for i, (t, d, c) in enumerate(st):
        x = 0.1 + i * step
        box(ax, x, 1.2, w, 0.62, t, c, fs=8, bold=True)
        ax.text(x + w / 2, 1.05, d, ha="center", va="top", fontsize=6.4, color=INK2, linespacing=1.25)
        if i < len(st) - 1: arrow(ax, x + w, 1.51, x + step, 1.51)
    arrow(ax, 0.1 + 4 * step + w / 2, 1.82, 0.1 + 2 * step + w / 2, 1.82, col=ORANGE, rad=0.45)
    ax.text(0.1 + 3 * step + w / 2, 2.72, "every design change is re-profiled and re-evaluated before it is kept", ha="center", fontsize=6.8, color=OR_D)
    save(fig, "fig_pipeline")


if __name__ == "__main__":
    for f in [fig_system, fig_megakernel, fig_pipeline]:
        f()
