"""Paper figures (static, print), in the palette of the reference thesis. Data: data/ladder_stats.json and the committed
megakernel records in results/megakernel. Run compute_stats.py first."""
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
ST = json.load(open(f"{S}/data/ladder_stats.json"))
RUNG = {  # label, colour, marker
    "r0": ("R0  Reference LeWM (CEM, eager FP32)", INK2, "o"),
    "r1": ("R1  + cached embeddings, no CPU sync", OR_L, "o"),
    "r2": ("R2  + iCEM (eager FP32)", ORANGE, "o"),
    "r2b": ("R2b LeWM-Flash (compile, TF32, BF16)", OR_D, "s"),
    "r3cem": ("R3-CEM megakernel, plain CEM", BLUE_L, "D"),
    "r3": ("R3  + megakernel (predictor + iCEM)", BLUE, "D"),
    "r4": ("R4  + fused encoder (one launch)", BLUE_D, "D"),
}
ORDER = ["r0", "r1", "r2", "r2b", "r3cem", "r3", "r4"]


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


def fig_icem_schedule():
    from math import ceil
    cem = [300] * 30
    ic = []
    for k in range(30):
        c = max(ceil(300 * 1.25 ** (-k)), 60); c = min(300, ceil(c / 60) * 60); ic.append(c)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(1, 31)
    ax.fill_between(x, cem, step="mid", color=INK2, alpha=0.12); ax.step(x, cem, where="mid", color=INK2, lw=1.4, label=f"CEM: 300 x 30 = {sum(cem):,} evaluations")
    ax.fill_between(x, ic, step="mid", color=BLUE, alpha=0.25); ax.step(x, ic, where="mid", color=BLUE, lw=1.6, label=f"bucketed iCEM: {sum(ic):,} evaluations (3.19x fewer)")
    ax.set_xlabel("optimiser iteration"); ax.set_ylabel("candidate sequences"); ax.set_ylim(0, 330); ax.set_xlim(0.5, 30.5)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.85))
    save(fig, "fig_icem_schedule")


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


def fig_cost_scaling():
    T = json.load(open(f"{ROOT}/results/megakernel/torch_baseline_h100.json")); K = json.load(open(f"{ROOT}/results/megakernel/parity_final_kernel_cl16_h100.json"))
    ns = [60, 120, 180, 240, 300]
    e = [T["eager_fp32"]["get_cost_ms_by_samples"][str(n)]["p50_ms"] for n in ns]
    f = [T["flash_compile_bf16"]["get_cost_ms_by_samples"][str(n)]["p50_ms"] for n in ns]
    k = [K["mk_cost_call_ms"][str(n)]["p50_ms"] for n in ns]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    for v, c, m, l in [(e, INK2, "o", "eager FP32 torch (R0-R2)"), (f, OR_D, "s", "LeWM-Flash: compile + TF32 + BF16 (R2b)"), (k, BLUE, "D", "megakernel, one cost call per launch")]:
        ax.plot(ns, v, color=c, marker=m, ms=5, lw=1.4, mec="white", mew=0.7, label=l)
    ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"]); ax.set_ylim(0.9, 30)
    ax.set_xticks(ns); ax.set_xlabel("candidate sequences per cost call"); ax.set_ylabel("one cost call, 5-step rollout (ms)")
    ax.grid(axis="y", color=GRID, lw=0.6, which="both"); ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.62))
    save(fig, "fig_cost_scaling")


def fig_ladder():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ys = np.arange(len(ORDER))[::-1]
    for y, r in zip(ys, ORDER):
        lab, col, mk = RUNG[r]; L = ST["latency"][r]
        pl = np.array(L["plans"]); jit = (np.random.default_rng(1).random(pl.size) - 0.5) * 0.35
        ax.scatter(pl, y + jit, s=4, color=col, alpha=0.45, lw=0, zorder=2)
        ax.plot([L["p50"]], [y], marker=mk, color=col, ms=7, mec="white", mew=0.8, zorder=3)
        ax.text(1250, y, f"{L['p50']:.1f}", va="center", ha="right", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_yticks(ys); ax.set_yticklabels([RUNG[r][0] for r in ORDER], fontsize=8)
    ax.text(1250, len(ORDER) - 0.35, "p50 (ms)", ha="right", fontsize=7.5, color=INK2)
    ax.set_xlim(30, 1300); ax.set_xticks([30, 50, 100, 200, 500, 1000]); ax.set_xticklabels(["30", "50", "100", "200", "500", "1,000"])
    ax.set_xlabel("end-to-end plan latency, one robot (ms, log scale)"); ax.grid(axis="x", color=GRID, lw=0.6, which="both")
    save(fig, "fig_ladder")


def fig_success():
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ys = np.arange(len(ORDER))[::-1]
    for y, r in zip(ys, ORDER):
        lab, col, mk = RUNG[r]; s = ST["success"][r]
        ax.plot(s["wilson"], [y, y], color=col, lw=2, solid_capstyle="round"); ax.plot([s["rate"]], [y], marker=mk, color=col, ms=7, mec="white", mew=0.8)
        ax.text(s["wilson"][1] + 0.004, y, f"{s['k']}/{s['n']}", va="center", fontsize=8)
    r0 = ST["success"]["r0"]["rate"]; ax.axvline(r0, color=INK2, ls="--", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([RUNG[r][0] for r in ORDER], fontsize=8); ax.set_xlim(0.80, 0.95)
    ax.set_xlabel("Push-T success over 300 paired episodes (Wilson 95% interval)"); ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "fig_success")


def fig_paired():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    rs = [r for r in ORDER if r != "r0"]; ys = np.arange(len(rs))[::-1]
    for y, r in zip(ys, rs):
        P = ST["paired"][f"{r}_vs_r0"]; col = RUNG[r][1]
        ax.barh(y, -P["b_only"], color="#9a9a96", height=0.6); ax.barh(y, P["a_only"], color=col, height=0.6)
        ax.text(-P["b_only"] - 0.6, y, str(P["b_only"]), va="center", ha="right", fontsize=7.5); ax.text(P["a_only"] + 0.6, y, str(P["a_only"]), va="center", fontsize=7.5)
        ax.text(27, y, f"{100*P['delta']:+.1f} pts [{100*P['ci'][0]:+.1f}, {100*P['ci'][1]:+.1f}], p = {P['mcnemar_p']:.2f}", va="center", fontsize=7.5)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(ys); ax.set_yticklabels([RUNG[r][0] for r in rs], fontsize=8); ax.set_xlim(-26, 50)
    ax.set_xticks([-20, -10, 0, 10, 20]); ax.set_xticklabels(["20", "10", "0", "10", "20"])
    ax.set_xlabel("discordant episodes: reference only (left)  |  rung only (right)")
    save(fig, "fig_paired")


def fig_breakdown():
    fig, ax = plt.subplots(figsize=(W, 1.9))
    rows = [("R3  megakernel, torch encoder", [("ViT encoder (torch FP32)", 6.9, ORANGE), ("megakernel solve", 41.3, BLUE), ("rest", 59.3 - 48.2, "#b7b7b2")]),
            ("R4  fused encoder", [("fused encoder (in-kernel)", 1.3, BLUE_D), ("megakernel solve", 41.3, BLUE), ("rest", 46.5 - 42.6, "#b7b7b2")])]
    for y, (lab, parts) in zip([1, 0], rows):
        x = 0
        for name, v, c in parts:
            ax.barh(y, v, left=x, color=c, height=0.55, edgecolor="white", lw=0.8)
            if v > 3: ax.text(x + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="white")
            x += v
        ax.text(x + 0.6, y, f"{x:.1f} ms", va="center", fontsize=8)
    ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.set_xlim(0, 66); ax.set_xlabel("batch-1 plan latency (ms, p50)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="torch FP32 encoder"), Patch(color=BLUE_D, label="fused encoder"), Patch(color=BLUE, label="megakernel solve"), Patch(color="#b7b7b2", label="CPU preprocessing, policy, host (by difference)")],
              frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.45, 1.45))
    save(fig, "fig_breakdown")


def fig_kernel_history():
    steps = [("first correct\nkernel", 76.9), ("vector DSMEM,\nno calls between\nWGMMAs", 55.3), ("relaxed cluster\narrives, fast erf", 43.8),
             ("no producer warp\n(255 registers)", 41.9), ("warpgroup split,\nparallel M-phases", 41.3)]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    x = np.arange(len(steps)); v = [s[1] for s in steps]
    ax.plot(x, v, color=BLUE, lw=1.6, marker="o", ms=6, mec="white", mew=0.8)
    for xi, vi in zip(x, v): ax.text(xi, vi + 2.5, f"{vi:.1f}", ha="center", fontsize=8)
    ax.axhline(82.3, color=OR_D, ls="--", lw=0.9); ax.text(4.35, 84.0, "LeWM-Flash solve, 82.3 ms", ha="right", fontsize=7.5, color=OR_D)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=7); ax.set_ylim(0, 92); ax.set_ylabel("single-env solve (ms)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    save(fig, "fig_kernel_history")


if __name__ == "__main__":
    for f in [fig_system, fig_icem_schedule, fig_megakernel, fig_ladder, fig_success, fig_paired, fig_breakdown, fig_kernel_history, fig_cost_scaling]:
        f()
