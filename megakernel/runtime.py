"""Python wrapper around the LeWM-MK persistent kernel."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
NE, SMAX = 50, 300


def source_hash() -> str:
    return hashlib.sha256((HERE / "lewm_mk.cu").read_bytes()).hexdigest()[:16]


def load_ext(build_dir: str, prof: bool = False):
    import torch  # noqa: F401  (the extension links against torch)

    tag = "_prof" if prof else ""
    stamp = Path(build_dir) / f"source_hash{tag}.txt"
    so_txt = Path(build_dir) / f"so_path{tag}.txt"
    so = Path(so_txt.read_text().strip()) if so_txt.exists() else Path(build_dir) / "lewm_mk.so"
    if not so.exists() or not stamp.exists() or stamp.read_text().strip() != source_hash():
        raise RuntimeError(f"prebuilt extension missing or stale in {build_dir}; run the build entrypoint")
    spec = importlib.util.spec_from_file_location(so.stem, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def icem_sample_counts(num_samples=300, n_steps=30, topk=30, sample_decay=1.25, bucket=60) -> list[int]:
    out = []
    for step in range(n_steps):
        if sample_decay == 1.0:
            out.append(num_samples)
            continue
        count = max(int(math.ceil(num_samples * (sample_decay ** (-step)))), min(num_samples, 2 * topk))
        if bucket > 0:
            count = min(num_samples, int(math.ceil(count / bucket)) * bucket)
        out.append(count)
    return out


class Megakernel:
    def __init__(self, model, ext, cl: int | None = None, clusters: int | None = None, max_envs: int = 1,
                 encoder: bool = False):
        import torch

        from megakernel import pack

        self.ext = ext
        fw = pack.fold(model)
        fe = pack.fold_encoder(model) if encoder else None
        raw, kinds = pack.panels(fw, fe)
        dev = "cuda"
        self.encoder = encoder
        self.panels = ext.swizzle_panels(raw.to(dev).contiguous(), kinds.to(dev).contiguous())
        t = lambda k: fw[k].to(dev).float().contiguous()
        self.w = [self.panels, t("b_qkv"), t("b_o"), t("b_1"), t("b_2"), t("b_p1"), t("b_p2"), t("b_e1"), t("b_e2"),
                  t("b_ada"), t("pos"), t("lnf_g"), t("lnf_b")]
        if encoder:
            e = lambda k: fe[k].to(dev).float().contiguous()
            self.w += [e("patch_b"), e("pos"), e("e_bqkv"), e("e_bo"), e("e_b1"), e("e_b2"), e("proj_b1"),
                       e("proj_b2")]
        f32 = dict(device=dev, dtype=torch.float32)
        bf = dict(device=dev, dtype=torch.bfloat16)
        E = self.max_envs = int(max_envs)
        self.emb0 = torch.zeros(E, 192, **f32)
        self.goal = torch.zeros(E, 192, **f32)
        self.init_mean = torch.zeros(E, NE, **f32)
        self.cand_in = torch.zeros(E * SMAX, NE, **f32)
        self.cand = torch.zeros(2 * E * SMAX * NE, **f32)
        self.emb_hist = torch.zeros(E * SMAX * 5 * 192, **f32)
        self.cost = torch.zeros(2 * E * SMAX, **f32)
        self.out = torch.zeros(2 * NE + 1, **f32)
        self.cost_out = torch.zeros(E * SMAX, **f32)
        self.trace = torch.zeros(32, **f32)
        self.actemb = torch.zeros(E * SMAX * 5 * 192, **bf)
        self.mod = torch.zeros(E * SMAX * 5 * 6 * 1152, **bf)
        self.kv = torch.zeros(E * SMAX * 2 * 6 * 16 * 128, **bf)
        self.counter = torch.zeros(1, device=dev, dtype=torch.int32)
        self.err = torch.zeros(1, device=dev, dtype=torch.int32)
        self.prof = torch.zeros(0, device=dev, dtype=torch.int64)
        self.st_mean = torch.zeros(E, NE, **f32)
        self.st_std = torch.zeros(E, NE, **f32)
        self.st_cost = torch.zeros(E, **f32)
        self.st_eidx = torch.zeros(E, 8, device=dev, dtype=torch.int32)
        self.buf = [self.emb0, self.goal, self.init_mean, self.cand_in, self.cand, self.emb_hist, self.cost, self.out,
                    self.cost_out, self.trace, self.actemb, self.mod, self.kv, self.counter, self.err, self.prof,
                    self.st_mean, self.st_std, self.st_cost, self.st_eidx]
        self.actacc = torch.zeros(2, E * SMAX * 5, 192, **f32)  # action-encoder fp32 partial sums (step parity)
        self.buf.append(self.actacc)
        if encoder:  # fused-encoder scratch for up to 2 images (observation + goal) per env
            rows = 2 * E * 257
            self.img = torch.zeros(2 * E, 224, 224, 3, device=dev, dtype=torch.uint8)
            self.img_dst = torch.zeros(2 * E, device=dev, dtype=torch.int32)
            self.ex = torch.zeros(rows, 192, **f32)
            self.eqkv = torch.zeros(rows, 576, **bf)
            self.eattn = torch.zeros(rows, 192, **bf)
            self.eh = torch.zeros(rows, 768, **bf)
        if cl is None:
            cl = 16 if ext.query_max_clusters(16) >= 8 else 8
        self.cl = cl
        self.max_clusters = ext.query_max_clusters(cl)
        self.clusters = min(clusters or self.max_clusters, self.max_clusters)
        if self.clusters < 1:
            raise RuntimeError(f"no co-resident {cl}-CTA clusters available")
        self.seed = 0

    def enable_profiling(self, on: bool = True):
        import torch

        self.prof = torch.zeros(self.cl * self.clusters * 16 if on else 0, device="cuda", dtype=torch.int64)
        self.buf[15] = self.prof

    PROF_NAMES = ["ring_wait", "cluster_sync", "grid_sync", "attention", "wgmma", "epilogue", "rollout_total",
                  "phase_act", "phase_mod", "chunks", "topk", "producer_empty_wait", "owner_attn_half", "-", "-",
                  "kernel_total"]

    def profile_summary(self):
        p = self.prof.view(-1, 16).double().cpu()
        names = self.PROF_NAMES
        cta0 = {n: p[0, i].item() / 1e3 for i, n in enumerate(names) if n != "-"}
        avg = {n: p[:, i].mean().item() / 1e3 for i, n in enumerate(names) if n != "-"}
        return {"cta0_us": cta0, "mean_us": avg}

    def _launch(self, counts, mode, topk=30, elite_ret=5, init_std=1.0, seed=None, envs=1, n_img=0):
        if seed is None:
            self.seed += 1
            seed = self.seed
        buf = list(self.buf)
        buf[0], buf[1] = self.emb0[:envs], self.goal[:envs]  # the kernel takes E from emb0's leading dim
        if counts:
            self.actacc[0, : envs * counts[0] * 5].zero_()  # step 0's partial sums; the kernel zeroes later steps
        if n_img:
            buf += [self.img[:n_img], self.img_dst[:n_img], self.ex, self.eqkv, self.eattn, self.eh]
        self.ext.solve(self.cl, self.clusters, self.w, buf, list(counts), mode, topk, elite_ret, float(init_std),
                       int(seed))

    def micro(self, mode: int, iters: int):
        self.ext.solve(self.cl, self.clusters, self.w, self.buf, [], mode, iters, 0, 1.0, 0)

    def set_problem(self, emb, goal):
        E = emb.reshape(-1, 192).shape[0]
        if E > self.max_envs:
            raise ValueError(f"{E} envs > max_envs={self.max_envs}")
        self.emb0[:E].copy_(emb.reshape(E, 192))
        self.goal[:E].copy_(goal.reshape(E, 192))
        return E

    def costs(self, emb, goal, candidates):
        """emb/goal [E, 192]; candidates [E, S, 5, 10] (or [S, 5, 10] for one env) -> costs [E, S] / [S]."""
        E = self.set_problem(emb, goal)
        cand = candidates.reshape(E, -1, NE)
        S = cand.shape[1]
        self.cand_in[:E * S].copy_(cand.reshape(E * S, NE))
        self._launch([S], 1, envs=E)
        out = self.cost_out[:E * S].view(E, S).clone()
        return out[0] if candidates.dim() == 3 else out

    def solve_batch(self, emb, goal, init_mean=None, counts=None, init_std=1.0, seed=None, topk=30, elite_ret=5):
        """All envs' full iCEM solves in ONE launch. emb/goal [E, 192] -> (mean [E,5,10], std [E,5,10], cost [E])."""
        E = self.set_problem(emb, goal)
        if init_mean is None:
            self.st_mean[:E].zero_()
        else:
            self.st_mean[:E].copy_(init_mean.reshape(E, NE))
        self.st_std[:E].fill_(float(init_std))
        self._launch(counts or icem_sample_counts(), 0, topk=topk, elite_ret=elite_ret, init_std=init_std, seed=seed,
                     envs=E)
        return self.st_mean[:E].view(E, 5, 10), self.st_std[:E].view(E, 5, 10), self.st_cost[:E]

    def _stage_images(self, obs_u8, goal_u8=None, goal_envs=None):
        """obs_u8 [E,224,224,3] uint8 (any device); goal_u8 [G,224,224,3] for the envs listed in goal_envs (new
        goals; the other envs keep their cached goal embedding in self.goal). Zeroes the embeddings that the
        encoder's projector accumulates into."""
        import torch

        E = obs_u8.shape[0]
        if E > self.max_envs:
            raise ValueError(f"{E} envs > max_envs={self.max_envs}")
        n = E
        self.img[:E].copy_(obs_u8, non_blocking=True)
        dst = list(range(0, 2 * E, 2))
        self.emb0[:E].zero_()
        if goal_u8 is not None and len(goal_envs):
            G = len(goal_envs)
            self.img[E:E + G].copy_(goal_u8, non_blocking=True)
            dst += [2 * e + 1 for e in goal_envs]
            self.goal[torch.as_tensor(goal_envs, device=self.goal.device)] = 0.0
            n += G
        self.img_dst[:n].copy_(torch.as_tensor(dst, dtype=torch.int32), non_blocking=True)
        return E, n

    def encode(self, obs_u8, goal_u8=None, goal_envs=None):
        """Fused encoder only: embeddings of observation (and listed goal) images -> (emb0 [E,192], goal [E,192])."""
        E, n = self._stage_images(obs_u8, goal_u8, goal_envs)
        self._launch([], 0, envs=E, n_img=n)
        return self.emb0[:E], self.goal[:E]

    def solve_images(self, obs_u8, goal_u8=None, goal_envs=None, init_mean=None, counts=None, init_std=1.0,
                     seed=None, topk=30, elite_ret=5):
        """One launch per plan: encode raw frames + run every env's full iCEM solve."""
        E, n = self._stage_images(obs_u8, goal_u8, goal_envs)
        if init_mean is None:
            self.st_mean[:E].zero_()
        else:
            self.st_mean[:E].copy_(init_mean.reshape(E, NE))
        self.st_std[:E].fill_(float(init_std))
        self._launch(counts or icem_sample_counts(), 0, topk=topk, elite_ret=elite_ret, init_std=init_std, seed=seed,
                     envs=E, n_img=n)
        return self.st_mean[:E].view(E, 5, 10), self.st_std[:E].view(E, 5, 10), self.st_cost[:E]

    def solve(self, emb, goal, init_mean=None, **kw):
        """Single-env iCEM solve in one launch. Returns (mean [5,10], std [5,10], final top-k mean cost)."""
        m, s, c = self.solve_batch(emb.reshape(1, 192), goal.reshape(1, 192),
                                   None if init_mean is None else init_mean.reshape(1, NE), **kw)
        return m[0], s[0], c[0]
