"""Fold LeWM cost-path weights (LayerNorm affines, BatchNorm, the k=1 conv) and pack them into 24 KB bf16 panels.

Panel kinds: P = [K=192, N=64] and Q = [K=64, N=192], row-major, then pre-swizzled on device into the ThunderKittens
shared-tile layout so the kernel can bulk-copy them straight into the GEMM operand slot. Panel ids must match the
OFF_* constants in lewm_mk.cu.
"""

from __future__ import annotations

D, NH, DH, FF, NL, HID_E, AD = 192, 16, 64, 2048, 6, 768, 10
OFF = dict(QKV=0, WO=288, W1=384, W2=576, P1=768, P2=800, E1=832, E2=844, ADA=856)
NPANEL = 964


def fold(model):
    """Return fp32 folded matrices in GEMM orientation y = x @ W + b (W is [in, out])."""
    import torch

    f = lambda t: t.detach().float().cpu()
    pred = model.predictor
    tr = pred.transformer
    out = {"pos": f(pred.pos_embedding[0, :3]), "lnf_g": f(tr.norm.weight), "lnf_b": f(tr.norm.bias)}
    qkv, bqkv, wo, bo, w1, b1, w2, b2, ada, bada = ([] for _ in range(10))
    for blk in tr.layers:
        g, b = f(blk.attn.norm.weight), f(blk.attn.norm.bias)
        wt = f(blk.attn.to_qkv.weight).t()  # [192, 3072]
        qkv.append(g[:, None] * wt)
        bqkv.append(b @ wt)
        wo.append(f(blk.attn.to_out[0].weight).t())  # [1024, 192]
        bo.append(f(blk.attn.to_out[0].bias))
        ln, l1, l2 = blk.mlp.net[0], blk.mlp.net[1], blk.mlp.net[4]
        g, b = f(ln.weight), f(ln.bias)
        w1t = f(l1.weight).t()  # [192, 2048]
        w1.append(g[:, None] * w1t)
        b1.append(b @ w1t + f(l1.bias))
        w2.append(f(l2.weight).t())  # [2048, 192]
        b2.append(f(l2.bias))
        ada.append(f(blk.adaLN_modulation[1].weight).t())  # [192, 1152]
        bada.append(f(blk.adaLN_modulation[1].bias))
    out.update(qkv=torch.stack(qkv), b_qkv=torch.stack(bqkv), wo=torch.stack(wo), b_o=torch.stack(bo),
               w1=torch.stack(w1), b_1=torch.stack(b1), w2=torch.stack(w2), b_2=torch.stack(b2),
               ada=torch.stack(ada), b_ada=torch.stack(bada))

    lin1, bn, lin2 = model.pred_proj.net[0], model.pred_proj.net[1], model.pred_proj.net[3]
    scale = f(bn.weight) / torch.sqrt(f(bn.running_var) + bn.eps)
    out["p1"] = f(lin1.weight).t() * scale[None, :]
    out["b_p1"] = (f(lin1.bias) - f(bn.running_mean)) * scale + f(bn.bias)
    out["p2"] = f(lin2.weight).t()
    out["b_p2"] = f(lin2.bias)

    emb = model.action_encoder
    wc, bc = f(emb.patch_embed.weight)[:, :, 0], f(emb.patch_embed.bias)  # [10, 10], [10]
    la, lb = emb.embed[0], emb.embed[2]
    wa = f(la.weight)  # [768, 10]
    out["e1"] = (wa @ wc).t()  # [10, 768]
    out["b_e1"] = wa @ bc + f(la.bias)
    out["e2"] = f(lb.weight).t()  # [768, 192]
    out["b_e2"] = f(lb.bias)
    return out


def panels(fw):
    """Return (raw [NPANEL, 192*64] bf16 row-major panels, kinds [NPANEL] int32: 0 = P, 1 = Q)."""
    import torch

    raw = torch.zeros(NPANEL, 192 * 64, dtype=torch.float32)
    kinds = torch.zeros(NPANEL, dtype=torch.int32)

    def put(i, mat, kind):
        assert mat.shape == ((192, 64) if kind == 0 else (64, 192)), (i, mat.shape)
        raw[i] = mat.reshape(-1)
        kinds[i] = kind

    for l in range(NL):
        for h in range(NH):
            for part in range(3):
                put(OFF["QKV"] + (l * NH + h) * 3 + part, fw["qkv"][l][:, part * 1024 + h * DH: part * 1024 + (h + 1) * DH], 0)
            put(OFF["WO"] + l * NH + h, fw["wo"][l][h * DH:(h + 1) * DH, :], 1)
        for s in range(FF // 64):
            put(OFF["W1"] + l * 32 + s, fw["w1"][l][:, s * 64:(s + 1) * 64], 0)
            put(OFF["W2"] + l * 32 + s, fw["w2"][l][s * 64:(s + 1) * 64, :], 1)
        for n in range(18):
            put(OFF["ADA"] + l * 18 + n, fw["ada"][l][:, n * 64:(n + 1) * 64], 0)
    for s in range(FF // 64):
        put(OFF["P1"] + s, fw["p1"][:, s * 64:(s + 1) * 64], 0)
        put(OFF["P2"] + s, fw["p2"][s * 64:(s + 1) * 64, :], 1)
    e1 = torch.zeros(192, HID_E)
    e1[:AD] = fw["e1"]
    for j in range(HID_E // 64):
        put(OFF["E1"] + j, e1[:, j * 64:(j + 1) * 64], 0)
        put(OFF["E2"] + j, fw["e2"][j * 64:(j + 1) * 64, :], 1)
    return raw.to(torch.bfloat16), kinds
