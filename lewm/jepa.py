"""JEPA Implementation"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.cache_embeddings = False
        self.rollout_prealloc = False
        self.rollout_action_cache = False
        self.compile_cost = False
        self.model_cost_autocast = "none"

    def set_embedding_cache(self, enabled: bool = True):
        self.cache_embeddings = enabled
        return self

    def embedding_cache_enabled(self) -> bool:
        return bool(getattr(self, "cache_embeddings", False))

    def set_rollout_prealloc(self, enabled: bool = True):
        self.rollout_prealloc = enabled
        if enabled:
            self.rollout_action_cache = False
        return self

    def rollout_prealloc_enabled(self) -> bool:
        return bool(getattr(self, "rollout_prealloc", False))

    def set_rollout_action_cache(self, enabled: bool = True):
        self.rollout_action_cache = enabled
        if enabled:
            self.rollout_prealloc = False
        return self

    def rollout_action_cache_enabled(self) -> bool:
        return bool(getattr(self, "rollout_action_cache", False))

    def set_compile_cost(self, enabled: bool = True):
        self.compile_cost = enabled
        return self

    def compile_cost_enabled(self) -> bool:
        return bool(getattr(self, "compile_cost", False))

    def set_model_cost_autocast(self, mode: str = "none"):
        if mode not in {"none", "bf16"}:
            raise ValueError(f"Unsupported model_cost_autocast mode: {mode!r}")
        self.model_cost_autocast = mode
        return self

    def model_cost_autocast_mode(self) -> str:
        return str(getattr(self, "model_cost_autocast", "none"))

    def model_cost_autocast_enabled(self) -> bool:
        return self.model_cost_autocast_mode() != "none"

    def model_cost_autocast_dtype(self):
        if self.model_cost_autocast_mode() == "bf16":
            return torch.bfloat16
        return None

    def _model_cost_autocast_context(self, reference: torch.Tensor):
        dtype = self.model_cost_autocast_dtype()
        if dtype is None or reference.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type=reference.device.type, dtype=dtype)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def _prepare_rollout_inputs(self, info, action_sequence):
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        if self.embedding_cache_enabled() and "emb" in info:
            emb = info["emb"]
        else:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            emb = _init["emb"]
            if self.embedding_cache_enabled():
                info["emb"] = emb
            else:
                info["emb"] = emb.unsqueeze(1).expand(B, S, -1, -1)
        emb = emb.unsqueeze(1).expand(B, S, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")
        return H, B, S, T, n_steps, emb, act, act_future

    def _rollout_dynamic(self, info, action_sequence, history_size: int = 3):
        H, B, S, T, n_steps, emb, act, act_future = self._prepare_rollout_inputs(
            info, action_sequence
        )

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def _rollout_prealloc(self, info, action_sequence, history_size: int = 3):
        H, B, S, T, n_steps, emb, act_0, act_future = self._prepare_rollout_inputs(
            info, action_sequence
        )
        batch_samples = B * S
        HS = history_size

        emb_buf = emb.new_empty((batch_samples, T + 1, emb.size(-1)))
        emb_buf[:, :H] = emb

        act_buf = action_sequence.new_empty((batch_samples, T, *action_sequence.shape[3:]))
        act_buf[:, :H] = act_0
        if n_steps > 0:
            act_buf[:, H:] = act_future
        act_emb_buf = self.action_encoder(act_buf)

        for t in range(n_steps + 1):
            end = H + t
            start = max(0, end - HS)
            emb_trunc = emb_buf[:, start:end]
            act_trunc = act_emb_buf[:, start:end]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb_buf[:, end : end + 1] = pred_emb

        pred_rollout = rearrange(emb_buf, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def _rollout_action_cache(self, info, action_sequence, history_size: int = 3):
        H, B, S, T, n_steps, emb, act_0, act_future = self._prepare_rollout_inputs(
            info, action_sequence
        )
        HS = history_size

        act = torch.cat([act_0, act_future], dim=1)
        act_emb = self.action_encoder(act)

        for t in range(n_steps + 1):
            end = H + t
            start = max(0, end - HS)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, start:end].contiguous()
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        if self.rollout_prealloc_enabled():
            return self._rollout_prealloc(info, action_sequence, history_size)
        if self.rollout_action_cache_enabled():
            return self._rollout_action_cache(info, action_sequence, history_size)
        return self._rollout_dynamic(info, action_sequence, history_size)

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)
        return self.cost_from_rollout(pred_emb, goal_emb)

    def cost_from_rollout(self, pred_emb: torch.Tensor, goal_emb: torch.Tensor):
        """Compute final-step cost from rollout and goal embeddings."""
        goal_emb = goal_emb[..., -1:, :]
        while goal_emb.ndim < pred_emb.ndim:
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb.expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def rollout_from_embeddings(
        self,
        emb: torch.Tensor,
        action_sequence: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        """Roll out from cached observation embeddings using tensor-only inputs."""
        H = emb.size(-2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        n_steps = T - H

        if emb.ndim == 3:
            emb = emb.unsqueeze(1).expand(B, S, -1, -1)
        elif emb.ndim != 4:
            raise ValueError(f"Expected emb rank 3 or 4, got shape {tuple(emb.shape)}")

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        return rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)

    def get_cost_from_embeddings(
        self,
        emb: torch.Tensor,
        goal_emb: torch.Tensor,
        action_candidates: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        """Tensor-only model-cost path for compile/graph experiments."""
        with self._model_cost_autocast_context(action_candidates):
            pred_rollout = self.rollout_from_embeddings(
                emb,
                action_candidates,
                history_size=history_size,
            )
            cost = self.cost_from_rollout(pred_rollout, goal_emb)
        return cost.float() if self.model_cost_autocast_enabled() else cost

    def _ensure_cost_embeddings(self, info_dict: dict, action_candidates: torch.Tensor):
        if not (self.embedding_cache_enabled() and "goal_emb" in info_dict):
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]

            for k in list(info_dict.keys()):
                if k.startswith("goal_") and k in goal:
                    goal[k[len("goal_") :]] = goal.pop(k)

            goal.pop("action", None)
            goal = self.encode(goal)
            info_dict["goal_emb"] = goal["emb"]

        H = info_dict["pixels"].size(2)
        act_0, _ = torch.split(action_candidates, [H, action_candidates.size(2) - H], dim=2)
        info_dict["action"] = act_0

        if self.embedding_cache_enabled() and "emb" in info_dict:
            emb = info_dict["emb"]
        else:
            _init = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            emb = _init["emb"]
            if self.embedding_cache_enabled():
                info_dict["emb"] = emb

        return emb, info_dict["goal_emb"]

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if self.compile_cost_enabled():
            emb, goal_emb = self._ensure_cost_embeddings(info_dict, action_candidates)
            return self.get_cost_from_embeddings(emb, goal_emb, action_candidates)

        if not (self.embedding_cache_enabled() and "goal_emb" in info_dict):
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]

            for k in info_dict:
                if k.startswith("goal_"):
                    goal[k[len("goal_") :]] = goal.pop(k)

            goal.pop("action")
            goal = self.encode(goal)

            info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
