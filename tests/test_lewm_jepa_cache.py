from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm"))

    from jepa import JEPA  # noqa: E402
    from tools.benchmarks.lewm_plan_benchmark import configure_model_for_variant  # noqa: E402


    class TinyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device_anchor = nn.Parameter(torch.zeros(()))

        def forward(self, pixels: torch.Tensor, interpolate_pos_encoding: bool = True):
            del interpolate_pos_encoding
            flat = pixels.reshape(pixels.shape[0], -1)
            hidden = torch.stack(
                [flat.mean(dim=1), flat.sum(dim=1)],
                dim=1,
            ).unsqueeze(1)
            return SimpleNamespace(last_hidden_state=hidden)


    class TinyActionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, action: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return action[..., :1].expand(*action.shape[:-1], 2)


    class TinyPredictor(nn.Module):
        def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
            return emb + 0.01 * act_emb


    def make_model() -> JEPA:
        return JEPA(
            encoder=TinyEncoder(),
            predictor=TinyPredictor(),
            action_encoder=TinyActionEncoder(),
            projector=nn.Identity(),
            pred_proj=nn.Identity(),
        )


    def make_inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        info = {
            "pixels": torch.arange(4, dtype=torch.float32).reshape(1, 2, 2, 1, 1, 1),
            "goal": torch.arange(4, 8, dtype=torch.float32).reshape(1, 2, 2, 1, 1, 1),
            "action": torch.zeros(1, 2, 2, 1),
        }
        action_candidates = torch.linspace(0.0, 1.0, steps=6).reshape(1, 2, 3, 1)
        return info, action_candidates


    def count_encode_calls(model: JEPA) -> dict[str, int]:
        counts = {"goal": 0, "initial": 0}
        original_encode = model.encode

        def counted_encode(info: dict[str, torch.Tensor]):
            bucket = "initial" if "action" in info else "goal"
            counts[bucket] += 1
            return original_encode(info)

        model.encode = counted_encode  # type: ignore[method-assign]
        return counts


@unittest.skipIf(torch is None, "torch is required for LeWM JEPA cache tests")
class LeWMJEPACacheTests(unittest.TestCase):
    def test_cache_embeddings_variant_enables_model_cache(self) -> None:
        model = make_model()

        configure_model_for_variant(model, "cache_embeddings")

        self.assertTrue(model.embedding_cache_enabled())

    def test_baseline_variant_leaves_model_cache_disabled(self) -> None:
        model = make_model()
        model.set_embedding_cache(True)
        model.set_rollout_prealloc(True)
        model.set_rollout_action_cache(True)

        configure_model_for_variant(model, "baseline")

        self.assertFalse(model.embedding_cache_enabled())
        self.assertFalse(model.rollout_prealloc_enabled())
        self.assertFalse(model.rollout_action_cache_enabled())

    def test_rollout_prealloc_variant_enables_cache_and_prealloc(self) -> None:
        model = make_model()

        configure_model_for_variant(model, "rollout_prealloc")

        self.assertTrue(model.embedding_cache_enabled())
        self.assertTrue(model.rollout_prealloc_enabled())
        self.assertFalse(model.rollout_action_cache_enabled())

    def test_rollout_action_cache_variant_enables_cache_and_action_cache(self) -> None:
        model = make_model()

        configure_model_for_variant(model, "rollout_action_cache")

        self.assertTrue(model.embedding_cache_enabled())
        self.assertFalse(model.rollout_prealloc_enabled())
        self.assertTrue(model.rollout_action_cache_enabled())

    def test_compile_cost_variant_uses_l1_base_without_l3_rollout_flags(self) -> None:
        model = make_model()

        configure_model_for_variant(model, "compile_cost")

        self.assertTrue(model.embedding_cache_enabled())
        self.assertFalse(model.rollout_prealloc_enabled())
        self.assertFalse(model.rollout_action_cache_enabled())
        self.assertFalse(model.compile_cost_enabled())

    def test_rollout_modes_are_mutually_exclusive(self) -> None:
        model = make_model()

        model.set_rollout_prealloc(True)
        self.assertTrue(model.rollout_prealloc_enabled())
        self.assertFalse(model.rollout_action_cache_enabled())

        model.set_rollout_action_cache(True)
        self.assertFalse(model.rollout_prealloc_enabled())
        self.assertTrue(model.rollout_action_cache_enabled())

    def test_cache_embeddings_reuses_goal_and_initial_embeddings(self) -> None:
        model = make_model().set_embedding_cache(True)
        counts = count_encode_calls(model)
        info, action_candidates = make_inputs()

        first = model.get_cost(info, action_candidates)
        second = model.get_cost(info, action_candidates)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(counts, {"goal": 1, "initial": 1})

    def test_baseline_recomputes_goal_and_initial_embeddings(self) -> None:
        model = make_model()
        counts = count_encode_calls(model)
        info, action_candidates = make_inputs()

        model.get_cost(info, action_candidates)
        model.get_cost(info, action_candidates)

        self.assertEqual(counts, {"goal": 2, "initial": 2})

    def test_rollout_prealloc_matches_dynamic_rollout_and_cost(self) -> None:
        model = make_model().eval()

        dynamic_info, dynamic_actions = make_inputs()
        dynamic_cost = model.get_cost(dynamic_info, dynamic_actions)
        dynamic_rollout = dynamic_info["predicted_emb"].clone()

        model.set_rollout_prealloc(True)
        prealloc_info, prealloc_actions = make_inputs()
        prealloc_cost = model.get_cost(prealloc_info, prealloc_actions)
        prealloc_rollout = prealloc_info["predicted_emb"]

        torch.testing.assert_close(prealloc_rollout, dynamic_rollout, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(prealloc_cost, dynamic_cost, rtol=1e-6, atol=1e-6)

    def test_rollout_action_cache_matches_dynamic_rollout_and_cost(self) -> None:
        model = make_model().eval()

        dynamic_info, dynamic_actions = make_inputs()
        dynamic_cost = model.get_cost(dynamic_info, dynamic_actions)
        dynamic_rollout = dynamic_info["predicted_emb"].clone()

        model.set_rollout_action_cache(True)
        action_cache_info, action_cache_actions = make_inputs()
        action_cache_cost = model.get_cost(action_cache_info, action_cache_actions)
        action_cache_rollout = action_cache_info["predicted_emb"]

        torch.testing.assert_close(action_cache_rollout, dynamic_rollout, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(action_cache_cost, dynamic_cost, rtol=1e-6, atol=1e-6)

    def test_cost_from_embeddings_matches_get_cost(self) -> None:
        model = make_model().eval().set_embedding_cache(True)
        info, action_candidates = make_inputs()

        expected = model.get_cost(info, action_candidates)
        actual = model.get_cost_from_embeddings(
            info["emb"],
            info["goal_emb"],
            action_candidates,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_compile_cost_get_cost_matches_dynamic_cost(self) -> None:
        dynamic_model = make_model().eval().set_embedding_cache(True)
        dynamic_info, dynamic_actions = make_inputs()
        expected = dynamic_model.get_cost(dynamic_info, dynamic_actions)

        compile_model = make_model().eval().set_embedding_cache(True).set_compile_cost(True)
        compile_info, compile_actions = make_inputs()
        actual = compile_model.get_cost(compile_info, compile_actions)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertIn("emb", compile_info)
        self.assertIn("goal_emb", compile_info)

    def test_rollout_prealloc_reduces_repeated_action_encoding(self) -> None:
        dynamic_model = make_model().eval()
        dynamic_info, dynamic_actions = make_inputs()
        dynamic_model.get_cost(dynamic_info, dynamic_actions)

        prealloc_model = make_model().eval().set_rollout_prealloc(True)
        prealloc_info, prealloc_actions = make_inputs()
        prealloc_model.get_cost(prealloc_info, prealloc_actions)

        self.assertEqual(dynamic_model.action_encoder.calls, 3)
        self.assertEqual(prealloc_model.action_encoder.calls, 2)

    def test_rollout_action_cache_reduces_repeated_action_encoding(self) -> None:
        dynamic_model = make_model().eval()
        dynamic_info, dynamic_actions = make_inputs()
        dynamic_model.get_cost(dynamic_info, dynamic_actions)

        action_cache_model = make_model().eval().set_rollout_action_cache(True)
        action_cache_info, action_cache_actions = make_inputs()
        action_cache_model.get_cost(action_cache_info, action_cache_actions)

        self.assertEqual(dynamic_model.action_encoder.calls, 3)
        self.assertEqual(action_cache_model.action_encoder.calls, 2)


if __name__ == "__main__":
    unittest.main()
