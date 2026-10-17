"""Torch-free checks that the megakernel's host-side schedule and weight layout match the kernel and solver."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from megakernel import pack
from megakernel.runtime import icem_sample_counts

REPO_ROOT = Path(__file__).resolve().parents[1]


class MegakernelScheduleTest(unittest.TestCase):
    def test_bucketed_icem_schedule(self) -> None:
        counts = icem_sample_counts()
        self.assertEqual(counts[:9], [300, 240, 240, 180, 180, 120, 120, 120, 60])
        self.assertEqual(len(counts), 30)
        self.assertEqual(sum(counts), 2820)  # model_cost_evals_per_cem_call of the accepted LeWM-Flash run

    def test_panel_offsets_match_kernel(self) -> None:
        src = (REPO_ROOT / "megakernel" / "lewm_mk.cu").read_text()
        kernel = {m.group(1): int(m.group(2)) for m in re.finditer(r"OFF_([A-Z0-9]+) = (\d+)", src)}
        self.assertEqual(kernel, pack.OFF)
        self.assertEqual(int(re.search(r"NPANEL = (\d+)", src).group(1)), pack.NPANEL)

    def test_panel_count_covers_model(self) -> None:
        per_layer = 16 * 3 + 16 + 32 + 32 + 18  # qkv per head, out-proj per head, W1/W2 slices, AdaLN columns
        predictor = 6 * per_layer + 32 + 32 + 12 + 12
        enc_layer = 9 + 3 + 12 + 12  # qkv / out / fc1 N-panels, fc2 (4 K-chunks x 3 N-panels)
        encoder = 12 + 12 * enc_layer + 32 + 32  # patch embed (4 K-chunks x 3), layers, projector
        self.assertEqual(pack.OFF["EPATCH"], predictor)
        self.assertEqual(pack.NPANEL, predictor + encoder)


if __name__ == "__main__":
    unittest.main()
