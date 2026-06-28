# Results Artifacts

This directory contains selected metrics and manifests used by the README
headline table.

Included runs:

- `lewm_baseline_telem100_seed0_l40s`
- `lewm_baseline_telem100_seed1_l40s`
- `lewm_baseline_telem100_seed2_l40s`
- `lewm_icem_bucket60_compile_prewarm_seed0-2_l40s`
- `lewm_icem_bucket60_bf16_seed0-2_l40s`
- `lewm_icem_static_chain_eager_seed0-2_l40s`
- `lewm_icem_static_chain_eager_bf16_seed0-2_l40s`

Recompute the headline summary:

```bash
uv run python scripts/summarize_headline.py
```
