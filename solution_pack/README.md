# LeWM-Flash solution pack

Datasets and the solution program for the report *Planning at the Speed of Control: Profiling and fusing world-model
robot planning into a single GPU kernel*.

## Run

Python 3.10+ with numpy and matplotlib. Runs on any computer (no GPU needed), in about five seconds.

```bash
pip install numpy matplotlib
python lewm_flash.py
```

This regenerates `outputs/results.json`, `outputs/tables.md` and `outputs/figs/*.png`, which hold every statistic,
table and data figure in the report.

| command | step of the analytics pipeline |
|---|---|
| `python lewm_flash.py prepare` | data preparation: rebuild `data/*.csv` from the raw benchmark records (run inside the project repository) |
| `python lewm_flash.py` (`analyse`) | preprocessing, compute-profile analysis, scaling-model fits, statistical evaluation, tables and figures |
| `python lewm_flash.py reproduce [--seeds N] [--yes]` | re-collect the raw latency data on a cloud H100 through Modal (account, LeWM checkpoint and kernel build required; costs money; dry run without `--yes`) |

## Contents

| path | content |
|---|---|
| `lewm_flash.py` | the solution program |
| `data/calls.csv` | 11,942 timed policy calls: 61 one-robot episodes for 3 systems and 4 probe configurations |
| `data/episodes.csv` | 900 episode outcomes: 300 paired Push-T episodes for each of the 3 systems |
| `data/cost_scaling.csv` | latency of one scoring call at 60 to 300 candidates, per execution method |
| `data/components.csv` | component timings, synchronisation primitives and planner telemetry |
| `data/parity.csv` | numerical agreement with FP32 on identical inputs |
| `outputs/` | generated results |
| `kernel/` (submission archive only) | the GPU kernel (`lewm_mk.cu`) and its Python runtime, packer and planner class |

The report's Appendix A documents every column. In the project repository the kernel lives in `megakernel/`, and the
cloud entry points are in `modal/megakernel_app.py`.
