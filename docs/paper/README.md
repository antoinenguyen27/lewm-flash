# Paper and presentation

- `paper.pdf` / `paper.docx`: *Planning at the Speed of Control: Fusing a JEPA World-Model Planner into One GPU Kernel*.
- `presentation.pptx` / `presentation.pdf`: 17-slide deck with speaker notes.

Sources are in `src/`: chapter markdown in `sections/`, figures from `make_figs.py`, statistics from `compute_stats.py`
(reads the committed `results/` records), the annotated bibliography in `litreview.md`, and the deck in `deck/make_deck.js`.

```bash
python src/compute_stats.py && python src/make_figs.py      # needs numpy, scipy, matplotlib
NODE_PATH=<node_modules with docx@9> PY=<python with pypdf> src/build_all.sh   # needs LibreOffice
NODE_PATH=<node_modules with pptxgenjs> node src/deck/make_deck.js
```
