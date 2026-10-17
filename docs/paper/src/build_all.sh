#!/usr/bin/env bash
# Build the paper: paper.docx (live TOC field) and paper.pdf (static TOC with page numbers from the rendered layout).
# Needs node with docx@9 (NODE_PATH), LibreOffice, and a python with pypdf. Usage: NODE_PATH=... PY=... ./build_all.sh
set -e; cd "$(dirname "$0")"
SOFF=${SOFF:-/Applications/LibreOffice.app/Contents/MacOS/soffice}; PY=${PY:-python3}; OUT=../
cat sections/00_front.md sections/10_intro.md sections/20_method.md sections/30_results.md sections/40_discussion.md \
    sections/50_future_conclusion.md sections/80_references.md sections/95_appendices.md > paper.md
node build/build_docx.js paper.md config.json paper.docx | tail -1
sed 's/^%TOC$/%STATICTOC page_map.json/' paper.md > paper_static.md
echo '[]' > page_map.json
$SOFF --headless --convert-to pdf --outdir . paper.docx >/dev/null 2>&1
$PY page_map.py paper.md paper.pdf page_map.json
for pass in 1 2 3; do
  node build/build_docx.js paper_static.md config.json paper_static.docx >/dev/null
  $SOFF --headless --convert-to pdf --outdir . paper_static.docx >/dev/null 2>&1
  cp page_map.json page_map_prev.json; $PY page_map.py paper.md paper_static.pdf page_map.json
  cmp -s page_map.json page_map_prev.json && { echo "page map stable after pass $pass"; break; }
done
cp paper_static.pdf $OUT/paper.pdf; cp paper.docx $OUT/paper.docx
rm -f paper_static.md paper_static.docx paper_static.pdf page_map_prev.json paper.pdf paper.docx paper.md
