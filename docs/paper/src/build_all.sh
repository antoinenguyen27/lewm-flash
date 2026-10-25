#!/usr/bin/env bash
# Build the report: report.docx (live TOC field) and report.pdf (static TOC with page numbers from the rendered layout).
# Needs node with docx@9 (NODE_PATH), LibreOffice, and a python with pypdf. Usage: NODE_PATH=... PY=... ./build_all.sh
set -e; cd "$(dirname "$0")"
SOFF=${SOFF:-/Applications/LibreOffice.app/Contents/MacOS/soffice}; PY=${PY:-python3}; OUT=..
cat sections/00_front.md sections/10_intro.md sections/20_background.md sections/30_method.md sections/40_profile.md \
    sections/50_solution.md sections/60_evaluation.md sections/70_discussion.md sections/80_conclusion.md \
    sections/90_references.md sections/95_appendices.md > report.md
node build/build_docx.js report.md config.json report.docx | tail -1
sed 's/^%TOC$/%STATICTOC page_map.json/' report.md > report_static.md
echo '[]' > page_map.json
$SOFF --headless --convert-to pdf --outdir . report.docx >/dev/null 2>&1
$PY page_map.py report.md report.pdf page_map.json
for pass in 1 2 3; do
  node build/build_docx.js report_static.md config.json report_static.docx >/dev/null
  $SOFF --headless --convert-to pdf --outdir . report_static.docx >/dev/null 2>&1
  cp page_map.json page_map_prev.json; $PY page_map.py report.md report_static.pdf page_map.json
  cmp -s page_map.json page_map_prev.json && { echo "page map stable after pass $pass"; break; }
done
cp report_static.pdf $OUT/report.pdf; cp report.docx $OUT/report.docx
rm -f report_static.md report_static.docx report_static.pdf page_map_prev.json report.pdf report.docx report.md
