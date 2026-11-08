#!/usr/bin/env bash
# Build the submission archive: the solution pack plus a copy of the GPU kernel sources.
set -e; cd "$(dirname "$0")"; ROOT=..; OUT=${1:-../docs/paper/solution_pack.zip}
TMP=$(mktemp -d); mkdir -p "$TMP/solution_pack/kernel"
cp -R lewm_flash.py README.md data outputs "$TMP/solution_pack/"
cp $ROOT/megakernel/{lewm_mk.cu,pack.py,runtime.py,solver.py,build.py,model_io.py} "$TMP/solution_pack/kernel/"
rm -f "$OUT"; (cd "$TMP" && zip -qr - solution_pack) > "$OUT"; rm -rf "$TMP"; echo "wrote $OUT"
