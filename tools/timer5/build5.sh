#!/bin/bash
# build5.sh: builds tools/timer5/libcsta5.<so|dylib> from csta5_kern.cc with the flags of
# tools/best/build_i.sh for libi_kern (-O3 -ffp-contract=off -fno-fast-math; no -march on the hosts: E5-4620 has no AVX2).
# Optional: build5.sh <src.cc> <out> builds another variant (A/B tests of single ideas).
cd "$(dirname "$0")" || exit 1
if [ "$(uname)" = Darwin ]; then X=dylib; SH=-dynamiclib; AR=-mcpu=native; CXX=${CXX:-clang++}; else X=so; SH=-shared; AR=; CXX=${CXX:-g++}; fi
SRC=${1:-csta5_kern.cc}; OUT=${2:-libcsta5.$X}
$CXX -std=c++17 -O3 -ffp-contract=off -fno-fast-math $AR -fPIC -pthread $SH "$SRC" -o "$OUT.tmp.$$" && mv -f "$OUT.tmp.$$" "$OUT" && echo "built $(pwd)/$OUT"
