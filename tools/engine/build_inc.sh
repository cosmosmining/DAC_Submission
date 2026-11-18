#!/bin/bash
# build_inc.sh: builds the incremental timer library next to this script.
#   Linux: libcsta_inc.so (+ libcsta_inc_ref.so, a byte copy loaded as a SEPARATE library by the tests, so that
#   the reference full analysis has its own static state).  macOS: libcsta_inc.dylib (+ _ref).
# Flags as tools/gapla/csta_build.sh: -O3, no -ffast-math, no FMA contraction.
cd "$(dirname "$0")" || exit 1
if [ "$(uname)" = Darwin ]; then CXX=${CXX:-clang++}; EXT=dylib; ARCH=-mcpu=native; else CXX=${CXX:-g++}; EXT=so; ARCH=; fi
$CXX -std=c++17 -O3 -ffp-contract=off -fno-fast-math $ARCH -shared -fPIC -pthread csta_inc_kern.cc -o libcsta_inc.$EXT.tmp.$$ \
  && mv -f libcsta_inc.$EXT.tmp.$$ libcsta_inc.$EXT && cp -f libcsta_inc.$EXT libcsta_inc_ref.$EXT.tmp.$$ \
  && mv -f libcsta_inc_ref.$EXT.tmp.$$ libcsta_inc_ref.$EXT && echo "built $(pwd)/libcsta_inc.$EXT ($($CXX --version | head -1))"
