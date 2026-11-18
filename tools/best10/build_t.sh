#!/bin/bash
# build_t.sh: the kernels of tools/best10 on the host (flags of tools/best5/build_n.sh; no -ffast-math).
cd "$(dirname "$0")" || exit 1
if [ "$(uname)" = Darwin ]; then X=dylib; SH=-dynamiclib; CXX=${CXX:-clang++}; else X=so; SH=-shared; CXX=${CXX:-g++}; fi
F="-O3 -std=c++17 -ffp-contract=off -fPIC $SH -pthread"
python3 mk_t_rr.py || exit 1
$CXX $F t_rr.cc -o libt_rr.$X.tmp.$$ && mv -f libt_rr.$X.tmp.$$ libt_rr.$X && echo built libt_rr.$X
$CXX -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread $SH t_csta.cc -o libt_csta.$X.tmp.$$ && mv -f libt_csta.$X.tmp.$$ libt_csta.$X && echo built libt_csta.$X
$CXX -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread $SH t_fpk.cc -o libt_fpk.$X.tmp.$$ && mv -f libt_fpk.$X.tmp.$$ libt_fpk.$X && echo built libt_fpk.$X
