#!/bin/bash
# build_n.sh: the kernels of tools/best5 (Linux .so, no AVX2 flags; macOS .dylib). Flags of tools/mid5/build_mid5.sh.
cd "$(dirname "$0")" || exit 1
if [ "$(uname)" = Darwin ]; then X=dylib; SH=-dynamiclib; CXX=${CXX:-clang++}; else X=so; SH=-shared; CXX=${CXX:-g++}; fi
python3 make_n_rr.py && python3 make_n_rs.py || exit 1
F="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread $SH"
$CXX $F n_rr.cc -o libn_rr.$X.tmp.$$ && mv -f libn_rr.$X.tmp.$$ libn_rr.$X && echo built libn_rr.$X &
$CXX $F n_rs.cc -o libn_rs.$X.tmp.$$ && mv -f libn_rs.$X.tmp.$$ libn_rs.$X && echo built libn_rs.$X &
$CXX $F n_sel.cc -o libn_sel.$X.tmp.$$ && mv -f libn_sel.$X.tmp.$$ libn_sel.$X && echo built libn_sel.$X &
$CXX $F n_norm.cc -o libn_norm.$X.tmp.$$ && mv -f libn_norm.$X.tmp.$$ libn_norm.$X && echo built libn_norm.$X &
$CXX $F n_b2.cc -o libn_b2.$X.tmp.$$ && mv -f libn_b2.$X.tmp.$$ libn_b2.$X && echo built libn_b2.$X &
$CXX $F n_seg.cc -o libn_seg.$X.tmp.$$ && mv -f libn_seg.$X.tmp.$$ libn_seg.$X && echo built libn_seg.$X &
$CXX $F n_ofc.cc -o libn_ofc.$X.tmp.$$ && mv -f libn_ofc.$X.tmp.$$ libn_ofc.$X && echo built libn_ofc.$X &
wait
