#!/bin/bash
# build_all.sh [repo root]: every native library chain_n / cold_n load: tools/best4/build_all.sh + tools/timer5 + tools/par7 kernels + tools/best5 kernels.
set -u
ROOT=$(cd "${1:-$(dirname "$0")/../..}" && pwd); cd "$ROOT" || exit 1
bash tools/best4/build_all.sh "$ROOT"; fail=$?
bash tools/timer5/build5.sh || fail=1
CXX=${CXX:-g++}
( cd tools/par7 && $CXX -O2 -std=c++17 -fPIC -shared -ffp-contract=off -fno-fast-math -pthread -o libp7_pf.so p7_pf.cc && $CXX -O3 -std=c++17 -ffp-contract=off -fPIC -shared -pthread -o libp7_dp.so p7_dp_kern.cc && echo built par7 ) || fail=1
bash tools/best5/build_n.sh || fail=1
for l in tools/best5/libn_rr tools/best5/libn_rs tools/best5/libn_sel tools/best5/libn_ofc tools/best5/libn_norm tools/best5/libn_b2 tools/best5/libn_seg tools/timer5/libcsta5 tools/par7/libp7_pf tools/par7/libp7_dp; do [ -s $l.so ] || { echo "MISSING $l.so"; fail=1; }; done
echo "best5 build_all done (failures: $fail)"
exit $fail
