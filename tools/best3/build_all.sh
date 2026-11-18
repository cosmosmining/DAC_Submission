#!/bin/bash
# build_all.sh [repo root]: builds, on a Linux x86 host, EVERY native library that tools/best3/chain_k.py
# and tools/best3/cold_k.py load (list = the ctypes.dlopen events of a cold_k run, an internal note section S2, plus the
# libraries of chain_k's flag-off paths: reshape and the chain_h helpers). Default root: two levels above this script.
# Flags = the ones each library's own build script or source header prescribes (never -ffast-math; no -march / AVX2: the target hosts are
# 2012 Xeons and the same flags must give the same bits everywhere). Each library is built to a temp name, then renamed.
# Needs: g++ with C++17 and OpenMP (GCC 8.5 on the hosts), external/InstantGR/src (Flute headers) for libflute_bridge.
# Prints one line per library; exits 1 if any build failed. ~2-4 min on one core of a host.
set -u
ROOT=$(cd "${1:-$(dirname "$0")/../..}" && pwd); cd "$ROOT" || exit 1
[ "$(uname)" = Linux ] || { echo "build_all.sh: Linux only (the Mac builds are the per-directory scripts)"; exit 2; }
CXX=${CXX:-g++}; echo "build_all in $ROOT with $($CXX --version | head -1)"
fail=0
b() { local out=$1; shift; if $CXX "$@" -o "$out.tmp.$$" && mv -f "$out.tmp.$$" "$out"; then echo "  built $out"; else rm -f "$out.tmp.$$"; echo "  BUILD FAILED: $out"; fail=1; fi; }
T=tools
# chain_i / chain_h helpers (= tools/best/build_i.sh)
b $T/best/libh_route_read.so -std=c++17 -O3 -DNDEBUG -fPIC -pthread -shared $T/best/h_route_read.cc
b $T/best/libh_iowrite.so    -std=c++17 -O2 -fPIC -pthread -shared $T/engine/io_write.cc
b $T/best/libi_kern.so       -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread -shared $T/best/i_kern.cc
b $T/best/libh_asm.so        -std=c++17 -O2 -fPIC -shared $T/best/h_asm.cc
b $T/best/libh_dem.so        -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fPIC -shared $T/best/h_dem.cc
# root-level kernels
b $T/libastar3d_linux.so -O3 -std=c++17 -shared -fPIC $T/astar3d.cpp
b $T/libflute_bridge.so  -O3 -shared -fPIC -std=c++17 -I external/InstantGR/src $T/flute_bridge.cpp
# promote / write / build / DP (flags of their headers and auto-builders)
b $T/c1hfast/libnorm_write.so       -O2 -std=c++17 -shared -fPIC $T/c1hfast/norm_write.cc
b $T/c1hfast/libpromote_kern.dylib  -O2 -std=c++17 -ffp-contract=off -shared -fPIC $T/c1hfast/promote_kern.cc   # Linux ELF: promote_cpp.py hard-codes this name
b $T/fast3x/libpromote_full.so      -O2 -std=c++17 -ffp-contract=off -shared -fPIC -pthread $T/fast3x/promote_full.cc
b $T/fast_path/libio_write.so       -O3 -std=c++17 -fPIC -shared $T/fast_path/io_write.cc
b $T/onepass/libbuild2_kern.so      -std=c++17 -O3 -fPIC -shared $T/onepass/build2_kern.cc
b $T/speedcut/libdp_split_kern.so   -O3 -std=c++17 -ffp-contract=off -fPIC -shared $T/speedcut/dp_split_kern.cc
# overflow stage (A* window, reroute loop), escape
b $T/ovf3/libastar3d_win_linux.so   -O3 -std=c++17 -shared -fPIC $T/ovf3/astar3d_win.cpp
b $T/wave3/ovf2/libw3_rr_linux.so   -std=c++17 -O3 -ffp-contract=off -fno-fast-math -shared -fPIC $T/wave3/ovf2/rr_loop.cc
b $T/speed4/libs4_esc.so            -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -shared $T/speed4/s4_esc.cc
b $T/speed4/libfastload.so          -std=c++17 -O3 -fopenmp -fPIC -shared $T/speed4/fastload.cc
# reshape (chain_k with --late5 0; = tools/reshape/build_r.sh, the two libraries chain_k can load from there)
b $T/reshape/librs_loop.so -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -shared $T/reshape/rs_loop.cc
b $T/reshape/librs_kern.so -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread -shared $T/reshape/r_kern.cc
# the three speed regions (= tools/pre5/build_pre5.sh without the AVX2 branch, tools/mid5/build_mid5.sh, tools/late5/build_l5.sh
# without regenerating l5_rs.cc)
b $T/pre5/libact_par.so  -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -fopenmp -shared $T/pre5/act_par.cc
b $T/pre5/libpre_kern.so -std=c++17 -O3 -fPIC -shared $T/pre5/pre_kern.cc
b $T/pre5/libgnn_kern.so -std=c++17 -O3 -fno-fast-math -fPIC -fopenmp -shared $T/pre5/gnn_kern.cc
b $T/mid5/libmid5.so     -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread -shared $T/mid5/mid5_kern.cc $T/mid5/mid5_rr.cc
b $T/late5/libl5_kern.so -O3 -std=c++17 -fPIC -shared -ffp-contract=off -fno-fast-math -pthread $T/late5/l5_kern.cc
b $T/late5/libl5_rs.so   -O3 -std=c++17 -fPIC -shared -ffp-contract=off -fno-fast-math -pthread $T/late5/l5_rs.cc
b $T/late5/libl5_pf.so   -O2 -std=c++17 -fPIC -shared -ffp-contract=off -fno-fast-math -pthread $T/late5/l5_pf.cc
# case-independent data the chain reads besides code: tools/pre5/weights/fold_*.npz (exported prior weights; NOT built here:
# sync_code.sh excludes *.npz, ship them with the code). Check:
for f in ariane bsg_chip MemPool NV_NVDLA_partition_c; do [ -s $T/pre5/weights/fold_$f.npz ] || { echo "  MISSING $T/pre5/weights/fold_$f.npz"; fail=1; }; done
echo "build_all done (failures: $fail)"
exit $fail
