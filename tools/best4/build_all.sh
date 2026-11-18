#!/bin/bash
# build_all.sh [repo root]: every native library tools/best4/chain_m.py / cold_m.py load on a Linux
# x86 host = tools/best3/build_all.sh's 26 (chain_k; big5 adds no library of its own, an internal note section 10.2) + the binary-route
# reader tools/router5/libr5bin.so (--route_bin 1; flags of tools/router5/r5bin.py's header). Exits 1 if any build failed.
# The router binary itself (InstantGR_r5b) is built by InstantGR's own build (not included).
set -u
ROOT=$(cd "${1:-$(dirname "$0")/../..}" && pwd); cd "$ROOT" || exit 1
bash tools/best3/build_all.sh "$ROOT"; fail=$?
CXX=${CXX:-g++}
if $CXX -O3 -std=c++17 -fPIC -shared -pthread tools/router5/r5bin.cc -o tools/router5/libr5bin.so.tmp.$$ && \
   mv -f tools/router5/libr5bin.so.tmp.$$ tools/router5/libr5bin.so; then echo "  built tools/router5/libr5bin.so"
else rm -f tools/router5/libr5bin.so.tmp.$$; echo "  BUILD FAILED: tools/router5/libr5bin.so"; fail=1; fi
echo "best4 build_all done (failures: $fail)"
exit $fail
