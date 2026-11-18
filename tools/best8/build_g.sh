#!/bin/bash
# build_g.sh: libgnn_cpu.so (g++); where nvcc exists libgnn_cuda.so. ARCH=sm_75 (T4), sm_90 (H100).
cd "$(dirname "$0")" || exit 1
ARCH=${ARCH:-sm_75}
g++ -O3 -std=c++17 -fno-fast-math -fPIC -shared -pthread -x c++ gnn_kern.cu -o libgnn_cpu.so.tmp.$$ && mv -f libgnn_cpu.so.tmp.$$ libgnn_cpu.so || exit 1
NV=${NVCC:-/usr/local/cuda/bin/nvcc}
if [ -x "$NV" ]; then
  LB=$(dirname "$(dirname "$NV")")/lib64
  $NV -O3 -std=c++17 -arch=$ARCH -Xcompiler -fPIC -shared gnn_kern.cu -lcublas -Xlinker -rpath -Xlinker "$LB" -o libgnn_cuda.so.tmp.$$ && mv -f libgnn_cuda.so.tmp.$$ libgnn_cuda.so || exit 1
fi
ls -la libgnn_*.so
