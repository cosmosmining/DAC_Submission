#!/bin/bash
# build_dg.sh: libdg_cpu.so, libdg_cpu_f64.so (g++); where nvcc exists libdg_cuda.so, libdg_cuda_f64.so. ARCH=sm_75 (T4), sm_90 (H100).
cd "$(dirname "$0")" || exit 1
ARCH=${ARCH:-sm_75}
g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math -fPIC -shared -pthread -x c++ dg_kern.cu -o libdg_cpu.so.tmp.$$ && mv -f libdg_cpu.so.tmp.$$ libdg_cpu.so || exit 1
g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math -fPIC -shared -pthread -DDG_F64 -x c++ dg_kern.cu -o libdg_cpu_f64.so.tmp.$$ && mv -f libdg_cpu_f64.so.tmp.$$ libdg_cpu_f64.so || exit 1
NV=${NVCC:-/usr/local/cuda/bin/nvcc}
if [ -x "$NV" ]; then
  $NV -O3 -std=c++17 -arch=$ARCH -fmad=false -Xcompiler -fPIC -shared dg_kern.cu -o libdg_cuda.so.tmp.$$ && mv -f libdg_cuda.so.tmp.$$ libdg_cuda.so || exit 1
  $NV -O3 -std=c++17 -arch=$ARCH -fmad=false -Xcompiler -fPIC -shared -DDG_F64 dg_kern.cu -o libdg_cuda_f64.so.tmp.$$ && mv -f libdg_cuda_f64.so.tmp.$$ libdg_cuda_f64.so || exit 1
fi
ls -la libdg_*.so
g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread -shared dgb.cc -o libdgb.so.tmp.$$ && mv -f libdgb.so.tmp.$$ libdgb.so && ls -la libdgb.so
