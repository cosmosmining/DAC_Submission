#!/bin/bash
# build8.sh: builds the load8 libraries next to this script (login nodes and hosts: g++; no -ffast-math).
cd "$(dirname "$0")" || exit 1
set -e
${PY:-python3} -I mk_i8.py > /dev/null
CXX=${CXX:-g++}
b() { local out=$1; shift; $CXX "$@" -o "$out.tmp.$$" && mv -f "$out.tmp.$$" "$out"; echo "built $out"; }
b libfl8.so -std=c++17 -O3 -fopenmp -fPIC -shared fl8.cc
b libl8_kern.so -std=c++17 -O3 -fopenmp -fPIC -shared l8_kern.cc
b libi8_kern.so -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread -shared gen8/i8_kern.cc
