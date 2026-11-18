#!/bin/sh
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cxx=${CXX:-clang++}
common='-std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -fPIC'

case "$(uname -s)" in
  Darwin)
    exec "$cxx" $common -dynamiclib "$here/route_cpp.cc" -o "$here/libroute_cpp.dylib"
    ;;
  Linux)
    exec "$cxx" $common -shared "$here/route_cpp.cc" -o "$here/libroute_cpp.so"
    ;;
  *)
    echo "unsupported host: $(uname -s)" >&2
    exit 2
    ;;
esac
