#!/usr/bin/env python3
"""ctypes/numpy binding for the C++17 contest-route reader and writer.

The public arrays use contest route units: x/y are DBU integers and z is a
zero-based metal index. ``net_id`` is the zero-based net-block index. A
``Route`` retains net names and block boundaries so ``Route.write`` emits the
same canonical bytes as ``route_export.seg_line(..., fmt="ispd25")``.

Build once with ``./build.sh``. The module has no dependency beyond numpy,
ctypes, and the system C++ runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_HERE = Path(__file__).resolve().parent
_LIB = _HERE / ("libroute_cpp.dylib" if os.uname().sysname == "Darwin" else "libroute_cpp.so")
_I32P = ctypes.POINTER(ctypes.c_int32)


def _load() -> ctypes.CDLL:
    if not _LIB.exists():
        raise RuntimeError(f"{_LIB} is missing; run {_HERE / 'build.sh'} first")
    lib = ctypes.CDLL(str(_LIB))
    lib.route_read.argtypes = [ctypes.c_char_p]
    lib.route_read.restype = ctypes.c_void_p
    lib.route_free.argtypes = [ctypes.c_void_p]
    lib.route_error.restype = ctypes.c_char_p
    for name in ("route_num_nets", "route_num_segments"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
        getattr(lib, name).restype = ctypes.c_size_t
    for name in ("route_net_id", "route_x0", "route_y0", "route_z0", "route_x1", "route_y1", "route_z1"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
        getattr(lib, name).restype = _I32P
    lib.route_offsets.argtypes = [ctypes.c_void_p]
    lib.route_offsets.restype = ctypes.POINTER(ctypes.c_uint64)
    lib.route_net_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.route_net_name.restype = ctypes.c_char_p
    lib.route_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _I32P, _I32P, _I32P, _I32P, _I32P, _I32P, ctypes.c_size_t]
    lib.route_write.restype = ctypes.c_int
    lib.route_convert_instantgr.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.route_convert_instantgr.restype = ctypes.c_int64
    return lib


def _as_array(ptr, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int32)
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def _bytes_path(path: os.PathLike[str] | str) -> bytes:
    return os.fsencode(os.fspath(path))


@dataclass
class Route:
    """Parsed route arrays and the metadata needed for a byte-stable writer."""

    net_id: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    z1: np.ndarray
    offsets: np.ndarray
    net_names: tuple[str, ...]
    _handle: ctypes.c_void_p | None = None
    _lib: ctypes.CDLL | None = None

    @property
    def n_segments(self) -> int:
        return int(self.x0.size)

    @property
    def n_nets(self) -> int:
        return len(self.net_names)

    @property
    def layers(self) -> np.ndarray:
        return np.column_stack((self.z0, self.z1))

    @property
    def segments(self) -> np.ndarray:
        """Return ``(x0, y0, x1, y1, z0, z1)`` as an int32 array."""
        return np.column_stack((self.x0, self.y0, self.x1, self.y1, self.z0, self.z1))

    def write(self, path: os.PathLike[str] | str, segments: np.ndarray | None = None) -> None:
        """Write canonical contest bytes, optionally replacing segment columns.

        ``segments`` has six columns in the order returned by ``segments``.
        Net IDs and block boundaries stay fixed, which is what preserves route
        topology while a layer-assignment pass changes z columns.
        """
        if self._handle is None or self._lib is None:
            raise RuntimeError("Route metadata handle is no longer alive")
        values = self.segments if segments is None else np.asarray(segments)
        if values.shape != (self.n_segments, 6):
            raise ValueError(f"segments must have shape {(self.n_segments, 6)}")
        if values.dtype != np.int32:
            values = values.astype(np.int32, copy=False)
        values = np.ascontiguousarray(values)
        # The C writer takes separate column pointers; keep this contiguous
        # owner alive for the duration of the call.
        # C ABI order is x0, y0, z0, x1, y1, z1; the public matrix keeps
        # endpoints together as x0, y0, x1, y1, z0, z1.
        cols = [np.ascontiguousarray(values[:, i], dtype=np.int32) for i in (0, 1, 4, 2, 3, 5)]
        result = self._lib.route_write(self._handle, _bytes_path(path), *(c.ctypes.data_as(_I32P) for c in cols), self.n_segments)
        if result != 0:
            raise RuntimeError(self._lib.route_error().decode("utf-8", "replace"))

    def close(self) -> None:
        if self._handle is not None and self._lib is not None:
            self._lib.route_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def read_route(path: os.PathLike[str] | str) -> Route:
    """Parse a contest .route into numpy arrays."""
    lib = _load()
    handle = lib.route_read(_bytes_path(path))
    if not handle:
        raise RuntimeError(lib.route_error().decode("utf-8", "replace"))
    try:
        nets = int(lib.route_num_nets(handle))
        segments = int(lib.route_num_segments(handle))
        names = tuple(lib.route_net_name(handle, i).decode("utf-8") for i in range(nets))
        arrays = dict(
            net_id=_as_array(lib.route_net_id(handle), segments),
            x0=_as_array(lib.route_x0(handle), segments),
            y0=_as_array(lib.route_y0(handle), segments),
            z0=_as_array(lib.route_z0(handle), segments),
            x1=_as_array(lib.route_x1(handle), segments),
            y1=_as_array(lib.route_y1(handle), segments),
            z1=_as_array(lib.route_z1(handle), segments),
        )
        offsets = np.ctypeslib.as_array(lib.route_offsets(handle), shape=(nets + 1,)).copy()
        return Route(**arrays, offsets=offsets, net_names=names, _handle=handle, _lib=lib)
    except Exception:
        lib.route_free(handle)
        raise


load_route = read_route


def convert_instantgr(route_in: os.PathLike[str] | str, route_out: os.PathLike[str] | str) -> int:
    """Convert six-integer InstantGR output to contest route bytes in C++."""
    lib = _load()
    count = int(lib.route_convert_instantgr(_bytes_path(route_in), _bytes_path(route_out)))
    if count < 0:
        raise RuntimeError(lib.route_error().decode("utf-8", "replace"))
    return count


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_route = sub.add_parser("route", help="convert InstantGR six-integer route to contest route")
    p_route.add_argument("--in", dest="route_in", required=True)
    p_route.add_argument("--out", dest="route_out", required=True)
    p_read = sub.add_parser("read", help="parse and report route array counts")
    p_read.add_argument("route")
    args = parser.parse_args(argv)
    if args.command == "route":
        print(f"wrote {args.route_out}: {convert_instantgr(args.route_in, args.route_out):,} segments")
        return 0
    route = read_route(args.route)
    print(f"nets={route.n_nets:,} segments={route.n_segments:,}")
    route.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
