#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tree_topology.py — per-net tree topologies for the ISPD'24/'25 flow.

ispd24_adapter.convert() connects a net's pins with a NEAREST-NEIGHBOUR CHAIN. Measured on
ISPD'24 ariane133_51 (topology_gap.py): the chain is 11.97% longer than a minimum spanning
tree in total, 28.6% longer on 10+-pin nets - and that matches the 12.5% wirelength gap
between every method here and the organizers' example solution. This module provides
deterministic tree orders so every consumer (pack builder, pattern-L, maze router, DGR
guide reader) agrees on the same (child, parent) pairs.

    tree_order(points, kind) -> (order, parent)   order[0] is the root, parent[k] < k
        kind="chain": convert()'s nearest-neighbour chain (parent[k] = k - 1)
        kind="mst":   Prim's minimum spanning tree on Manhattan distance, rooted at
                      points[0], ties broken by insertion index (deterministic)
    tree_pairs(points, kind) -> [(child, parent), ...]
NEW file.
"""
import numpy as np


def _chain(points):
    order, rest = [0], set(range(1, len(points)))
    while rest:
        lx, ly = points[order[-1]]
        nxt = min(rest, key=lambda j: abs(points[j][0] - lx) + abs(points[j][1] - ly))
        order.append(nxt)
        rest.remove(nxt)
    return [points[j] for j in order], [-1] + list(range(len(order) - 1))


def _mst(points):
    n = len(points)
    p = np.asarray(points, dtype=np.int64)
    inside = np.zeros(n, dtype=bool)
    inside[0] = True
    dist = np.abs(p - p[0]).sum(1)
    link = np.zeros(n, dtype=np.int64)            # nearest in-tree node (original index)
    order, pos = [0], {0: 0}
    parent = [-1]
    big = np.iinfo(np.int64).max
    for _ in range(n - 1):
        d = np.where(inside, big, dist)
        j = int(d.argmin())                        # first minimum -> deterministic
        inside[j] = True
        parent.append(pos[int(link[j])])
        pos[j] = len(order)
        order.append(j)
        nd = np.abs(p - p[j]).sum(1)
        better = nd < dist
        dist = np.where(better, nd, dist)
        link = np.where(better, j, link)
    return [points[j] for j in order], parent


def tree_order(points, kind="chain"):
    points = list(points)
    if len(points) <= 1:
        return points, [-1] * len(points)
    if kind == "mst":
        return _mst(points)
    if kind == "chain":
        return _chain(points)
    raise ValueError(kind)


def tree_pairs(points, kind="chain"):
    order, parent = tree_order(points, kind)
    return [(order[k], order[parent[k]]) for k in range(1, len(order))]


def total_length(points, kind):
    return sum(abs(c[0] - p[0]) + abs(c[1] - p[1]) for c, p in tree_pairs(points, kind))


if __name__ == "__main__":
    pts = [(0, 0), (10, 0), (1, 1), (11, 1), (5, 8)]
    o, par = tree_order(pts, "mst")
    assert o[0] == (0, 0) and all(0 <= par[k] < k for k in range(1, len(o)))
    assert total_length(pts, "mst") <= total_length(pts, "chain")
    assert len(tree_pairs(pts, "mst")) == len(pts) - 1
    o2, par2 = tree_order(pts, "chain")
    assert par2 == [-1, 0, 1, 2, 3]
    print(f"  [PASS] MST {total_length(pts, 'mst')} <= chain {total_length(pts, 'chain')}, "
          f"deterministic, parent[k] < k")
