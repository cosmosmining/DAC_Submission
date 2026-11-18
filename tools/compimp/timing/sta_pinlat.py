#!/usr/bin/env python3
"""sta_pinlat.py -- per-PIN ideal clock latency for the in-run timer fast_sta_v3 (new file, 2026-10-31; timing workstream
of the component-improve workflow). fast_sta_v3.py and every other repository file are imported read-only; the timer
class is patched in memory only (install()).

The bug. fast_sta_v3.parse_sdc reads `set_clock_latency` only for `[get_clocks ...]` targets. The mempool_group and
mempool_cluster SDCs (visible and blind) also set PIN latencies, which OpenSTA applies and fast_sta ignores:
  group v / b    65,616 / 66,280 pins: latch enables G of the icache L0 prefetchers at -0.23 (clock latency +0.07),
                 SRAM macro clk pins at +0.37, a few flop CK pins;
  cluster v / b  262,241 / 262,405 pins: latch G at -0.3 (no clock latency), SRAM clk at +0.3, a few CK.
The six small designs (ariane, bsg_chip, NV_NVDLA_partition_c, mempool_tile_wrap; visible and blind) set none, so the
patch changes nothing there (apply() returns before touching an array). Evidence: the official report of group v
F15T-U (F15R_pkg/experiments/ispd25_ppa/mempool_group/*f15tu_s0.crit.rpt) prints "clock network delay (ideal) -0.23000"
at the launching latch of its worst path, which fast_sta launches at +0.07; the in-run estimate of that route is WNS
-0.8409 / TNS -62,626 against the official -0.5495 / -21,095 (cluster v: -0.5788 / -226,440 vs -0.5250 / -64,664),
while on the small designs est and official agree within 1-2 %.

OpenSTA semantics (ideal clocks, setup): the network latency seen at a register clock pin p is p's own
set_clock_latency when one is set, else its clock's. With delta_p = L_pin(p) - L_clock(clock of p):
  launch   the arrival at the startpoint p (a register clock pin) is delta_p instead of 0, both edges;
  capture  every register check (setup / recovery) whose clock pin is p gets its required time later by delta_p,
           i.e. its row offset `off` (subtracted from the required time) decreases by delta_p.
Only values change (no graph, arrival class or setup relation changes); StaTimer.analyze and tools/fast_sta_incr.py
read D.sp_at and D.ck_rows["off"] at call time. Pins that are not register clock pins are counted, not modelled (none
in these SDCs after matching). Lines with -source / -rise / -fall / -min / -early / -late / -clock are counted and
skipped (none in the ISPD'25 SDCs).

  python tools/compimp/timing/sta_pinlat.py --self_test
  python tools/compimp/timing/sta_pinlat.py --count <sdc>...                      (pin-latency lines per SDC)
  python tools/compimp/timing/sta_pinlat.py check --design D --suite S --route R [--crit C.crit.rpt --ppa P.ppa.txt]
      (one StaDesign: analyze the route without, then with the pin latencies; WNS / TNS and, with --crit, the
       per-endpoint slack error against OpenROAD's report, as fast_sta_v3.calib computes it)
  in a driver: import sta_pinlat; sta_pinlat.install()   (before the first StaDesign is built)
"""
import argparse
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _root():
    d = HERE
    for _ in range(6):
        if os.path.exists(os.path.join(d, "fast_sta_v3.py")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("fast_sta_v3.py not found above " + HERE)


ROOT = _root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LAT = {"on": False, "log": []}
_PIN_LINE = re.compile(r"\[get_pins\s+(.*?)\]\s*$")
_SKIP_OPTS = ("-source", "-rise", "-fall", "-min", "-early", "-late", "-clock")


def pin_latencies(sdc_path):
    """`set_clock_latency <v> [get_pins {{name}} ...]` lines -> ({pin key: latency}, n_lines, n_skipped).
    The pin key is fast_sta's (fast_timer._key: backslashes removed)."""
    out, n, skipped = {}, 0, 0
    with open(sdc_path) as fh:
        for raw in fh:
            if not raw.startswith("set_clock_latency") or "get_pins" not in raw:
                continue
            line = raw.split("#")[0].strip()
            n += 1
            tok = line.split()
            if any(o in tok for o in _SKIP_OPTS) or len(tok) < 3:
                skipped += 1
                continue
            try:
                v = float(tok[1])
            except ValueError:
                skipped += 1
                continue
            m = _PIN_LINE.search(line)
            if not m:
                skipped += 1
                continue
            for t in m.group(1).split():
                k = t.strip("{}").replace("\\", "")
                if k:
                    out[k] = v
    return out, n, skipped


def apply(D, sdc_path):
    """add the SDC's pin latencies to a built fast_sta_v3.StaDesign D (after D._clocks) -> record dict."""
    t0 = time.time()
    lat, n_lines, n_skip = pin_latencies(sdc_path)
    rec = {"design": getattr(D, "design", "?"), "suite": getattr(D, "suite", "?"), "sdc": sdc_path,
           "pin_lines": n_lines, "skipped": n_skip, "pins": len(lat)}
    if not lat:
        rec.update(changed=False, t=time.time() - t0)
        LAT["log"].append(rec)
        return rec
    pid = D._pid
    ids, vals, unmatched = [], [], 0
    for k, v in lat.items():
        i = pid.get(k)
        if i is None:
            unmatched += 1
            continue
        ids.append(i)
        vals.append(v)
    ids = np.asarray(ids, np.int64)
    vals = np.asarray(vals, float)
    clk = D.clk_of[ids]
    reg = D.reg_ck[ids] & (clk >= 0)
    names, ck = D.clk_names, D.sdc["clocks"]
    lc = np.array([ck[names[c]]["lat"] if c >= 0 else 0.0 for c in clk], float)
    delta = np.zeros(D.P)
    delta[ids[reg]] = vals[reg] - lc[reg]
    # launch: startpoints at register clock pins (input ports are never register clock pins)
    sp = D.sp_pin
    sel = D.reg_ck[sp] & (delta[sp] != 0.0)
    sp_at = np.asarray(D.sp_at, float).copy()
    sp_at[sel] += delta[sp[sel]]
    # capture: the register-check rows are the first rows of ck_rows, in D.chk order filtered by _clocks' keep
    Cc = D.chk
    keep = D.reg_ck[Cc["cpin"]] & (D.const[Cc["pin"]] < 0)
    if (D.const >= 0).any():
        keep &= ~D._when_false(Cc["pin"], Cc["when"])
    nreg = int(keep.sum())
    if not np.array_equal(D.ck_rows["pin"][:nreg], Cc["pin"][keep]):
        raise RuntimeError("sta_pinlat: check rows are not in D.chk order -- fast_sta_v3 changed; nothing applied")
    cp = Cc["cpin"][keep]
    off = np.asarray(D.ck_rows["off"], float).copy()
    d_cap = delta[cp]
    off[:nreg] -= d_cap
    D.sp_at = sp_at
    D.ck_rows["off"] = off
    vv = vals[reg] - lc[reg]
    rec.update(changed=True, matched=int(len(ids)), unmatched=unmatched, reg_ck=int(reg.sum()),
               not_reg=int((~reg).sum()), startpoints=int(sel.sum()), check_rows=int((d_cap != 0).sum()),
               delta_values={f"{v:+.4f}": int(c) for v, c in zip(*np.unique(np.round(vv, 4), return_counts=True))}
               if len(vv) <= 200000 else {}, t=time.time() - t0)
    if len(rec["delta_values"]) > 12:                          # keep the record short: the most frequent values
        top = sorted(rec["delta_values"].items(), key=lambda kv: -kv[1])[:12]
        rec["delta_values"] = dict(top)
    LAT["log"].append(rec)
    return rec


def _log(m):
    print(f"[sta_pinlat {time.strftime('%H:%M:%S')}] {m}", flush=True)


def install(fs_mod=None):
    """wrap StaDesign._clocks of fast_sta_v3 (and of the module aliased as fast_sta, if different) so that every
    design built afterwards gets its SDC's pin latencies. Idempotent."""
    mods = []
    if fs_mod is not None:
        mods.append(fs_mod)
    import fast_sta_v3
    mods.append(fast_sta_v3)
    if "fast_sta" in sys.modules:
        mods.append(sys.modules["fast_sta"])
    for M in mods:
        cls = M.StaDesign
        if getattr(cls, "_pinlat_wrapped", False):
            continue
        orig = cls._clocks

        def _clocks(self, sdc_path, _orig=orig):
            _orig(self, sdc_path)
            rec = apply(self, sdc_path)
            if rec.get("changed"):
                _log(f"{rec['design']}[{rec['suite']}]: {rec['pins']:,} pin latencies ({rec['matched']:,} matched, "
                     f"{rec['reg_ck']:,} register clock pins, {rec['not_reg']} other, {rec['unmatched']} unmatched, "
                     f"{rec['skipped']} skipped lines) -> {rec['startpoints']:,} startpoint arrivals and "
                     f"{rec['check_rows']:,} check offsets shifted; delta {rec['delta_values']} ({rec['t']:.1f}s)")
            else:
                _log(f"{rec['design']}[{rec['suite']}]: no pin latency in the SDC ({rec['pin_lines']} lines) -> "
                     f"timer unchanged")

        cls._clocks = _clocks
        cls._pinlat_wrapped = True
    LAT["on"] = True


# ─────────────────────────────── self-test (tiny design, no contest files) ───────────────────────────────

def self_test():
    import tempfile
    import fast_sta_v3 as FS
    import fast_timer as FT
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name} {info}")

    lib = FS.StaLib()
    lib._add(FT._parse_groups(FS._TINY_LIB))
    lib.tab.freeze()
    d = tempfile.mkdtemp(prefix="pinlat_")
    kit = os.path.join(d, "kit")
    os.makedirs(os.path.join(kit, "visible/tiny"))
    os.makedirs(os.path.join(kit, "NanGate45/lib"))
    open(os.path.join(kit, "NanGate45/lib/tiny.lib"), "w").write(FS._TINY_LIB)
    comps = [("ff1", "DFF"), ("ff2", "DFF"), ("u1", "INV"), ("g1", "AND2"), ("lat", "DLL")]
    nets = {"clk": [("PIN", "clk"), ("ff1", "CK"), ("ff2", "CK"), ("g1", "A1")],
            "n1": [("ff1", "Q"), ("u1", "A"), ("lat", "D")],
            "n2": [("u1", "ZN"), ("ff2", "D"), ("PIN", "out1")],
            "n3": [("PIN", "in1"), ("g1", "A2")],
            "gck": [("g1", "ZN"), ("lat", "GN")]}
    with open(os.path.join(kit, "visible/tiny/tiny.def"), "w") as fh:
        fh.write(f"COMPONENTS {len(comps)} ;\n" + "".join(f"- {a} {b} + PLACED ( 0 0 ) N ;\n" for a, b in comps)
                 + "END COMPONENTS\nPINS 3 ;\n- clk + NET clk + DIRECTION INPUT ;\n- in1 + NET n3 + DIRECTION INPUT ;\n"
                 "- out1 + NET n2 + DIRECTION OUTPUT ;\nEND PINS\n"
                 f"NETS {len(nets)} ;\n" + "".join(f"- {n}\n  " + " ".join(f"( {a} {b} )" for a, b in cs) + " ;\n"
                                                    for n, cs in nets.items()) + "END NETS\n")
    aps = {"ff1/Q": (0, 1, 1), "u1/A": (0, 2, 1), "lat/D": (0, 1, 2), "u1/ZN": (0, 2, 1), "ff2/D": (0, 52, 1),
           "out1": (0, 52, 2)}
    with open(os.path.join(kit, "visible/tiny/tiny.net"), "w") as fh:
        for n in ("n1", "n2"):
            fh.write(f"{n}\n(\n" + "".join(
                f"{(a + '/' + b) if a != 'PIN' else b}, -1e-11, [{aps[(a + '/' + b) if a != 'PIN' else b]}]\n"
                for a, b in nets[n]) + ")\n")
    base_sdc = ("create_clock [get_ports {clk}] -name core -period 1.0 -waveform {0 0.5}\n"
                "set_clock_latency 0.07  [get_clocks {core}]\n"
                "set_input_delay 0.1 -clock [get_clocks {core}] [get_ports {in1}]\n"
                "set_output_delay 0.2 -clock [get_clocks {core}] [get_ports {out1}]\n"
                "set_clock_uncertainty 0.05 [get_clocks {core}]\n")
    g = lambda v: v * FS.GRID + FS.GRID // 2
    uv = lambda x, y, z0, z1: "".join(f"{g(x)} {g(y)} metal{z} {g(x)} {g(y)} metal{z + 1}\n" for z in range(z0, z1))
    r = (f"n1\n(\n" + uv(1, 1, 1, 3) + f"{g(1)} {g(1)} metal3 {g(2)} {g(1)} metal3\n" + uv(2, 1, 1, 3)
         + f"{g(1)} {g(1)} metal2 {g(1)} {g(2)} metal2\n" + uv(1, 2, 1, 2) + ")\n")
    r += (f"n2\n(\n" + uv(2, 1, 1, 3) + f"{g(2)} {g(1)} metal3 {g(52)} {g(1)} metal3\n" + uv(52, 1, 1, 3)
          + f"{g(52)} {g(1)} metal2 {g(52)} {g(2)} metal2\n" + uv(52, 2, 1, 2) + ")\n")
    rpath = os.path.join(d, "r.route")
    open(rpath, "w").write(r)

    def run(extra):
        open(os.path.join(kit, "visible/tiny/tiny.sdc"), "w").write(base_sdc + extra)
        D = FS.StaDesign("tiny", "visible", lib=lib, kit=kit, verbose=False)
        sp0, off0 = np.asarray(D.sp_at, float).copy(), np.asarray(D.ck_rows["off"], float).copy()
        rec = apply(D, os.path.join(kit, "visible/tiny/tiny.sdc"))
        res = FS.StaTimer(D).analyze(FS.read_route(D, rpath), keep=False)
        sl = dict(zip([D.pin_name[i] for i in res["ep_pin"]], res["ep_slack"]))
        return D, rec, sl, sp0, off0

    D0, rec0, s0, sp0, off0 = run("")
    check("no pin latency: nothing changed", rec0["changed"] is False and np.array_equal(sp0, D0.sp_at)
          and np.array_equal(off0, D0.ck_rows["off"]), f"{rec0}")
    # launch latency on ff1/CK (pin -0.23, clock +0.07: delta -0.30): every path from ff1 gains 0.30
    D1, rec1, s1, _, _ = run("set_clock_latency -0.23  [get_pins {{ff1/CK}}]\n")
    check("launch pin latency: ff2/D and lat/D slack +0.30, in1->g1 unchanged",
          abs((s1["ff2/D"] - s0["ff2/D"]) - 0.30) < 1e-9 and abs((s1["lat/D"] - s0["lat/D"]) - 0.30) < 1e-9
          and abs(s1["g1/A2"] - s0["g1/A2"]) < 1e-12 and abs(s1["out1"] - s0["out1"] - 0.30) < 1e-9,
          f"ff2/D {s0['ff2/D']:.4f}->{s1['ff2/D']:.4f}, lat/D {s0['lat/D']:.4f}->{s1['lat/D']:.4f}, rec {rec1}")
    # capture latency on ff2/CK (pin +0.37: delta +0.30): only ff2/D gains 0.30
    D2, rec2, s2, _, _ = run("set_clock_latency 0.37  [get_pins {{ff2/CK}}]\n")
    check("capture pin latency: ff2/D slack +0.30, others unchanged",
          abs((s2["ff2/D"] - s0["ff2/D"]) - 0.30) < 1e-9 and abs(s2["lat/D"] - s0["lat/D"]) < 1e-12
          and abs(s2["out1"] - s0["out1"]) < 1e-12, f"ff2/D {s0['ff2/D']:.4f}->{s2['ff2/D']:.4f}")
    # latch enable latency on lat/GN (-0.23: delta -0.30): the check at lat/D loses 0.30 (earlier closing edge)
    D3, rec3, s3, _, _ = run("set_clock_latency -0.23  [get_pins {{lat/GN}}]\n")
    check("latch enable pin latency: lat/D slack -0.30, ff2/D unchanged",
          abs((s3["lat/D"] - s0["lat/D"]) + 0.30) < 1e-9 and abs(s3["ff2/D"] - s0["ff2/D"]) < 1e-12,
          f"lat/D {s0['lat/D']:.4f}->{s3['lat/D']:.4f}")
    # both ends of ff1 -> ff2 at the same pin latency: the relation is unchanged (only the difference counts)
    D4, rec4, s4, _, _ = run("set_clock_latency 0.2  [get_pins {{ff1/CK}}]\nset_clock_latency 0.2  [get_pins {{ff2/CK}}]\n")
    check("same pin latency at launch and capture: ff2/D unchanged", abs(s4["ff2/D"] - s0["ff2/D"]) < 1e-9,
          f"{s0['ff2/D']:.6f} vs {s4['ff2/D']:.6f}")
    # a non-register pin (g1/A1, clock network) and an unknown pin: counted, not modelled
    D5, rec5, s5, _, _ = run("set_clock_latency 0.5  [get_pins {{g1/A1}}]\nset_clock_latency 0.5  [get_pins {{nope/CK}}]\n")
    check("non-register / unknown pins counted, slacks unchanged", rec5["not_reg"] == 1 and rec5["unmatched"] == 1
          and all(abs(s5[k] - s0[k]) < 1e-12 for k in s0), f"{rec5}")
    # parser: the real SDC line format, and the options it must skip
    tmp = os.path.join(d, "p.sdc")
    open(tmp, "w").write("set_clock_latency -0.23  [get_pins {{gen_tiles[10].i_tile/a_reg[2][2]/G}}]\n"
                         "set_clock_latency 0.07  [get_clocks {clk_i}]\n"
                         "set_clock_latency -source 0.1 [get_pins {{x/CK}}]\n"
                         "set_clock_latency 0.37  [get_pins {{gen_tiles[3].i_tile/bank\\[1\\]/clk}}]\n")
    lat, n, sk = pin_latencies(tmp)
    check("parser: brackets, escapes, skipped options", lat == {"gen_tiles[10].i_tile/a_reg[2][2]/G": -0.23,
                                                                 "gen_tiles[3].i_tile/bank[1]/clk": 0.37}
          and n == 3 and sk == 1, f"{lat} lines {n} skipped {sk}")
    print("PINLAT SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ─────────────────────────────── check against an official report ───────────────────────────────

def check_route(a):
    if os.environ.get("FAST_STA_V3", "1"):
        import fast_sta_v3 as FS
    D = FS.StaDesign(a.design, a.suite)
    seg = FS.read_route(D, a.route)
    Tm = FS.StaTimer(D)
    rc = FS.build_rc(D, seg)
    out = {}
    cps = FS.parse_crit_paths(a.crit) if a.crit else []
    official = None
    if a.ppa:
        import fast_timer as FT
        official = FT.parse_ppa(a.ppa)
    for arm in ("old", "pinlat"):
        if arm == "pinlat":
            sdc = os.path.join(FS.KIT, a.suite, a.design, a.design + ".sdc")
            rec = apply(D, sdc)
            _log(f"applied: { {k: v for k, v in rec.items() if k != 'sdc'} }")
        t0 = time.time()
        r = Tm.analyze(seg, rat=False, rc=rc)
        row = {"wns": r["wns"], "tns": r["tns"], "nviol": r["nviol"], "n_ep": r["n_ep"], "t": time.time() - t0}
        if cps:
            est = np.full(D.P, np.nan)
            est[r["ep_pin"]] = r["ep_slack"]
            pairs, seen = [], set()
            for cp in cps:
                p = D._pid.get(cp["end"])
                if p is None or cp["end"] in seen or not np.isfinite(est[p]):
                    continue
                seen.add(cp["end"])
                pairs.append((cp["slack"], est[p], cp["group"]))
            if pairs:
                so = np.array([x[0] for x in pairs]); se = np.array([x[1] for x in pairs])
                row.update(ep_n=len(pairs), ep_mae=float(np.mean(np.abs(se - so))), ep_bias=float(np.mean(se - so)),
                           ep_r=float(np.corrcoef(so, se)[0, 1]) if len(pairs) > 2 and so.std() > 0 else float("nan"))
                for grp in sorted(set(x[2] for x in pairs)):
                    m = np.array([x[2] == grp for x in pairs])
                    row[f"mae[{grp}]"] = float(np.mean(np.abs(se[m] - so[m])))
                    row[f"bias[{grp}]"] = float(np.mean(se[m] - so[m]))
        out[arm] = row
        _log(f"{a.design}[{a.suite}] {arm}: WNS {r['wns']:.4f} TNS {r['tns']:.1f} ({r['nviol']}/{r['n_ep']} viol)"
             + (f" | official WNS {official[0]:.4f} TNS {official[1]:.1f}" if official else "")
             + (f" | endpoints {row.get('ep_n', 0)} mae {row.get('ep_mae', float('nan')):.4f} bias "
                f"{row.get('ep_bias', float('nan')):+.4f} r {row.get('ep_r', float('nan')):.4f}" if cps else "")
             + f" ({row['t']:.0f}s)")
        for k in sorted(row):
            if k.startswith("mae[") or k.startswith("bias["):
                _log(f"    {k} {row[k]:+.4f}")
    if a.json:
        import json
        with open(a.json, "w") as fh:
            json.dump({"design": a.design, "suite": a.suite, "route": a.route, "official": official, "arms": out,
                       "pinlat": [{k: v for k, v in x.items()} for x in LAT["log"]]}, fh, indent=1, default=str)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("check",), default=None)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--count", nargs="*")
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible")
    ap.add_argument("--route")
    ap.add_argument("--crit")
    ap.add_argument("--ppa")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.count:
        for p in a.count:
            lat, n, sk = pin_latencies(p)
            vals = np.array(list(lat.values())) if lat else np.zeros(0)
            u, c = np.unique(np.round(vals, 3), return_counts=True) if len(vals) else ([], [])
            top = sorted(zip(c, u), reverse=True)[:5]
            print(f"{p}: {n} pin-latency lines, {len(lat)} pins, {sk} skipped; top values {[(float(v), int(k)) for k, v in top]}")
        return 0
    if a.cmd == "check":
        return check_route(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
