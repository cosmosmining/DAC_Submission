#!/usr/bin/env python3
"""ispd25_score.py — official ISPD'25 quality score + every published number to beat.

  score = w1*(WNS - r1) + w2*(TNS - r2)/N_end + w3*(power - r3) + w4*tof      (lower = better)

w1..w4 are read from the design's .cap (header line 2); r1..r3 and N_end come from GAP-LA
Table II (arXiv 2507.13375v?, "GAP-LA: GPU-Accelerated Performance-Driven Layer Assignment");
tof = the official evaluator's overflow cost. Table II misprints case 1's weights as
10/100/300/3e-6 — the .cap and a recomputation of Table III both give -10/-100/300/3e-7.

REPORTED numbers (not reproduced by us) from GAP-LA Table IV (metrics), Table III (scores of
the ISPD'25 top-3 and GAP-LA) and Table V (scores of "winner 2-D route + GAP-LA layer
assignment"). The per-case bar to beat is the minimum over all of them.

  python ispd25_score.py --design ariane --ppa X.ppa.txt --eval X.evaluator.txt
  python ispd25_score.py --table            # the bars
  python ispd25_score.py --self_test        # recompute Table III from Table IV
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CAPDIR = os.path.join(ROOT, "ispd_bench/ISPD2025/ISPD2025_benchmarks")

# case: (design dir name, r1, r2, r3, N_end, weights from Table II [fixed for case 1])
CASES = {
    1: ("ariane", -0.485, -1398.39, 0.646, 20218, (-10, -100, 300, 3e-7)),
    2: ("bsg_chip", -0.44, -10802.7, 3.05, 214821, (-10, -100, 25, 4e-8)),
    3: ("NV_NVDLA_partition_c", -94.78, -669471, 2.96, 45925, (-0.05, -0.5, 25, 1.5e-7)),
    4: ("mempool_tile_wrap", -0.695, -3590.83, 0.1455, 13350, (-1, -10, 300, 7e-7)),
    5: ("mempool_group", -0.815, -41740.2, 7.82, 347869, (-1, -10, 20, 3e-8)),
    6: ("mempool_cluster", -0.68, -79748, 23.7, 1082397, (-1, -10, 0.3, 5e-9)),
    7: ("ariane@blind", -1.628, -523.038, 0.156, 20218, (-0.2, -2, 100, 4e-7)),
    8: ("bsg_chip@blind", 0, 0, 0.305, 214915, (-0.1, -1, 50, 2e-8)),
    9: ("NV_NVDLA_partition_c@blind", 0, 0, 0.136, 45925, (-0.01, -0.1, 100, 1e-7)),
    10: ("mempool_tile_wrap@blind", -0.483, -1920.724, 0.145, 13350, (-3, -30, 100, 1e-6)),
    11: ("mempool_group@blind", -0.68, -40487.883, 8.547, 347869, (-0.5, -5, 3, 4e-8)),
    12: ("mempool_cluster@blind", -0.29, -53249, 24.261, 1082397, (-0.4, -4, 2, 1e-8)),
}
# Table IV: case -> method -> (WNS, TNS, power, congestion)  [reported; obvious sign typos fixed]
T4 = {
    1: {"1st": (-0.432, -1325.8, 0.646092, 5846461), "2nd": (-0.398, -1340.2, 0.646078, 5902619),
        "3rd": (-0.381, -1138.4, 0.646080, 8003814), "GAP-LA": (-0.365, -997.9, 0.646076, 6662339)},
    2: {"1st": (-0.420, -10366.2, 3.05441, 21332547), "2nd": (-0.424, -10774.9, 3.05301, 20468992),
        "3rd": (-0.407, -9344.3, 3.05345, 25412296), "GAP-LA": (-0.388, -9602.9, 3.05349, 22947622)},
    3: {"1st": (-56.919, -521344.6, 2.94054, 13315606), "2nd": (-55.841, -514660.3, 2.94011, 13686714),
        "3rd": (-55.914, -514674.3, 2.93806, 15687876), "GAP-LA": (-56.253, -515014.3, 2.93766, 13562374)},
    4: {"1st": (-0.379, -1610.0, 0.14310, 3019425), "2nd": (-0.489, -2225.5, 0.14515, 2478318),
        "3rd": (-0.520, -2524.3, 0.14522, 3469019), "GAP-LA": (-0.387, -1711.6, 0.14383, 2689765)},
    5: {"1st": (-0.305, -20473.8, 7.69208, 55005746), "2nd": (-0.347, -21709.1, 7.73386, 49071540),
        "3rd": (-0.275, -19338.7, 7.55296, 97506553), "GAP-LA": (-0.327, -17619.7, 7.59214, 67934233)},
    6: {"1st": (-0.266, -59712.5, 23.4497, 236920427), "2nd": (-0.315, -62460.0, 23.5781, 192012708),
        "3rd": (-0.276, -59533.5, 23.2200, 289712328), "GAP-LA": (-0.247, -52800.0, 23.1486, 244407995)},
    7: {"1st": (-1.756, -650.1, 0.15614, 4349151), "2nd": (-0.737, -149.5, 0.15613, 4651055),
        "3rd": (-0.633, -162.4, 0.15611, 4386728), "GAP-LA": (-0.645, -111.5, 0.15614, 4371048)},
    8: {"1st": (-3.984, -2859.8, 0.30473, 22567636), "2nd": (0.0, 0.0, 0.30477, 21307154),
        "3rd": (-2.747, -2936.0, 0.30478, 20564539), "GAP-LA": (0.0, 0.0, 0.30481, 22506663)},
    9: {"1st": (0.0, 0.0, 0.13625, 14061155), "2nd": (0.0, 0.0, 0.13626, 16352692),
        "3rd": (0.0, 0.0, 0.13628, 13637302), "GAP-LA": (0.0, 0.0, 0.13629, 13242873)},
    10: {"1st": (-0.342, -1356.1, 0.14303, 2232008), "2nd": (-0.472, -1920.7, 0.14485, 2122543),
         "3rd": (-0.440, -1806.8, 0.14520, 2168599), "GAP-LA": (-0.353, -1312.2, 0.14390, 2087873)},
    11: {"1st": (-0.433, -32016.1, 8.36880, 52783071), "2nd": (-0.460, -34846.1, 8.35859, 49722314),
         "3rd": (-0.373, -26801.4, 7.94192, 132027917), "GAP-LA": (-0.404, -27116.0, 8.19256, 68117588)},
    12: {"1st": (-0.223, -42564.1, 24.2439, 242737068), "2nd": (-0.294, -50372.4, 24.3801, 214110807),
         "3rd": (-0.200, -39516.2, 23.9679, 335020596), "GAP-LA": (-0.199, -38106.6, 24.0595, 232170935)},
}
# Table III scores [reported]: 1st, 2nd, 3rd, GAP-LA ; Table V: 1st+GAP-LA, 2nd+, 3rd+, [26]+
T3 = {1: (0.896, 0.639, 0.094, -1.158), 2: (0.562, 0.722, 0.094, -0.078),
      3: (-1.995, -2.077, -1.824, -2.132), 4: (-0.405, 0.401, 1.370, -0.333),
      5: (-2.029, -1.295, -3.599, -3.700), 6: (0.510, 0.399, 0.714, 0.375),
      7: (1.792, 1.659, 1.531, 1.525), 8: (0.849, 0.415, 0.689, 0.441),
      9: (1.431, 1.661, 1.392, 1.354), 10: (0.344, 2.074, 1.803, 0.221),
      11: (1.332, 1.232, 3.116, 1.331), 12: (2.327, 2.370, 2.677, 1.826)}
T5 = {1: (-1.161, -1.223, -1.182, -1.189), 2: (-0.054, -0.127, -0.196, 0.013),
      3: (-1.996, -1.965, -2.023, -2.032), 4: (-0.369, -0.453, -0.435, -0.319),
      5: (-3.808, -6.167, -4.456, -6.015), 6: (0.384, 0.366, 0.329, 0.397),
      7: (1.532, 1.535, 1.524, 1.551), 8: (0.426, 0.446, 0.425, 0.491),
      9: (1.347, 1.588, 1.410, 1.379), 10: (0.193, 0.097, 0.046, 0.131),
      11: (1.337, 1.178, 1.182, 1.254), 12: (1.772, 1.734, 1.630, 1.720)}
# Table V metrics (WNS, TNS, power, congestion) for 1st+, 2nd+, 3rd+, [26]+ GAP-LA [reported]
T5M = {
    1: [(-0.366, -995.9, 0.646078, 6655094), (-0.366, -986.8, 0.646074, 6615019), (-0.367, -987.7, 0.646078, 6684393), (-0.364, -992.5, 0.646082, 6668336)],
    2: [(-0.391, -9594.3, 3.05347, 22926581), (-0.387, -9541.2, 3.05335, 22565487), (-0.384, -9478.6, 3.05335, 22404691), (-0.389, -9635.2, 3.05345, 24564053)],
    3: [(-57.589, -520231.3, 2.93845, 13512436), (-56.475, -518380.2, 2.93804, 14294566), (-57.642, -519322.0, 2.93750, 13539305), (-56.905, -517720.5, 2.93792, 13774919)],
    4: [(-0.383, -1678.2, 0.143804, 2691325), (-0.384, -1673.6, 0.143590, 2667440), (-0.387, -1669.5, 0.143594, 2691663), (-0.390, -1735.9, 0.143795, 2695234)],
    5: [(-0.276, -17315.2, 7.58974, 67951941), (-0.258, -17153.2, 7.57485, 68592745), (-0.253, -16941.9, 7.55822, 68488719), (-0.314, -17674.9, 7.57887, 68206894)],
    6: [(-0.249, -52866.6, 23.1439, 246164776), (-0.246, -52649.0, 23.1022, 245829220), (-0.251, -52342.4, 23.0783, 239426580), (-0.248, -52505.1, 23.0946, 252417493)],
    7: [(-0.643, -111.0, 0.156168, 4382560), (-0.610, -101.3, 0.156157, 4410642), (-0.614, -102.0, 0.156148, 4383521), (-0.668, -117.4, 0.156158, 4417159)],
    8: [(0.0, 0.0, 0.304879, 21601536), (0.0, 0.0, 0.304863, 22622941), (0.0, 0.0, 0.304885, 21556732), (0.0, 0.0, 0.304852, 24942228)],
    9: [(0.0, 0.0, 0.136300, 13167451), (0.0, 0.0, 0.136306, 15576558), (0.0, 0.0, 0.136301, 13799305), (0.0, 0.0, 0.136310, 13475394)],
    10: [(-0.348, -1315.0, 0.143850, 2074929), (-0.329, -1303.2, 0.143695, 2077647), (-0.333, -1289.2, 0.143565, 2059059), (-0.329, -1299.0, 0.143783, 2113216)],
    11: [(-0.403, -27048.8, 8.19537, 68079290), (-0.352, -25870.8, 8.14163, 69214810), (-0.351, -25364.8, 8.15759, 68307458), (-0.366, -26550.1, 8.17356, 68304067)],
    12: [(-0.193, -37883.3, 24.0157, 235753772), (-0.209, -37334.2, 23.9685, 241013713), (-0.194, -36737.6, 23.9422, 236685305), (-0.212, -37835.3, 23.9641, 240225253)],
}
# Xiao et al., ISPD'26 "Gradient-Guided RC Weighting" (best paper) [reported; re-extracted from the
# PDF by the user 2026-10-01]. Table 2: per-case total score of the ISPD'25 top-3 (as re-evaluated
# by Xiao) and Xiao's own result; their case order 0..11 == our cases 1..12 ("_r" = blind).
# Table 3: (TNS, WNS, power, congestion) of Xiao's own result (power printed to 3 decimals only).
XIAO_NAMES = ("1st(Xiao eval)", "2nd(Xiao eval)", "3rd(Xiao eval)", "Xiao ISPD26")
XIAO = {1: (0.897, 0.640, 0.095, -0.945), 2: (0.562, 0.722, 0.094, -0.409),
        3: (-1.995, -2.077, -1.824, -2.144), 4: (-0.405, 0.401, 1.370, 0.443),
        5: (-2.029, -1.295, -3.599, -2.180), 6: (0.510, 0.399, 0.714, 0.497),
        7: (1.780, 1.646, 1.519, 1.461), 8: (0.860, 0.426, 0.700, 0.424),
        9: (1.402, 4.375, 1.364, 1.391), 10: (0.358, 2.088, 1.817, 1.205),
        11: (1.333, 1.233, 3.117, 1.346), 12: (2.327, 2.371, 2.678, 2.313)}
XIAO_M = {1: (-1026.76, -0.379, 0.646, 6438399), 2: (-8666.18, -0.401, 3.053, 22573155),
          3: (-515533.69, -55.926, 2.941, 13056103), 4: (-2085.02, -0.461, 0.145, 2910471),
          5: (-19042.52, -0.283, 7.673, 65052484), 6: (-54359.36, -0.264, 23.320, 252306571),
          7: (-45.87, -0.431, 0.156, 4374734), 8: (0.0, 0.0, 0.305, 21213727),
          9: (0.0, 0.0, 0.136, 13936159), 10: (-1600.96, -0.392, 0.144, 2246898),
          11: (-29386.86, -0.421, 8.349, 55695359), 12: (-37846.03, -0.216, 24.152, 261589161)}
# Xiao ISPD'26 Table 4 [reported, re-extracted by the user 2026-10-01]: the POWER-optimized variant
# (uses per-cell switching activity; the paper keeps it out of its Table 2/3 comparison). Per case:
# score, and (switching power Ps W, total power Pt W, congestion) of the power variant.
XIAO_P = {1: -0.977, 2: -0.409, 3: -2.285, 4: -0.349, 5: -7.962, 6: 0.426,
          7: 1.462, 8: 0.424, 9: 1.394, 10: 1.172, 11: 0.235, 12: -0.332}
XIAO_PM = {1: (0.0009, 0.6460, 6443602), 2: (0.0773, 3.0531, 22576420), 3: (0.1561, 2.9345, 13175395),
           4: (0.0454, 0.1417, 3069599), 5: (3.0019, 7.3692, 72775512), 6: (9.7320, 22.8011, 269961040),
           7: (0.0001, 0.1561, 4376830), 8: (0.0024, 0.3048, 21180832), 9: (0.0013, 0.1363, 13972036),
           10: (0.0447, 0.1410, 2416448), 11: (3.2413, 7.8588, 64429910), 12: (9.4767, 22.6697, 292718663)}
T3_NAMES = ("1st", "2nd", "3rd", "GAP-LA")
T5_NAMES = ("1st+GAP-LA", "2nd+GAP-LA", "3rd+GAP-LA", "[26]+GAP-LA")


def case_of(design, suite="visible"):
    key = design if suite == "visible" else f"{design}@{suite}"
    for k, v in CASES.items():
        if v[0] == key:
            return k
    raise KeyError(key)


def cap_weights(design, suite="visible"):
    design = design.split("@")[0]
    p = os.path.join(CAPDIR, suite, design, design + ".cap")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        L = int(fh.readline().split()[0])
        t = fh.readline().split()
    return tuple(float(v) for v in t[2 + L: 6 + L])


def score(case, wns, tns, power, tof, weights=None):
    _, r1, r2, r3, nend, w = CASES[case]
    w1, w2, w3, w4 = weights or w
    parts = (w1 * (wns - r1), w2 * (tns - r2) / nend, w3 * (power - r3), w4 * tof)
    return sum(parts), parts


def t5_consistent(case, k, tol=0.02):
    """does the reported Table V score match the score recomputed from its own metrics?"""
    s, _ = score(case, *T5M[case][k])
    return abs(s - T5[case][k]) <= tol, s


def bar(case):
    """(best reported score, who). A Table V score that does not match its own reported
    metrics is replaced by the recomputed value (flagged 'recomp')."""
    cands = (list(zip(T3_NAMES, T3[case])) + list(zip(XIAO_NAMES, XIAO[case]))
             + [("Xiao ISPD26 power", XIAO_P[case])])
    for k, (name, rep) in enumerate(zip(T5_NAMES, T5[case])):
        ok, s = t5_consistent(case, k)
        cands.append((name if ok else name + "(recomp)", rep if ok else s))
    return min(cands, key=lambda kv: kv[1])[::-1]


def parse_ppa(path):
    t = open(path).read()
    g = lambda k: float(re.search(rf"^{k}\s+(-?[\d.eE+-]+)", t, re.M).group(1))
    pw = re.search(r"^Total\s+\S+\s+\S+\s+\S+\s+(\S+)", t, re.M)
    return g("wns"), g("tns"), float(pw.group(1)), t.count("Missing route to pin")


def parse_tof(path):
    t = open(path).read()
    return float(re.search(r"overflow cost\s+([\d.eE+-]+)", t).group(1))


def self_test():
    ok, worst = True, 0.0
    for c, meths in T4.items():
        for name, rep in zip(T3_NAMES, T3[c]):
            s, _ = score(c, *meths[name])
            err = abs(s - rep)
            worst = max(worst, err)
            if err > 0.02:
                ok = False
                print(f"  case{c} {name}: recomputed {s:.3f} vs reported {rep:.3f}  MISMATCH")
    print(f"  recomputed all 48 Table III scores from Table IV: max |err| = {worst:.3f}")
    bad = []
    for c in T5M:
        for k in range(4):
            good, s = t5_consistent(c, k)
            if not good:
                bad.append(f"case{c} {T5_NAMES[k]}: reported {T5[c][k]:+.3f} vs recomputed {s:+.3f}")
    print(f"  Table V: {48 - len(bad)}/48 scores match their own metrics; inconsistent: " + ("; ".join(bad) if bad else "none"))
    for c in CASES:
        suite = "blind" if "@" in CASES[c][0] else "visible"
        cw = cap_weights(CASES[c][0], suite)
        if cw:
            same = all(abs(a - b) <= 1e-12 + 1e-9 * abs(b) for a, b in zip(cw, CASES[c][5]))
            ok &= same
            print(f"  case{c} {CASES[c][0]:22s} .cap weights {cw} == table: {same}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design")
    ap.add_argument("--suite", default="visible", choices=["visible", "blind"])
    ap.add_argument("--ppa")
    ap.add_argument("--eval", dest="evalf")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.table:
        for c in CASES:
            s, who = bar(c)
            print(f"case{c:<3d} {CASES[c][0]:22s} best reported {s:+.3f}  ({who})")
        return 0
    c = case_of(a.design, a.suite)
    wns, tns, pw, miss = parse_ppa(a.ppa)
    tof = parse_tof(a.evalf)
    s, parts = score(c, wns, tns, pw, tof)
    b, who = bar(c)
    print(f"{a.design}[{a.suite}] case{c}: WNS {wns:.4f} TNS {tns:.1f} power {pw:.6f} tof {tof:,.0f} missing_pins {miss}")
    print(f"  score {s:+.3f}  = WNS {parts[0]:+.3f} TNS {parts[1]:+.3f} power {parts[2]:+.3f} "
          f"overflow {parts[3]:+.3f}   | best reported {b:+.3f} ({who}) -> "
          f"{'BEATS' if s < b else 'behind'} by {abs(s - b):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
