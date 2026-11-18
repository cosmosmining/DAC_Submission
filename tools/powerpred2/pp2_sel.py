#!/usr/bin/env python3
"""pp2_sel.py (powerpred2, NEW 2026-11-13; round 2 the same day): run-time part of the power-round predictor. Reached from the patched chain
source as the global PPL2 (pp2_patch.py), and installs its pre / select / wrap into tools/powerpred/pp_lib.py IN MEMORY (the file is not edited).

Layer data: the flow's own S.la.C (fF/um), S.la.dirs (0 = H, 1 = V), S.la.um (um per GCell), from TimingLA (timing_promote.py:158-160) and
route_export.LayerAssigner; fallback (only if S.la cannot be read) = pp2_lib.NG45_C. Every log line names the source used.
Wire length per net and layer: the net's current segments S.new.get(i, S.segs[i]) (same walk as pp_lib.feats, pp_lib.py:43-48), exact float64.

Env (checked by preflight() before the flow starts; a bad value stops the driver at once):
  PP_MODE   log | off (the flow's own round; features and labels only) and the modes added here:
              potential        keep floor(PP_FRAC x n2) of the power part (n2 nets) by activity x reducible cap; PP_FRAC required
              potential_share  keep the smallest prefix of the power part by potential holding PP_TARGET (default 0.95) of its potential [TIME a]
              mlp_share        same, predicted gain = sigmoid(MLP) x potential (PP2_MLP_GAIN=p_pot, default) or sigmoid(MLP) (=p)         [TIME b]
              adaptive         old name: mlp_share with PP2_MLP_GAIN=p and PP_TARGET default 0.98 (p is NOT calibrated: logged)
            the old pp_lib modes base | learned are refused (they prune the timing part o1 and count with ceil)
  PP_TARGET  share of the predicted gain to keep (0 < t <= 1; default 0.95; adaptive 0.98)
  PP2_KEEP_O1  1 (DEFAULT): the timing-weight part o1 (the first n_o1 nets of the list) is always kept; only the power part is pruned.
               0: o1 is ranked with the rest (ablation only; RULES.md amendment)
  PP_MODEL   .npz of tools/powerpred/pp_train.py (mu, sd, W1, b1, W2, b2 over the 43 pp_lib features); model_wo_<D>.npz must name the running design
  PP2_LIST   plain (default) | score: SCORE route, the power part o2 = the same number of nets, taken from all priced nets not in o1 (the flow's
             universe cp) ranked by activity x reducible cap (ties: activity x total cap)
  PP_LAYER_DIR  optional check only (directions always come from S.la.dirs): comma list of 0/1, must equal the ISPD 2025 table 0,1,0,1,... (10)
  PP_OUT    log lines also go to <PP_OUT>.pp2.jsonl; records: kind=sel (selection), kind=list (SCORE list), kind=round (one per wrapped power
            round, skipped or not), kind=unwrapped (a power round that reached PPL.pre outside PPL.wrap: server SCHED paths; it has no round
            record, so pp2_summary leaves TIME not decided). Label files <PP_OUT>.c<k>.npz and <PP_OUT>.jsonl are pp_lib's, written only for
            rounds that were not skipped.
Timing of one round (kind=round), all time.perf_counter wall seconds of this process:
  round_s  the chain's own duration of power_round_i (its 4th return value: from its first line to its return);
  log_s    PPL.pre (features + label arrays: logging only);
  sel_s    PPL.select (the variant's own selection; for plain / score the fall-through, about 0);
  list_s   the SCORE list (score_list);
  batch_s  sum over the top-level batches of the batch loop (PPL2.timed_batch around `try_batch(cb, allr[..], 0)`: prom_pw + gate + its
           bisection); each batch is split by net count into o1_s (timing-weight nets) and pw_s (power-part nets);
  own_s    = round_s - log_s (MLP modes: round_s, because their selection reads the features);
  own_pw_s = pw_s + sel_s (+ log_s in the MLP modes): the seconds the TIME rule compares (RULES.md amendment A6, condition 2): the power
           part's own re-assignment work plus the variant's own selection, without label logging the variant does not need. None when the
           round has no batch loop (pw_batch <= 0).
  n_batches / n_batches_o2 / n_mixed  top-level batches (all / holding at least one power-part net / holding both parts);
  n_gates / n_gates_o2  gate re-times (prom_pw + gate) of all top-level batches, bisection included / of those holding power-part nets
           (the growth of info["batches"]; None if the chain has no such list). A batch costs about one gate re-time whatever its size, so pw_s
           steps with the batch and gate counts, not with n_kept_o2; a mixed batch is split by net count (reported only, skeptic round 2).
PPL.post (label files) runs after the round returns and is in none of them.
Not read for any verdict: the chain json's pw_info n_crit / n_pw / n_pw_cand (chain_i.py:1449) are set BEFORE the inserted selection and so
show the unpruned list in pruned variants; the kept counts are only in the kind=sel / kind=round records here (skeptic round 2).
Skipped rounds: reason_code pw_skip (the bound was below --pw_skip: N/A) or no_activity (S.kpn is None, chain_i.py:1433-1436: an input
fault, which pp2_summary makes 'not decided', never N/A).
Start checks (preflight): every PP_* / PP2_* name must be in KNOWN_ENV (a misspelt name would silently fall back to a default), and a name
that the chosen PP_MODE ignores is refused (PP_FRAC outside potential, PP_TARGET outside the share modes, PP_MODEL / PP2_MLP_GAIN outside the
MLP modes). In the MLP modes the model file must be named model_wo_<design>.npz for the running --design."""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import pp2_lib as L

LIST = os.environ.get("PP2_LIST", "plain")
MODES = ("potential", "potential_share", "mlp_share", "adaptive")
MLP_MODES = ("mlp_share", "adaptive")
OK_MODES = ("log", "off") + MODES
MODEL_KEYS = ("mu", "sd", "W1", "b1", "W2", "b2")
SHARE_MODES = ("potential_share", "mlp_share", "adaptive")
KNOWN_ENV = ("PP_MODE", "PP_TARGET", "PP_FRAC", "PP_MODEL", "PP_OUT", "PP_STOP", "PP_LAYER_DIR", "PP2_KEEP_O1", "PP2_LIST", "PP2_MLP_GAIN",
             "PP2_DRYRUN", "PP2_SKIP_ON")
G = {"pp_lib": None, "model": None, "design": None, "round": 0, "bound": None, "list_s": 0.0, "batches": [], "in_wrap": False}


def _design(argv=None):
    a = sys.argv if argv is None else argv
    return a[a.index("--design") + 1] if "--design" in a and a.index("--design") + 1 < len(a) else ""


def keep_o1(env=None):
    return (os.environ if env is None else env).get("PP2_KEEP_O1", "1") != "0"


def check_model(path, design):
    """-> (model dict or None, error text or None)"""
    if not path:
        return None, "PP_MODEL is not set (needed by PP_MODE mlp_share / adaptive)"
    if not os.path.isfile(path):
        return None, "PP_MODEL %s does not exist" % path
    b = os.path.basename(path)
    if b != "model_wo_%s.npz" % design:       # skeptic round 2: any other name (model_all.npz, ...) could be in-sample
        return None, ("PP_MODEL %s: the model must be named model_wo_%s.npz (trained without the running design); any other name may be "
                      "in-sample" % (b, design))
    try:
        z = np.load(path); M = {q: z[q] for q in z.files}
    except Exception as e:
        return None, "PP_MODEL %s cannot be read: %r" % (path, e)
    miss = [k for k in MODEL_KEYS if k not in M]
    if miss:
        return None, "PP_MODEL %s lacks %s" % (path, miss)
    nf = len(L.NAMES)
    if M["mu"].shape[0] != nf or M["sd"].shape[0] != nf or M["W1"].shape[0] != nf:
        return None, "PP_MODEL %s has %d / %d / %d inputs, the features have %d (pp_lib.NAMES)" % (path, M["mu"].shape[0], M["sd"].shape[0],
                                                                                                   M["W1"].shape[0], nf)
    return M, None


def preflight(env=None, argv=None):
    """All env / model / layer checks, before the flow starts. -> list of error texts (empty = OK)."""
    env = os.environ if env is None else env; argv = sys.argv if argv is None else argv
    E = []
    mode = env.get("PP_MODE", "log")
    unk = sorted(k for k in env if (k.startswith("PP_") or k.startswith("PP2_")) and k not in KNOWN_ENV)
    if unk:
        E.append("unknown pp2 environment name(s) %s (known: %s); a misspelt name would silently use the default" % (unk, ", ".join(KNOWN_ENV)))
    if "PP_FRAC" in env and mode != "potential":
        E.append("PP_FRAC is used only by PP_MODE=potential (PP_MODE=%s would ignore it)" % mode)
    if "PP_TARGET" in env and mode not in SHARE_MODES:
        E.append("PP_TARGET is used only by PP_MODE %s (PP_MODE=%s would ignore it)" % ("/".join(SHARE_MODES), mode))
    for k in ("PP_MODEL", "PP2_MLP_GAIN"):
        if k in env and mode not in MLP_MODES:
            E.append("%s is used only by PP_MODE %s (PP_MODE=%s would ignore it)" % (k, "/".join(MLP_MODES), mode))
    if mode not in OK_MODES:
        E.append("PP_MODE=%s not allowed by the pp2 drivers (allowed: %s; base/learned prune o1 and count with ceil)" % (mode, ", ".join(OK_MODES)))
    if env.get("PP2_LIST", "plain") not in ("plain", "score"):
        E.append("PP2_LIST must be plain or score")
    if env.get("PP2_SKIP_ON", "plain") != "plain":
        E.append("PP2_SKIP_ON=%s: removed; the --pw_skip test always runs on the flow's own list (the selection now comes after it)" % env["PP2_SKIP_ON"])
    if env.get("PP2_KEEP_O1", "1") not in ("0", "1"):
        E.append("PP2_KEEP_O1 must be 0 or 1 (default 1)")
    if env.get("PP2_MLP_GAIN", "p_pot") not in ("p", "p_pot"):
        E.append("PP2_MLP_GAIN must be p or p_pot")
    if "PP_TARGET" in env:
        try:
            t = float(env["PP_TARGET"]); assert 0.0 < t <= 1.0
        except Exception:
            E.append("PP_TARGET=%r must be a number in (0, 1]" % env["PP_TARGET"])
    if mode == "potential":
        try:
            f = float(env["PP_FRAC"]); assert 0.0 < f <= 1.0
        except KeyError:
            E.append("PP_MODE=potential needs PP_FRAC (share of the power part kept; no silent default)")
        except Exception:
            E.append("PP_FRAC=%r must be a number in (0, 1]" % env.get("PP_FRAC"))
    if mode in MLP_MODES:
        d = _design(argv)
        if not d:
            E.append("PP_MODE=%s needs --design on the command line (to refuse an in-sample model)" % mode)
        else:
            _, e = check_model(env.get("PP_MODEL", ""), d)
            if e:
                E.append(e)
    if env.get("PP_LAYER_DIR"):
        try:
            ld = [int(x) for x in env["PP_LAYER_DIR"].split(",")]
        except ValueError:
            ld = None
        if ld is None or ld != L.NG45_DIR.tolist():
            E.append("PP_LAYER_DIR=%s is wrong: the ISPD 2025 designs have 10 layers, directions %s (metal1 = 0 = H); the run uses S.la.dirs anyway"
                     % (env["PP_LAYER_DIR"], ",".join(str(x) for x in L.NG45_DIR)))
    out = env.get("PP_OUT", "")
    if out and not os.path.isdir(os.path.dirname(os.path.abspath(out))):
        E.append("PP_OUT folder %s does not exist" % os.path.dirname(os.path.abspath(out)))
    return E


def log(rec):
    s = json.dumps(rec, default=float)
    print("PP2 " + s, flush=True)
    out = os.environ.get("PP_OUT", "")
    if out:
        with open(out + ".pp2.jsonl", "a") as fh:
            fh.write(s + "\n")


def layer_table(S):
    """(C, dirs, um, source) of the running design."""
    try:
        la = S.la; nl = int(la.L)
        C = np.asarray(la.C, np.float64).ravel()[:nl]; d = np.asarray(la.dirs, np.int64).ravel()[:nl]
        um = float(getattr(la, "um", 1.0))
        assert len(C) == nl == len(d)
        return C, d, um, "exact S.la.C/S.la.dirs/S.la.um (L=%d)" % nl
    except Exception as e:
        return L.NG45_C.copy(), L.NG45_DIR.copy(), 1.0, "fallback %s; S.la unreadable: %r" % (L.NG45_SRC, e)


def lengths(S, nets, nl):
    """(n, nl) wire length in GCells per net and layer (vias ignored), from the net's current segments."""
    W = np.zeros((len(nets), nl)); names = S.SD.net_names
    for j, k in enumerate(np.asarray(nets).tolist()):
        i = S.ridx.get(names[k])
        if i is None:
            continue
        for x1, y1, z1, x2, y2, z2 in S.new.get(i, S.segs[i]):
            if z1 == z2:
                W[j, int(z1)] += abs(x2 - x1) + abs(y2 - y1)
    return W


def red_cap(S, nets):
    """reducible capacitance in fF per net (exact layer table of the run). -> (red, source)"""
    C, d, um, src = layer_table(S)
    return L.reducible_cap(lengths(S, nets, len(C)) * um, C, d), src


def _model():
    if G["model"] is None:
        M, e = check_model(os.environ.get("PP_MODEL", ""), _design())
        assert e is None, e
        G["model"] = M
    return G["model"]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_select(orig):
    def select(allr):
        t = time.perf_counter()
        pl = G["pp_lib"]; P = pl.ST["pre"]; mode = os.environ.get("PP_MODE", "log")
        if P is not None:
            # review F22: the stored features must belong to this list
            assert np.array_equal(np.asarray(P["ids"]), np.asarray(allr)) and P["X"].shape[0] == len(allr), \
                "pp2 select: stored features (%d rows) do not belong to this list (%d nets)" % (P["X"].shape[0], len(allr))
        if P is None or mode not in MODES:
            res = orig(allr)
            if P is not None:
                P["sel_s"] = time.perf_counter() - t; P["sel_uses_X"] = False
            return res
        allr = np.asarray(allr); n = len(allr); no1 = int(P["n_o1"])
        forced = (np.arange(n) < no1) if keep_o1() else None
        # skeptic round 2: with o1 kept, only the power part is ranked, so only its segments are walked (sel_s is inside own_pw_s)
        r0 = no1 if forced is not None else 0
        red = np.zeros(n); rr, src = red_cap(P["S"], allr[r0:]); red[r0:] = rr
        pot = np.asarray(P["kp"], np.float64) * red
        tgt = float(os.environ.get("PP_TARGET", "0.98" if mode == "adaptive" else "0.95"))
        rec = {"kind": "sel", "call": int(pl.ST["calls"]), "mode": mode, "n": int(n), "n_o1": no1, "keep_o1": forced is not None, "c_source": src,
               "ranked_from": r0, "pot_total": float(pot[r0:].sum()), "red0_share": float((red[r0:] <= 0).mean()) if n > r0 else 0.0}
        if mode == "potential":
            frac = float(os.environ["PP_FRAC"]); keep = np.zeros(n, bool)
            if forced is not None:
                keep |= forced
            free = np.nonzero(~keep)[0]
            k = L.k_at_most(frac, len(free))
            if k == 0 and forced is None and n:
                k = 1
            keep[free[np.argsort(-pot[free], kind="stable")[:k]]] = True
            idx = np.nonzero(keep)[0]
            pf = pot[free].sum()
            sh = float(pot[np.intersect1d(idx, free)].sum() / pf) if pf > 0 else float("nan"); rec["frac_param"] = frac
        else:
            if mode == "potential_share":
                pred = pot
            else:
                gm = os.environ.get("PP2_MLP_GAIN", "p" if mode == "adaptive" else "p_pot")
                p = _sig(pl._mlp(P["X"], _model())); pred = p if gm == "p" else p * pot
                rec["mlp_gain"] = gm; rec["model"] = os.path.basename(os.environ["PP_MODEL"]); rec["p_calibrated"] = False
            idx, _, sh = L.choose_prefix(pred, tgt, forced)
            rec["target"] = tgt
        keep = np.zeros(n, bool); keep[idx] = True
        P["sel"] = keep
        P["sel_s"] = time.perf_counter() - t; P["sel_uses_X"] = mode in MLP_MODES
        rec.update(n_kept=int(keep.sum()), n_o2=int(n - no1), n_kept_o2=int(keep[no1:].sum()), frac_kept=float(keep.mean()) if n else 0.0,
                   frac_kept_o2=float(keep[no1:].mean()) if n > no1 else None, pred_share_kept=sh, sel_s=P["sel_s"])
        log(rec)
        return allr[keep]
    return select


def make_pre(orig):
    def pre(*args):
        if not G["in_wrap"]:
            log({"kind": "unwrapped", "call": int(G["pp_lib"].ST["calls"]) + 1,
                 "note": "power round reached PPL.pre outside PPL.wrap (no round record); TIME not decided"})
        t = time.perf_counter()
        orig(*args)
        P = G["pp_lib"].ST["pre"]
        P["log_s"] = time.perf_counter() - t
    return pre


def make_wrap(pl):
    """replaces pp_lib.wrap: one kind=round record per power round (skipped or not), then pp_lib.post (labels) for rounds not skipped."""
    def wrap(f, cur):
        pl.ST["pre"] = None; G["bound"] = None; G["list_s"] = 0.0; G["batches"] = []
        G["in_wrap"] = True
        try:
            res = f(cur)
        finally:
            G["in_wrap"] = False
        G["round"] += 1
        P = pl.ST["pre"]; rs = float(res[3])
        rec = {"kind": "round", "round": G["round"], "round_s": rs, "mode": os.environ.get("PP_MODE", "log"), "list": LIST}
        if P is None:      # the round returned before the hook point: --pw_skip, or no activity file
            b = G["bound"]
            psk = b is not None and b[1] is not None and b[0] < b[1]
            # skeptic round 2: the only return before the --pw_skip bound is `if S.kpn is None` (no activity file, chain_i.py:1433-1436,
            # chain_y.py:2434): an input fault, recorded as no_activity (pp2_summary: not decided), never as a pw_skip (N/A)
            rec.update(skipped=True, reason_code="pw_skip" if psk else "no_activity",
                       reason=("pw_skip: bound %.6g < %.6g" % (b[0], b[1])) if psk
                       else "no_activity: the round returned before the --pw_skip bound (S.kpn is None: no activity file); input fault, not decided",
                       ub=None if b is None else b[0], pw_skip=None if b is None else b[1], log_s=0.0, sel_s=0.0, list_s=0.0, own_s=rs,
                       batch_s=0.0, o1_s=0.0, pw_s=0.0, own_pw_s=0.0, n_batches=0, n_batches_o2=0, n_mixed=0, n_gates=0, n_gates_o2=0)
            log(rec)
            return res
        ls = float(P.get("log_s", 0.0)); ss = float(P.get("sel_s", 0.0)); ux = bool(P.get("sel_uses_X", False))
        n = int(len(P["ids"])); nk = int(P["sel"].sum()) if P["sel"] is not None else n; no1 = int(P["n_o1"])
        nk2 = int(P["sel"][no1:].sum()) if P["sel"] is not None else n - no1
        bt = batch_split(G["batches"])
        pw = bt["pw_s"] if (G["batches"] or nk == 0) else None
        rec.update(skipped=False, call=int(pl.ST["calls"]), n=n, n_o1=no1, n_kept=nk, n_o2=n - no1, n_kept_o2=nk2,
                   frac_kept_o2=(nk2 / (n - no1)) if n > no1 else None, log_s=ls, sel_s=ss, list_s=G["list_s"],
                   own_s=rs if ux else rs - ls, features_counted=ux, batch_s=bt["batch_s"], o1_s=bt["o1_s"], pw_s=pw,
                   own_pw_s=None if pw is None else pw + ss + (ls if ux else 0.0), n_batches=len(G["batches"]),
                   n_batches_o2=bt["n_batches_o2"], n_mixed=bt["n_mixed"], n_gates=bt["n_gates"], n_gates_o2=bt["n_gates_o2"],
                   ub=None if G["bound"] is None else G["bound"][0], pw_skip=None if G["bound"] is None else G["bound"][1])
        log(rec)
        pl.post(res)       # may end the process (PP_STOP=1); the round record is already written
        return res
    return wrap


def install(pp_lib):
    """replace pp_lib.pre / select / wrap (in memory); modes not in MODES fall through to the original select"""
    if G["pp_lib"] is None:
        G["pp_lib"] = pp_lib
        pp_lib.select = make_select(pp_lib.select)
        pp_lib.pre = make_pre(pp_lib.pre)
        pp_lib.wrap = make_wrap(pp_lib)
    return pp_lib


def batch_split(B):
    """B = [(n nets, n timing-weight nets, seconds[, gate re-times or None])] of the top-level batches -> batch_s, o1_s, pw_s (a mixed batch
    split by net count), n_batches_o2 (batches with a power-part net), n_mixed, n_gates / n_gates_o2 (None when a count is unknown)"""
    bs = sum(b[2] for b in B)
    o1 = sum(b[2] * b[1] / b[0] for b in B if b[0] > 0)
    g = [b[3] if len(b) > 3 else None for b in B]
    known = all(x is not None for x in g)
    return {"batch_s": bs, "o1_s": o1, "pw_s": bs - o1, "n_batches_o2": sum(1 for b in B if b[0] > b[1]),
            "n_mixed": sum(1 for b in B if 0 < b[1] < b[0]), "n_gates": sum(g) if known else None,
            "n_gates_o2": sum(x for b, x in zip(B, g) if b[0] > b[1]) if known else None}


def timed_batch(f, cb, nets, depth, o1, info=None):
    """called by the patched batch loop in place of try_batch(cb, allr[..], 0): wall seconds of one top-level batch (pp2_patch BT) and its
    gate re-times = growth of info["batches"] (one entry per try_batch call that ran prom_pw + gate, bisection included)"""
    nb0 = len(info["batches"]) if isinstance(info, dict) and isinstance(info.get("batches"), list) else None
    t = time.perf_counter()
    res = f(cb, nets, depth)
    dt = time.perf_counter() - t
    ng = (len(info["batches"]) - nb0) if nb0 is not None else None
    nets = np.asarray(nets)
    G["batches"].append((int(len(nets)), int(np.isin(nets, np.asarray(o1)).sum()), dt, ng))
    return res


def note_bound(ub, thr):
    """called by the patched source right after the --pw_skip bound is computed (before the test)"""
    G["bound"] = (float(ub), None if thr is None else float(thr))


def split_parts(o1, o2_, allr):
    """after the selection: o1 / o2_ = their nets that are still in allr, order kept; asserts allr == [o1, o2_] so that the boundary
    n1_ = len(o1) of the batch loop (chain_y / chain_tf / chain_d1) is right (review F24, task d)"""
    o1 = np.asarray(o1); o2_ = np.asarray(o2_); allr = np.asarray(allr)
    k1 = o1[np.isin(o1, allr)]; k2 = o2_[np.isin(o2_, allr)]
    assert np.array_equal(np.concatenate([k1, k2]), allr), "pp2: the selected list is not [kept o1, kept o2] in order"
    return k1, k2


def score_list(S, r_, kp, cc, o1, cp, rank, o2_, a):
    """SCORE route. Called after the --pw_skip block (pp2_patch.py). PP2_LIST=plain: returns (o2_, rank) unchanged. PP2_LIST=score: cp
    (priced nets not in o1, the flow's universe) re-ranked by activity x reducible cap, ties by cc = activity x total cap (the flow's key);
    o2_ = the first len(o2_) of it (the flow's count, min(pw_nets, len(cp))); rank = the whole new order (used by --pw2_nets)."""
    if LIST != "score":
        return o2_, rank
    t = time.perf_counter(); cp = np.asarray(cp); m = len(o2_)
    red, src = red_cap(S, cp)
    pot = np.asarray(kp, np.float64)[cp] * red
    o = np.lexsort((-np.asarray(cc, np.float64)[cp], -pot))
    rank_n = cp[o]; o2n = rank_n[:m]
    pos = {int(v): j for j, v in enumerate(cp.tolist())}
    pot_old = float(sum(pot[pos[int(v)]] for v in np.asarray(o2_).tolist()))
    pl = G["pp_lib"]
    G["list_s"] = time.perf_counter() - t
    log({"kind": "list", "call": int(pl.ST["calls"]) + 1 if pl is not None else -1, "list": "score", "n_cp": int(len(cp)), "n_o1": int(len(o1)),
         "n_o2": int(m), "same_set": bool(set(o2n.tolist()) == set(np.asarray(o2_).tolist())), "overlap": int(len(np.intersect1d(o2n, o2_))),
         "cc_sum_plain": float(np.asarray(cc)[np.asarray(o2_, int)].sum()), "cc_sum_score": float(np.asarray(cc)[o2n].sum()),
         "pot_sum_plain": pot_old, "pot_sum_score": float(pot[o[:m]].sum()), "red0_share_cp": float((red <= 0).mean()) if len(cp) else 0.0,
         "c_source": src, "s": G["list_s"]})
    return o2n, rank_n
