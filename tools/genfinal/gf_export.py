#!/usr/bin/env python3
"""gf_export.py (genfinal 2026-11-12, NEW): gf_train.py model.pt -> <out>.bin in the layout of tools/best8/gnn_kern.cu (as
tools/groupgnn/export11.py writes it) + one hook config per beta, <out>_b<beta>.json, for tools/groupgnn/gg_hook3.py (the R3 zero-step
path): bin, beta, tau 0.01, a constant gate (sigmoid(30) = 1, so the start is 4 x onehot + beta x o), H/layers/gin/cin, md5.
Check: the kernel (libgnn_cpu.so through tools/best8/gnn_f.py, set_inputs) and torch give the same z on a dump (max |dz| printed,
must be < 1e-3), so the hook runs exactly the trained network.
  gf_export.py <model.pt> <out prefix> <check dump.npz> [betas, default 0.6,1,2,4]"""
import hashlib, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools", "best8"))
import gf_train as GT
RELS = ("grid__connects__grid", "candidate__passes_through__grid", "grid__influences__candidate", "candidate__competes__candidate")


def main():
    ckp, out, chk = sys.argv[1], sys.argv[2], sys.argv[3]
    betas = [float(b) for b in (sys.argv[4] if len(sys.argv) > 4 else "0.6,1,2,4").split(",")]
    ck = torch.load(ckp, map_location="cpu", weights_only=False); st = ck["state"]
    f = lambda n: st[n].detach().cpu().numpy().astype(np.float32).reshape(-1)
    H = st["input_proj.grid.weight"].shape[0]; gin = st["input_proj.grid.weight"].shape[1]; cin = st["input_proj.candidate.weight"].shape[1]
    layers = 1 + max(int(n.split(".")[1]) for n in st if n.startswith("convs."))
    parts = [f("input_proj.grid.weight"), f("input_proj.grid.bias"), f("input_proj.candidate.weight"), f("input_proj.candidate.bias")]
    for l in range(layers):
        for r in RELS:
            parts += [f(f"convs.{l}.{r}.lin_l.weight"), f(f"convs.{l}.{r}.lin_l.bias"), f(f"convs.{l}.{r}.lin_r.weight")]
    parts += [f("out.weight"), f("out.bias")]
    b = np.array([H, layers, gin, cin], np.int32).tobytes() + np.concatenate(parts).tobytes()
    open(out + ".bin", "wb").write(b)
    md5 = hashlib.md5(b).hexdigest()
    # check: kernel vs torch on a dump
    import gnn_f as GN
    Z = np.load(chk)
    net = GN.Net(out + ".bin", gpu=False, T=8)
    net.R, net.K = Z["sid"].shape
    net.set_inputs(Z["xg"], Z["xc"], [(Z["off0"], Z["src0"]), (Z["off1"], Z["src1"]), (Z["off2"], Z["src2"]), (Z["off3"], Z["src3"])])
    zk, _ = net.forward(want=True)
    m = GT.Net(H, layers, gin, cin); m.load_state_dict(st); m.eval()
    ng, nc = len(Z["xg"]), len(Z["xc"])
    G = {"xg": torch.as_tensor(Z["xg"]), "xc": torch.as_tensor(Z["xc"])}
    G["A0"], G["A0T"] = GT.csr(Z["off0"], Z["src0"], ng, "cpu"); G["A1"], G["A1T"] = GT.csr(Z["off1"], Z["src1"], nc, "cpu")
    G["A2"], G["A2T"] = GT.csr(Z["off2"], Z["src2"], ng, "cpu"); G["A3"], G["A3T"] = GT.csr(Z["off3"], Z["src3"], nc, "cpu")
    with torch.no_grad():
        zt = m(G).numpy()
    dz = float(np.abs(zt - zk).max()); rel = dz / max(float(np.abs(zt).max()), 1e-9)
    gate = {"l1w": [[0.0] * 5 for _ in range(16)], "l1b": [0.0] * 16, "l2w": [[0.0] * 16], "l2b": [30.0]}
    for beta in betas:
        J = {"bin": os.path.abspath(out + ".bin"), "beta": beta, "tau": 0.01, "gate": gate, "ckpt": os.path.abspath(ckp), "step": ck.get("step"),
             "version": "genfinal1", "beta_sel": 1.0, "H": H, "layers": layers, "gin": gin, "cin": cin, "md5": md5,
             "train": ck.get("train"), "test": ck.get("test"), "check_max_abs_dz": dz}
        json.dump(J, open(out + "_b%g.json" % beta, "w"), indent=1)
    rec = {"bin": out + ".bin", "md5": md5, "check_dump": chk, "nc": nc, "max_abs_dz": dz, "rel": rel, "ok": bool(dz < 1e-3 or rel < 1e-4),
           "z_mean": float(zt.mean()), "z_std": float(zt.std()), "betas": betas}
    print(json.dumps(rec), flush=True)
    json.dump(rec, open(out + ".check.json", "w"), indent=1)
    sys.exit(0 if rec["ok"] else 4)


if __name__ == "__main__":
    main()
