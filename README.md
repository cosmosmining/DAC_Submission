# DTLA: code snapshot for an anonymous submission

## 1. What this is

This is the code behind the submitted paper on DTLA, the paper that links to this repository.
It reassigns the metal layers of a routed design by gradient descent on a smooth surrogate of the ISPD 2025 contest score, inside a flow of net-by-net stages that are each guarded by an acceptance check.
Anonymous submission, provided for review.

The files were selected by tracing imports, library names, include lines and build scripts from the launcher of the final flow. The snapshot has not been run as a whole in this form; see Section 7.

## 2. Repository map

Section, equation and algorithm line numbers refer to the submitted paper. Line ranges refer to the files as they are in this snapshot.

| Path and lines | What it holds | Paper |
|---|---|---|
| `ispd25_score.py`, `score()` lines 146-150; `cap_weights()` lines 135-143 | contest score; `score()` takes its weights from the CASES table, `cap_weights()` reads them from the `.cap` file | Sec. II-C, Eq. (1) |
| `tools/offix/of_official.py`, `overflow_cost()` lines 27-37 | congestion cost with the evaluator's rule for blocked edges | Eq. (2) |
| `tools/best7/dgb.cc`, `dgb_nets()` from line 20; `dgb_ovf_pot()` lines 257-275 | free segments and candidates per free net; congestion potential of a net | Sec. III-A; Alg. 1 line 2 |
| `tools/best8/dgr_stage_r.py`, `sensitivities()` lines 192-233 | sensitivity a_n: timing part lines 217-222, congestion part line 226, switching power part line 230 | Sec. III-A; Alg. 1 line 2 |
| `tools/best8/dgr_stage_r.py`, `prepare()` lines 236-250 | choice of the N free nets; the 7 percent rule at lines 248-249 | Alg. 1 line 3; Sec. IV |
| `tools/best7/dg_kern.cu`, `K_soft` lines 136-153, `K_pt` lines 154-175, `K_cell` lines 176-186 | softmax over the candidates, expected R and C, switching power, expected congestion as a sum of `log1p` factors per edge | Eq. (3); Alg. 1 line 7 |
| `tools/weights/gt_kern_w.cu`, `grp_fwd()` lines 644-670 | GPU timer, included by `dg_kern.cu`; log-sum-exp arrival times at lines 660-667 | Sec. III-B |
| `tools/best7/dg_kern.cu`, `arc_ev()` lines 430-443 | an arc whose inputs moved by at most `dl` keeps its delay and slew | Sec. III-B, the 3 percent rule |
| `tools/best7/dg_kern.cu`, `forward_all()` lines 587-631 | smoothed TNS line 620, smoothed WNS line 622, objective `val[4]` line 631 | Eqs. (4)-(6); Alg. 1 line 8 |
| `tools/best7/dg_kern.cu`, `backward_all()` lines 635-661; `K_step` lines 371-387 | gradient through the timer; softmax adjoint and Adam update | Alg. 1 line 9 |
| `tools/best7/dg_kern.cu`, `dg_run()` lines 666-711 | descent loop: step size rule line 676, annealing line 686, Adam bias terms lines 695-696 | Alg. 1 lines 5-10 |
| `tools/best7/dg.py`, `Dg.start()` lines 420-427; `Dg.run()` lines 429-439 | plain start x = init times one-hot of the current layers; host side of the descent | Alg. 1 lines 4-10 |
| `tools/best8/dgr_stage_r.py`, `round_rows()` lines 324-349 | rounding of every free segment to its most probable layer | Alg. 1 line 11 |
| `tools/best8/dgr_stage_r.py`, `run_()` from line 363, lines 410-456 | acceptance check of the rounded change, then up to `--dgr_split` halvings by first-order gain, lines 432-456 | Alg. 1 lines 12-19 |
| `tools/final/chain_y.py`, `dgr_run()` lines 2777-2822 | rounds of one DTLA call, loop line 2801, stop on a gain below `--dgr_round_min` line 2816 | Alg. 1 lines 1, 17-20 |
| `tools/sched/cold_n1.py`, `rules_patch()` from line 447, lines 451-455, `_a1()` lines 540-547 | stop when a round gains less than a share of the first round | Alg. 1 line 20, eta |
| `tools/sched/cold_n1.py`, `extra_patch()` lines 431-444 | end rounds of DTLA before the route is written | Sec. III-G |
| `tools/final/chain_y.py`, `gate()` lines 2398-2410, `retime_inc()` lines 2289-2343; `tools/engine/csta_inc.py` | acceptance check and incremental score estimator | Sec. III-G |
| `tools/par7/p7_dp.py`, `assign_pair()` lines 77-105; `tools/par7/p7_dp_kern.cc`, `net_dp()` from line 61, level schedule lines 289-292 | first pass: tree dynamic program per net, most critical first | Sec. III-G |
| `tools/final/chain_y.py`, `crit_round()` lines 2348-2364, `power_round_i()` from line 2427 | critical-path rounds and power round | Sec. III-G |
| `tools/best8/gnn_kern.cu`, `gnn_forward()` lines 190-234 | inference of the learned-start network | Sec. III-D, Eq. (7) |
| `tools/genfinal/gf_train.py`, `Net` lines 50-79, `loss_of()` lines 140-146; `tools/syndata1/syn_train_gen.py` | network definition, cross-entropy loss, training on synthetic designs | Sec. III-D |
| `tools/genfinal2/gf2_hook.py`, `_offsets()` lines 114-136, `descend()` lines 139-174 | learned start: bounded offsets line 130, start logits line 164 | Sec. III-D; replaces Alg. 1 line 4 |
| `tools/netsel/ns_hook.py`, `model_score()` lines 155-162, `select()` lines 165-197; `tools/netsel/ns_train.py`, `torch_fit()` lines 56-78 | learned net selection and its training | Sec. III-E |
| `tools/powerpred2/pp2_lib.py`, `cheapest_by_dir()` lines 35-46, `reducible_cap()` lines 49-66, `potential()` lines 104-106; `tools/powerpred2/pp2_sel.py` | power-aware net ranking by activity times removable capacitance | Sec. III-F |
| `tools/sched/run_n1.sh` and the flag files of Section 5 | launcher of the final flow | Sec. IV |

The rest of the snapshot is the support code these files load: route readers and writers, the timers, the overflow and reshape stages, and the native kernels with their build scripts.

## 3. Requirements and external tools

As stated in the files of this snapshot:

- Linux. `tools/best3/build_all.sh` stops on any other system (it tests only for Linux; the kernels and the CUDA toolkit assume x86-64).
- An NVIDIA GPU with the CUDA toolkit. `tools/best7/build_dg.sh` and `tools/best8/build_g.sh` take `ARCH=sm_90` for an H100 or `ARCH=sm_75` for a T4, and call `nvcc` from `/usr/local/cuda/bin/nvcc` unless `NVCC` is set. `libgnn_cuda.so` links cuBLAS.
- g++ with C++17 and OpenMP. `tools/best3/build_all.sh` names GCC 8.5.
- Python 3 with numpy. `tools/mid5/mid5_kern.cc` emulates the set table of CPython 3.12 to keep the order of floating-point sums, so Python 3.12 is the version the kernels match.
- torch, for module-level imports in `ispd24_adapter.py`, `dtla.py` and `dtla_gnn.py` and for training. `run_n1.sh` puts a stand-in for torch_scatter on `PYTHONPATH` from `tools/shims/pyshim`.
- GNU time at `/usr/bin/time` and `/proc/loadavg`, both used by `run_n1.sh`.

Public tools and data that are needed and not included:

- ISPD 2025 contest benchmarks with the NanGate45 kit, expected under `<repo>/ispd_bench/ISPD2025/ISPD2025_benchmarks/<suite>/<design>/`.
- ISPD 2025 contest evaluator and the organizers' OpenROAD fork, for official scores only; the flow does not call them.
- InstantGR, the GPU global router that provides the start routes `<design>.igrnf_e0r1.route`. Its `src/flute.hpp` is needed at `external/InstantGR/src` to build `tools/libflute_bridge.so` for FLUTE.
- `data.py` of the public DGR code, imported by `ispd24_adapter.py` at line 60.

## 4. Build

From the repository root, as the build scripts are written:

```bash
bash tools/best5/build_all.sh .      # runs tools/best4/build_all.sh and tools/best3/build_all.sh,
                                     # tools/timer5/build5.sh, the tools/par7 kernels, tools/best5/build_n.sh
ARCH=sm_90 bash tools/best7/build_dg.sh   # libdg_cpu.so, libdg_cuda.so and their _f64 variants, libdgb.so
ARCH=sm_90 bash tools/best8/build_g.sh    # libgnn_cpu.so, libgnn_cuda.so; learned start only
bash tools/best10/build_t.sh
bash tools/best11/build8.sh
bash tools/engine/build_inc.sh
CXX=g++ sh tools/fast_io/route_cpp/build.sh
```

The DTLA kernel itself is built by `tools/best7/build_dg.sh` with

```bash
nvcc -O3 -std=c++17 -arch=$ARCH -fmad=false -Xcompiler -fPIC -shared dg_kern.cu -o libdg_cuda.so
g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fPIC -pthread -shared dgb.cc -o libdgb.so
```

No build line in this snapshot covers `tools/gnnfix/dtimer/librc_topo.so`, which `dgr_stage_r.py` line 101 loads, `libh_iowrite_wf.so` from `tools/timefix/io_write_wf.cc`, `tools/gapla/libcsta` from `csta_kern.cc`, or the optional big-design variants `libcsta_b`, `libio_write_b` and `libroute_cpp_b` under `tools/srvw/bigcase`. `tools/best3/build_all.sh` line 51 also checks for `tools/pre5/weights/fold_*.npz`, which are not included, and then reports a failure.

## 5. Run

The final flow is started by `tools/sched/run_n1.sh`. Before a run, set the environment variable `T` (line 19) so that the repository is `$T/repo` and Python is `$T/env/venv/bin/python`, or set `R` and `PY` directly. Line 22 is a comment where the original machine sourced an environment script that is not in this snapshot; set `TMPDIR` and cache folders yourself. Set `KEEP=1` to keep the output routes; line 72 deletes them otherwise.

Ariane, BSG, NVDLA and MemPool-Tile use arm `n1y` in small mode. The settings are those recorded in `tools/exactpolish/cases_h100.txt`:

```bash
OUTD=<output dir> KEEP=1 SARM=Y DLS=99999 \
BIG8_SDFREE=0 FCX=dem,rcsub,b2par,ecopy,names FC_THP=1 SCHED_H100=1 \
SCHED_X="--sched_model off --sched_stop_frac 0.02 --sched_a4 1 --skip_ovf_rule 1 --sched_extra_dgr 4" \
bash tools/sched/run_n1.sh n1y small <visible|blind> <design> <tag> --dgr 20 --dgr_nets 16000 --dgr_rounds 5
```

The flow also reads a timer cache `experiments/gapla/csta/cache/<design>.<v|b>.npz`, `chain_y.py` line 395, or the file given by `--cache`. It is written by `save_cache()` in `tools/gapla/csta.py` line 184; no script in this snapshot produces it. MemPool-Group uses arm `n1` in big mode with the variable `D1X`; `cases_h100.txt` line 5 gives it as `"$MEM $DF --deadline_s <dls>"`, and `$MEM` and `$DF` are not defined in this snapshot. The value recorded for it by the authors, not run in this snapshot, is `D1X="--fp2 1 --writefast 1 --term_pinlat 2 --wprep 2 $(cat tools/flags_final/D.flags) --deadline_s 99999"`; each of these flags is declared in `tools/day1/chain_d1.py` (lines 1433-1440 and 1606-1608 for the `--dgr_*` ones).

The files `tools/day1/dl_est/D/<suite>.<design>.json` hold per-stage time estimates in seconds; `run_n1.sh` lines 59-61 read them and pass the file as `--deadline_est` when `DLS` is not `none`.

The start route is read from `$R/experiments/ispd25_routes[_blind]/<design>/<design>.igrnf_e0r1.route`, `run_n1.sh` line 50, unless `RT` names another file. In small mode the chain command is `BASE` with `CHOSEN.flags`, then `SPEED.flags`, `cfg_d.flags`, `D.flags` and `cfg_x.flags`, then the extra flags of the case row. `D.flags` sets `--dgr 8 --dgr_nets 8000 --dgr_rounds 3`, and the case row adds `--dgr 20 --dgr_nets 16000 --dgr_rounds 5` after it. `--dgr_t1 0.2` comes only from `D.flags`; the code default of `--dgr_t1` is 0.05.

| Setting in the paper | Value | Flag or variable | File |
|---|---|---|---|
| free nets N | 16000 | `--dgr_nets 16000` | `tools/exactpolish/cases_h100.txt` rows |
| when more than 300,000 nets can be freed, N is replaced by max(N, 7 percent of them) | fixed in code | none | `tools/best8/dgr_stage_r.py` lines 248-249 |
| steps K | 20 | `--dgr 20` | `cases_h100.txt` rows |
| Adam step size | lr = max of `--dgr_lr` 0.1 and `--dgr_lrk` / K | `--dgr_lrk 20` | `tools/flags_final/D.flags`; rule `dg_kern.cu` line 676 |
| Adam moment factors | 0.9 and 0.999 | fixed in code | `tools/best7/dg_kern.cu` line 567 |
| start scale lambda | 4.0 | `--dgr_init`, default | `tools/final/chain_y.py` line 1558 |
| first temperature T_1 | 1.0 | `--dgr_t0`, default | `chain_y.py` line 1558 |
| last temperature T_K | 0.2 | `--dgr_t1 0.2` | `tools/flags_final/D.flags` only; the code default is 0.05 |
| smoothing tau (ns in the paper) | 0.002 | `--dgr_tau`, default | `chain_y.py` line 1555 |
| rounds M | 5 | `--dgr_rounds 5` | `cases_h100.txt` rows |
| stop threshold epsilon (score units) | 0.01 | `--dgr_round_min`, default | `chain_y.py` line 1563 |
| stop share eta | 0.02 | `--sched_stop_frac 0.02` in `SCHED_X` | `cases_h100.txt` line 17 |
| end rounds: one extra call of at most 4 rounds, same stop rules as any call | 4 | `--sched_extra_dgr 4` in `SCHED_X` | `cases_h100.txt` line 17 |
| 3 percent arc rule | 0.03 | `--dgr_dl 0.03` | `tools/flags_final/D.flags` |
| halvings of a rejected change | 3 | `--dgr_split`, default | `chain_y.py` line 1562 |
| `--dgr_at fp`: DTLA after the first pass. `--dgr_skip_fp 3`: the first pass is also run whenever it and one round fit the deadline (needs `--lam 0 --gate 0`). `--dgr_defer 1`: at a state without a timing violation the overflow stage goes before the descent (help text, `chain_y.py` lines 1564-1565) | | `--dgr_at fp --dgr_skip_fp 3 --dgr_defer 1` | `tools/flags_final/D.flags` |
| arrival change that is not passed on | 0.0001 | `--dgr_dat 0.0001` | `tools/flags_final/D.flags`; not stated in the paper |
| CPU threads | 8 | `THR`, default | `run_n1.sh` line 23 |
| MemPool-Cluster free nets and time limit | the paper gives other values for the scored runs; the launch line is not in this snapshot | | |

## 6. Supporting techniques

None of them is on in the final flow. Each is switched on as follows.

- Learned start: `GF2_MODE=gens` with `GF2_MODELS=<dir>`, where `<dir>/<family>/gen1.bin` is read at `gf2_hook.py` line 117 and the families are mapped at line 21. The flow is started through `tools/genfinal2/gf2_drive.py`; `gf2_patch.py --runner <in> <out>` writes a launcher that calls it from `GF2_DIR`. The number of steps is `--dgr`. One weight file is included at `experiments/syndata1_2026-11-14/train/gen/models/ariane/gen1.bin`. Only the family ariane has weights here; for another design `gf2_hook.descend()` raises "model ..." and the run falls back to the plain start with a `[gf2] FALLBACK` line.
- Learned net selection: `NS_MODE=learned` and `NS_MODEL=<model.npz>`, started through `tools/netsel/ns_launch.py <driver> <args>` with `NS_REPO=<repo root>`. It is combined with `run_n1.sh` by putting `tools/netsel/ns_launch.py` in place of the `gg_launch3.py` slot of line 67 (that slot is used only when `GG_GNN` is set; do not set `GG_GNN`, as `tools/groupgnn` is not included). `NS_LOG=<prefix>` writes the logs that `ns_train.py fit` reads. `NS_FRAC` defaults to 0.15, `NS_FROM` to 1 and `NS_FORCE_PREV` to 1. The model is written by `ns_train.py fit --logs <dir> --out model.npz` from runs with `NS_MODE=log`.
- Power-aware ranking: `PP2_LIST=score`, with `tools/powerpred2y/cold_y_pp2.py` in place of `cold_n1.py` in the folder named by `SCHED_DIR`, as its docstring describes. `PP_MODE` is `off` or `log`, and `PP_OUT` names the record files.

## 7. What is not included, and why

- Contest benchmarks, the evaluator, the router and FLUTE: public tools of their owners, listed in Section 3.
- Start routes and timer caches: generated data, too large for a code snapshot.
- The trained net-selection model, the training data of the learned start and of the net selection, and the synthetic designs: generated data, not code.
- The launch lines of the MemPool-Cluster runs and of the tests of the supporting techniques: they are in machine-specific scripts that were left out.
- The environment script of the original machine and the prior weights `tools/pre5/weights/fold_*.npz`: left out with the machine-specific files. The flow loads the prior weights only when `--lam` or `--gate` is above zero, `chain_y.py` line 360, and the final flow sets both to zero.
- The module `dtla_lagnn` (imported by `ispd24_adapter.py`, `fastmode.py`, `fastmode_f3.py`, `fastmode4.py`, `tools/fast_la/gapla_dp_v2.py`, `tools/pre5/pre5_load.py`, `tools/pre5/gnn_np.py`, `tools/fast3x/h_graph.py`, `chain_y.py` line 365 and `chain_d1.py` line 403) is not included. The imports in the final flow sit on optional paths (`--pre5_gnn torch|check`, `--gnn`). Also not included and used only when the matching switch is on: `d2_stage` (`--d2 >= 0`), `prof8` (`LOAD8_PROF`), `sched_rt` (`--sched_model` not `off`), `gg_hook3` (`GG_GNN`).
- Big mode (`--fp2 1`) imports `load8_lib_f2`, `build2_fast_f2` and `fastmode_f3` by module name; they are placed at `tools/best11/load8_lib_f2.py`, `tools/onepass/build2_fast_f2.py` and at the root (`fastmode_f3.py`, `sortkern_f3.cc`). They are near copies of `load8_lib.py`, `build2_fast.py` and `fastmode.py`.
- Build lines are missing for several libraries; see Section 4. The check of `tools/best3/build_all.sh` for prior weights ends with a failure status even when all libraries build, and `tools/powerpred2/pp2_hook.py` names `c1hfast/libnorm_write` (changed from `final/libnorm_write` in this snapshot, to match the path where `build_all.sh` builds it).
- Experiment history, earlier versions of the flow and result files.

## 8. License

This snapshot accompanies an anonymous submission.

Its timers and its activity propagation were written to reproduce OpenSTA and are ports of parts of it, or build on such ports:

- the delay calculation in `fast_sta.py`, `fast_sta_v2.py` and `fast_sta_v3.py`, which is OpenSTA's `dmp_ceff_elmore` delay calculator, ported and vectorized, as their headers say;
- its C++ and GPU versions in `tools/gapla/csta_kern.cc`, `tools/timer5/csta5_kern.cc`, `tools/srvw/bigcase/csta_kern_b.cc` and `tools/weights/gt_kern_w.cu`;
- the activity propagation in `fast_power.py`, `tools/speed4/act_kern.cc` and `tools/pre5/act_par.cc`.

OpenSTA is distributed under the GNU General Public License, version 3. The snapshot as a whole is therefore distributed under the GNU General Public License, version 3; its text is in the file `LICENSE`.

Tools that the flow needs and that are not included keep their own licenses; see Section 3.
