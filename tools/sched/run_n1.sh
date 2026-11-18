#!/bin/bash
# run_n1.sh (sched 2026-11-12, NEW; plan N1 / N2): run_sched.sh (below) with two more arms that start the chain through
# tools/sched/cold_n1.py (cold_sched.py + fixcut's cuts + the N2 big-mode --dgr_skip_fp fix):
#   n1 / n1y   big (chain_d1) / smaller designs (chain_y); cold_n1 switches from env SCHED_X; cuts from env FCX (default
#              "dem,rcsub,b2par,ecopy,names" as tools/fixcut/run_fc.sh; FCX="" = no cut); FC_THP=1 adds
#              GLIBC_TUNABLES=glibc.malloc.hugetlb=1 (fixcut's THP). GNN start, UM headroom, GPU library: env passed through.
# run_sched.sh (sched 2026-11-12, NEW). One run of the flow on the H100, exactly as tools/h100big/run_s3h.sh (same command, same
# environment, same summary line), with four more arms that start the chain through tools/sched/cold_sched.py:
#   d1   / cur    current: tools/day1/cold_d1.py (big designs) / tools/final/cold_y.py (frozen arm, smaller designs)   [= run_s3h.sh]
#   sd1  / scur   fixed:   same chain (chain_d1 / chain_y) with the learned deadline rule (tools/sched/sched_rt.py, model_v1.json)
#   sfd1 / sfcur  fixed + DTLA sizing (--sched_fit 1): a DTLA round that does not fit frees fewer nets instead of being skipped
#   xd1  / xy     cold_sched.py with the switches of env SCHED_X (profile, anytime fill, ...)
#   run_sched.sh <arm> <mode big|small> <suite> <design> <tag> [extra chain flags...]
# Env as run_s3h.sh (OUTD, THR, KEEP, NICE, DLS for small mode, D1X for d1/sd1/sfd1, SARM), plus
#   SQ (unit/tail quantile, default 0.8), SMG (margin s, default 0), SCHED_H100 (1 = H100 quantiles; default: detected from the
#   driver), SCHED_DIR (default: this file's folder; holds cold_sched.py,
#   sched_rt.py, sched_model.py, model_v1.json). The fixed arms log their decisions to <out>.sched.jsonl and walls.sched of the json.
# Nothing in the repo is changed; the repo (R) is only read.
T=${T:-/path/to/workdir}; R=${R:-$T/repo}; PY=${PY:-$T/env/venv/bin/python}   # set T (or R and PY) in the environment
SD=${SCHED_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd $R || exit 1
# (environment setup script of the original machine: not included; set TMPDIR and cache folders yourself)
ARM=$1; MODE=$2; S=$3; D=$4; N=$5; shift 5; TH=${THR:-8}; X=$([ $S = blind ] && echo _blind)
O=${OUTD:-$T/runs/sched_2026-11-12/val}; mkdir -p $O; K=$S.$D.$N.$ARM
CH="$(cat tools/best3/jobs/CHOSEN.flags)"; CD="$(cat tools/best5/cfg_d.flags)"; SPD="$(cat tools/best3/jobs/SPEED.flags)"
BASE="--design $D --suite $S --snaps 5:4 --only_b --escape auto --of_rule official --promote_impl full --ovf auto --h_all 1 --h_pre 1 $CH"
FCE=""
SCH="--sched_root $R --sched_model $SD/model_v1.json --sched_q ${SQ:-0.8} --sched_margin ${SMG:-0} --sched_log $O/$K.sched.jsonl"
case $ARM in
  cur)   PR=tools/final/cold_y.py; AF="";;
  d1)    PR=tools/day1/cold_d1.py; AF="${D1X:-}";;
  scur)  PR="$SD/cold_sched.py --sched_src y $SCH"; AF="";;
  sd1)   PR="$SD/cold_sched.py --sched_src d1 $SCH"; AF="${D1X:-}";;
  sfcur) PR="$SD/cold_sched.py --sched_src y $SCH --sched_fit 1"; AF="";;
  sfd1)  PR="$SD/cold_sched.py --sched_src d1 $SCH --sched_fit 1"; AF="${D1X:-}";;
  xd1)   PR="$SD/cold_sched.py --sched_src d1 --sched_root $R ${SCHED_X:-}"; AF="${D1X:-}";;    # any cold_sched switches (env SCHED_X)
  xy)    PR="$SD/cold_sched.py --sched_src y --sched_root $R ${SCHED_X:-}"; AF="";;
  n1)    PR="$SD/cold_n1.py --sched_src d1 --sched_root $R ${SCHED_X:-}"; AF="${D1X:-}"; FCE="FIXCUT=${FCX-dem,rcsub,b2par,ecopy,names}";;
  n1y)   PR="$SD/cold_n1.py --sched_src y --sched_root $R ${SCHED_X:-}"; AF=""; FCE="FIXCUT=${FCX-dem,rcsub,b2par,ecopy,names}";;
  *) echo "arm?"; exit 2;;
esac
[ "${FC_THP:-0}" = 1 ] && FCE="$FCE GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
touch $O/$K.done
if [ $MODE = big ]; then
  RT=${RT:-$T/runs/fast_path/books/${D}_$S/$D.igrnf_e0r1.route}
  CMD="$PR --load8 1 $BASE --lam 0 --mid5 1 --late5 1 --big5 1 --big5_mem 1 --big5_act thread --big8 1 $CD --route_wall 0 --io_cold_lib --io_bind_mem --route $RT --dl_chunks 0 --gate 0 --ovf off --dl_ovf_chunk 1"
  XE="P11_WAIT_ROUTE=$O/$K.done FASTLOAD_THREADS=$TH MALLOC_ARENA_MAX=2"
else
  [ -n "$DLS" ] || { echo "small mode needs DLS (seconds, or none)"; exit 2; }
  RT=${RT:-$R/experiments/ispd25_routes$X/$D/$D.igrnf_e0r1.route}
  case ${SARM:-Y} in
    Y)  AX="$(cat tools/flags_final/D.flags)"; EX="";;
    YH) AX="$(cat tools/flags_final/D.flags)"; EX="--dl_chunks 2";;
    F0) AX="--lam 0 --gate 0 --ovf_max_runs 200"; EX="--dl_chunks 0";;
    *) echo "SARM?"; exit 2;;
  esac
  CMD="$PR $BASE $SPD --big5 1 --route $RT $CD $AX $(cat tools/best10/cfg_x.flags) --load8 1 --dl_ovf_chunk 1 --dl_order term $EX"
  [ "$DLS" != none ] && CMD="$CMD --deadline_s $DLS"
  E=tools/day1/dl_est/D/$S.$D.json
  [ -s $E ] || E=tools/best8/dl_est/D/$S.$D.json; [ -s $E ] || E=tools/timefix/dl_est/D/$S.$D.json
  [ "${SARM:-Y}" != F0 ] && [ "$DLS" != none ] && [ -s $E ] && CMD="$CMD --deadline_est $E"
  XE="FASTLOAD_THREADS=$TH"
fi
L0=$(cut -d' ' -f1-3 /proc/loadavg); t0=$(date +%s.%N)
env $XE $FCE ${DGR_LOCK:+DGR_LOCK=$DGR_LOCK} DL_T0_EPOCH=$t0 FAST_STA_V3=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=$TH CSTA_THREADS=$TH PF_THREADS=$TH MID5_THREADS=$TH \
  PYTHONPATH=$R/tools/shims/pyshim ONEPASS_DATA_ROOT=$R LC_ALL=C H_TMP=$T/tmp \
  nice -n ${NICE:-10} /usr/bin/time -v -o $O/$K.time $PY -u $R/tools/h100big/hard_exit.py ${GG_GNN:+$R/tools/groupgnn/gg_launch3.py} $CMD $AF "$@" --audit_open $O/$K.opened --cold_json $O/$K.cold.json --wait_route $O/$K.done \
  --out $O/$K --json $O/$K.json > $O/$K.out 2>&1
rc=$?
t1=$(date +%s.%N)
h=$(md5sum < $O/$K.b4.route 2>/dev/null | cut -c1-32)
[ "${KEEP:-0}" = 1 ] || rm -f $O/$K.*.route
rm -f $O/$K.done $O/$K.opened $O/$K.opened.child
W=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.2f", b-a}')
echo "$K arm=$ARM rc=$rc md5 ${h:-NONE} wall_s $W maxrss_kb $(grep Maximum $O/$K.time | awk '{print $NF}') host $(hostname -s) thr $TH load0 ${L0// /,} load1 $(cut -d' ' -f1-3 /proc/loadavg | tr ' ' ,) sarm ${SARM:-Y} dls ${DLS:-} fc [$FCE] flags_extra [$AF $*]" | tee -a $O/summary.log
