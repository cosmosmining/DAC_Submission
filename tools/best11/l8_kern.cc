// l8_kern.cc: the single-thread numpy calls of the big-design load, on threads, same integers (tests: tools/best11/test_kern8.py, test_sta8.py).
//   l8_argsort     == np.argsort(key, kind="stable")                      (LSD radix on threads, stable)
//   l8_take8/1     == src[idx]                                            (8-byte / 1-byte elements)
//   l8_bounds      == np.searchsorted(a, np.arange(m)) for sorted a       (one pass)
//   l8_ce_*        == the cell rf-edge expansion + sort + group tables of fast_sta_v3.StaDesign._graph (tools/pre5/pre5_kern.graph)
//   l8_levelize    == tools/pre5/pre_kern.cc pre_levelize (longest-path levels; edges sorted per level pass on threads)
//   l8_pin_maps    == ovf_lib.pin_count_map and the per-net pin counts of chain_gb's _pin_maps_arrays, without the global sort
//   l8_seg32/64    == csta.seg_from_route's column work
// Build: g++ -std=c++17 -O3 -fopenmp -fPIC -shared l8_kern.cc -o libl8_kern.so
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <atomic>
#include <omp.h>
typedef int64_t i64; typedef uint64_t u64; typedef uint32_t u32;

static inline int bits_of(u64 v){ int b=0; while(v){ b++; v>>=1; } return b; }

struct KV { u64 k; u64 i; };

// stable LSD radix on (k, i); digits of 11 bits; a digit equal on all keys is skipped
static void radix(i64 n, KV*& A, KV*& B, int kbits, int nt){
  const int R=11, S=1<<R;
  std::vector<i64> hist((size_t)nt*S), pos((size_t)nt*S);
  for(int sh=0; sh<kbits; sh+=R){
    std::fill(hist.begin(),hist.end(),0);
    #pragma omp parallel num_threads(nt)
    {
      int t=omp_get_thread_num(); i64 lo=n*t/nt, hi=n*(t+1)/nt; i64* h=&hist[(size_t)t*S];
      for(i64 i=lo;i<hi;i++) h[(A[i].k>>sh)&(S-1)]++;
    }
    i64 cum=0; bool one=false;
    for(int d=0;d<S;d++){ i64 dn=0; for(int t=0;t<nt;t++){ pos[(size_t)t*S+d]=cum; cum+=hist[(size_t)t*S+d]; dn+=hist[(size_t)t*S+d]; } if(dn==n) one=true; }
    if(one) continue;
    #pragma omp parallel num_threads(nt)
    {
      int t=omp_get_thread_num(); i64 lo=n*t/nt, hi=n*(t+1)/nt; i64* p=&pos[(size_t)t*S];
      for(i64 i=lo;i<hi;i++){ B[p[(A[i].k>>sh)&(S-1)]++]=A[i]; }
    }
    std::swap(A,B);
  }
}

// ----------------------------------------------------------------- csta.seg_from_route's columns (the entry points are below)
template<class T> static i64 seg_cols(i64 nn, const i64* de, const u64* off, const T* x0, const T* y0, const T* z0, const T* x1, const T* y1, const T* z1, i64 grid, i64* out, int nt){
  // out has room for off[nn] rows of 7; rows with de < 0 are dropped (same order)
  std::vector<i64> kept(nn+1,0);
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 n=0;n<nn;n++) kept[n+1]= de[n]>=0 ? (i64)(off[n+1]-off[n]) : 0;
  for(i64 n=0;n<nn;n++) kept[n+1]+=kept[n];
  auto fd=[&](i64 a){ i64 q=a/grid; if((a%grid!=0)&&((a<0)!=(grid<0))) q--; return q; };
  #pragma omp parallel for num_threads(nt) schedule(dynamic,2048)
  for(i64 n=0;n<nn;n++){
    if(de[n]<0) continue;
    i64 r=kept[n]*7;
    for(u64 s=off[n];s<off[n+1];s++){
      out[r]=de[n]; out[r+1]=fd(x0[s]); out[r+2]=fd(y0[s]); out[r+3]=z0[s]; out[r+4]=fd(x1[s]); out[r+5]=fd(y1[s]); out[r+6]=z1[s]; r+=7;
    }
  }
  return kept[nn];
}

extern "C" {

// keys of any sign: offset by the minimum (order and ties unchanged)
void l8_argsort(i64 n, const i64* key, i64* out, int nt){
  if(n<=0) return;
  i64 mn=key[0], mx=key[0];
  #pragma omp parallel for num_threads(nt) reduction(min:mn) reduction(max:mx) schedule(static)
  for(i64 i=0;i<n;i++){ if(key[i]<mn) mn=key[i]; if(key[i]>mx) mx=key[i]; }
  KV* A=(KV*)malloc(n*sizeof(KV)); KV* B=(KV*)malloc(n*sizeof(KV));
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 i=0;i<n;i++){ A[i].k=(u64)key[i]-(u64)mn; A[i].i=(u64)i; B[i].k=0; }
  radix(n,A,B,bits_of((u64)mx-(u64)mn),nt);
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 i=0;i<n;i++) out[i]=(i64)A[i].i;
  free(A); free(B);
}

void l8_take8(i64 n, const i64* idx, const i64* src, i64* dst, int nt){
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 i=0;i<n;i++) dst[i]=src[idx[i]];
}
void l8_take1(i64 n, const i64* idx, const uint8_t* src, uint8_t* dst, int nt){
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 i=0;i<n;i++) dst[i]=src[idx[i]];
}

// out[v] = number of elements of the sorted a that are < v, v = 0 .. m-1
void l8_bounds(i64 n, const i64* a, i64 m, i64* out){
  i64 j=0;
  for(i64 v=0;v<m;v++){ while(j<n&&a[j]<v) j++; out[v]=j; }
}

// ----------------------------------------------------------------- levelize: pre_levelize's values, level by level on threads
// lev[j] = longest path; pins on / after a cycle: level 0, left = 1. Duplicate edges count separately.
int l8_levelize(i64 P, i64 ne, const i64* efr, const i64* eto, i64* lev, uint8_t* left, int nt){
  std::vector<i64> off(P+1,0), adj(ne); std::vector<std::atomic<int32_t>> indeg(P);
  #pragma omp parallel num_threads(nt)
  {
    int t=omp_get_thread_num(), T=omp_get_num_threads(); const i64 a=P*t/T, b=P*(t+1)/T;      // each thread: one range of pins, one scan of the edge list (sequential reads)
    for(i64 i=a;i<b;i++) indeg[i].store(0,std::memory_order_relaxed);
    #pragma omp barrier
    for(i64 e=0;e<ne;e++){ i64 f=efr[e], x=eto[e]; if(f>=a&&f<b) off[f+1]++; if(x>=a&&x<b) indeg[x].store(indeg[x].load(std::memory_order_relaxed)+1,std::memory_order_relaxed); }
  }
  for(i64 i=0;i<P;i++) off[i+1]+=off[i];
  { std::vector<i64> pos(off.begin(),off.end()-1);
    #pragma omp parallel num_threads(nt)
    {
      int t=omp_get_thread_num(), T=omp_get_num_threads(); const i64 a=P*t/T, b=P*(t+1)/T;
      for(i64 e=0;e<ne;e++){ i64 f=efr[e]; if(f>=a&&f<b) adj[pos[f]++]=eto[e]; }
    } }
  std::vector<uint8_t> done(P,0); std::vector<i64> front, nextf;
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 i=0;i<P;i++){ lev[i]=0; }
  for(i64 i=0;i<P;i++) if(indeg[i].load(std::memory_order_relaxed)==0){ front.push_back(i); done[i]=1; }
  i64 k=0;
  while(!front.empty()){
    k++;
    i64 nf=(i64)front.size(); nextf.clear();
    if(nf<20000){
      for(i64 q=0;q<nf;q++){ i64 i=front[q]; for(i64 e=off[i];e<off[i+1];e++){ i64 j=adj[e]; if(indeg[j].fetch_sub(1,std::memory_order_relaxed)==1){ nextf.push_back(j); lev[j]=k; done[j]=1; } } }
    } else {
      #pragma omp parallel num_threads(nt)
      {
        std::vector<i64> loc;
        #pragma omp for schedule(dynamic,4096) nowait
        for(i64 q=0;q<nf;q++){ i64 i=front[q]; for(i64 e=off[i];e<off[i+1];e++){ i64 j=adj[e]; if(indeg[j].fetch_sub(1,std::memory_order_acq_rel)==1){ loc.push_back(j); lev[j]=k; done[j]=1; } } }
        #pragma omp critical
        nextf.insert(nextf.end(),loc.begin(),loc.end());
      }
    }
    front.swap(nextf);
  }
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 i=0;i<P;i++){ if(!done[i]) lev[i]=0; left[i]=!done[i]; }
  return 0;
}

// ----------------------------------------------------------------- _graph: cell rf-edges
// Mask of (orf, irf) on arc a: td = orf ? tcf : tcr; same = irf == orf; ok_sense = sense == 0 ? same : (sense == 1 ? !same : 1);
// ok_kind = kind == 1 ? irf == 0 : (kind == 2 ? irf == 1 : ok_sense); keep = td >= 0 && ok_kind. Order: (orf, irf) = 00 01 10 11, each by position in ai.
static inline bool ce_keep(int orf, int irf, i64 a, const i64* kind, const i64* sense, const i64* tcr, const i64* tcf){
  i64 td=orf?tcf[a]:tcr[a]; if(td<0) return false;
  bool same=(irf==orf); i64 sn=sense[a]; bool oks= sn==0? same : (sn==1? !same : true);
  i64 kd=kind[a]; bool okk= kd==1? irf==0 : (kd==2? irf==1 : oks);
  return okk;
}
// returns the number of edges m
i64 l8_ce_count(i64 na, const i64* ai, const i64* kind, const i64* sense, const i64* tcr, const i64* tcf, int nt){
  i64 m=0;
  #pragma omp parallel for num_threads(nt) schedule(static) reduction(+:m)
  for(i64 q=0;q<na;q++){ i64 a=ai[q]; for(int o=0;o<2;o++) for(int r=0;r<2;r++) if(ce_keep(o,r,a,kind,sense,tcr,tcf)) m++; }
  return m;
}
// fills the eight ce_ arrays (sorted by (lev[to], to, orf), ties in the expansion order == np.lexsort((orf, to, lev[to]))) and ce_grp, ce_lo (nlev + 1); returns the number of groups ng;
// g_start must have room for m. Returns -1 if the composite key does not fit in 64 bits (nothing written; the caller uses numpy).
i64 l8_ce_fill(i64 na, const i64* ai, const i64* fr, const i64* to, const i64* kind, const i64* sense, const i64* tcr, const i64* tcf, const i64* trt, const i64* tft, const i64* lib,
               const i64* lev, i64 nlev, i64 P, i64 m,
               i64* ce_arc, i64* ce_irf, i64* ce_orf, i64* ce_td, i64* ce_ts, i64* ce_from, i64* ce_to, i64* ce_lib, i64* ce_lo, i64* ce_grp, i64* g_start, int nt){
  int bp=bits_of((u64)std::max<i64>(P-1,0)), bl=bits_of((u64)std::max<i64>(nlev-1,0));
  if(bp+1+bl>64) return -1;
  // positions: chunks in order inside each of the four combinations
  const int nch=std::max(1,nt*8);
  std::vector<i64> cnt((size_t)4*nch,0);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    i64 lo=na*c/nch, hi=na*(c+1)/nch; i64 cc[4]={0,0,0,0};
    for(i64 q=lo;q<hi;q++){ i64 a=ai[q]; for(int o=0;o<2;o++) for(int r=0;r<2;r++) if(ce_keep(o,r,a,kind,sense,tcr,tcf)) cc[o*2+r]++; }
    for(int w=0;w<4;w++) cnt[(size_t)w*nch+c]=cc[w];
  }
  std::vector<i64> base((size_t)4*nch); { i64 s=0; for(int w=0;w<4;w++) for(int c=0;c<nch;c++){ base[(size_t)w*nch+c]=s; s+=cnt[(size_t)w*nch+c]; } }
  KV* A=(KV*)malloc(m*sizeof(KV)); KV* B=(KV*)malloc(m*sizeof(KV));
  i64* xarc=(i64*)malloc(m*sizeof(i64));          // expansion order: arc id; the combination comes from the position
  uint8_t* xw=(uint8_t*)malloc(m);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    i64 lo=na*c/nch, hi=na*(c+1)/nch; i64 p[4]; for(int w=0;w<4;w++) p[w]=base[(size_t)w*nch+c];
    for(i64 q=lo;q<hi;q++){ i64 a=ai[q];
      for(int o=0;o<2;o++) for(int r=0;r<2;r++) if(ce_keep(o,r,a,kind,sense,tcr,tcf)){ i64 x=p[o*2+r]++; xarc[x]=a; xw[x]=(uint8_t)(o*2+r);
        i64 t=to[a]; A[x].k=((u64)lev[t]<<(bp+1))|((u64)t<<1)|(u64)o; A[x].i=(u64)x; B[x].k=0; } }
  }
  radix(m,A,B,bp+1+bl,nt);
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 i=0;i<m;i++){
    i64 x=(i64)A[i].i, a=xarc[x]; int w=xw[x], o=w>>1, r=w&1;
    ce_arc[i]=a; ce_irf[i]=r; ce_orf[i]=o; ce_td[i]=o?tcf[a]:tcr[a]; ce_ts[i]=o?tft[a]:trt[a];
    ce_from[i]=fr[a]; ce_to[i]=to[a]; ce_lib[i]=lib[a];
  }
  free(xarc); free(xw); free(A); free(B);
  // ce_lo == np.searchsorted(lev[ce_to], np.arange(nlev + 1))
  #pragma omp parallel for num_threads(nt) schedule(dynamic,16)
  for(i64 v=0;v<=nlev;v++){ i64 a=0,b=m; while(a<b){ i64 h=(a+b)/2; if(lev[ce_to[h]]<v) a=h+1; else b=h; } ce_lo[v]=a; }
  // groups: key = ce_to * 2 + ce_orf; new group where it changes; ce_grp = cumsum(new) - 1; g_start = positions of the new groups
  std::vector<i64> gc(nch+1,0);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){ i64 lo=m*c/nch, hi=m*(c+1)/nch, g=0; for(i64 i=lo;i<hi;i++) if(i==0||ce_to[i]*2+ce_orf[i]!=ce_to[i-1]*2+ce_orf[i-1]) g++; gc[c+1]=g; }
  for(int c=0;c<nch;c++) gc[c+1]+=gc[c];
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){ i64 lo=m*c/nch, hi=m*(c+1)/nch, g=gc[c]; for(i64 i=lo;i<hi;i++){ if(i==0||ce_to[i]*2+ce_orf[i]!=ce_to[i-1]*2+ce_orf[i-1]) g_start[g++]=i; ce_grp[i]=g-1; } }
  return gc[nch];
}
// g_to, g_orf, g_lo (nlev + 1; == np.searchsorted(lev[ce_to][g_start], np.arange(nlev + 1)))
void l8_ce_groups(i64 ng, const i64* g_start, const i64* ce_to, const i64* ce_orf, const i64* lev, i64 nlev, i64* g_to, i64* g_orf, i64* g_lo, int nt){
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(i64 g=0;g<ng;g++){ g_to[g]=ce_to[g_start[g]]; g_orf[g]=ce_orf[g_start[g]]; }
  #pragma omp parallel for num_threads(nt) schedule(dynamic,16)
  for(i64 v=0;v<=nlev;v++){ i64 a=0,b=ng; while(a<b){ i64 h=(a+b)/2; if(lev[g_to[h]]<v) a=h+1; else b=h; } g_lo[v]=a; }
}

// ----------------------------------------------------------------- overflow pin tables
// nn nets; pin_off[nn+1], ap_off[np+1], aps[nap*3] int32 (l, x, y). cnt[n] = pins of net n with >= 1 access point.
// pc[x*Y+y] (float64, X*Y) = number of canon nets with an in-bounds access point at (x, y) (each net once per cell) == bincount(unique(ap_net*XY + x*Y + y) % XY)
void l8_pin_maps(i64 nn, const i64* pin_off, const i64* ap_off, const int32_t* aps, const uint8_t* canon, i64 X, i64 Y, double* pc, i64* cnt, int nt){
  const i64 XY=X*Y;
  std::vector<i64> tot(XY,0);
  #pragma omp parallel num_threads(nt)
  {
    std::vector<i64> h(XY,0), v;
    #pragma omp for schedule(dynamic,4096) nowait
    for(i64 n=0;n<nn;n++){
      i64 c=0; v.clear(); bool cn=!canon||canon[n];
      for(i64 p=pin_off[n];p<pin_off[n+1];p++){
        i64 a=ap_off[p], b=ap_off[p+1]; if(b>a) c++;
        if(cn) for(i64 t=a;t<b;t++){ i64 x=aps[t*3+1], y=aps[t*3+2]; if(x>=0&&x<X&&y>=0&&y<Y) v.push_back(x*Y+y); }
      }
      cnt[n]=c;
      if(v.size()>1){ std::sort(v.begin(),v.end()); v.erase(std::unique(v.begin(),v.end()),v.end()); }
      for(i64 q: v) h[q]++;
    }
    #pragma omp critical
    for(i64 i=0;i<XY;i++) tot[i]+=h[i];
  }
  for(i64 i=0;i<XY;i++) pc[i]=(double)tot[i];
}

// ----------------------------------------------------------------- csta.seg_from_route's columns
i64 l8_seg32(i64 nn, const i64* de, const u64* off, const int32_t* x0, const int32_t* y0, const int32_t* z0, const int32_t* x1, const int32_t* y1, const int32_t* z1, i64 grid, i64* out, int nt){ return seg_cols(nn,de,off,x0,y0,z0,x1,y1,z1,grid,out,nt); }
i64 l8_seg64(i64 nn, const i64* de, const u64* off, const i64* x0, const i64* y0, const i64* z0, const i64* x1, const i64* y1, const i64* z1, i64 grid, i64* out, int nt){ return seg_cols(nn,de,off,x0,y0,z0,x1,y1,z1,grid,out,nt); }

}  // extern C
