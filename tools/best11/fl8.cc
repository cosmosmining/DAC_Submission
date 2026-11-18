// fl8.cc: tools/comp2/loader/fastload.cc with its serial phases on threads; same output arrays and blobs, byte for byte (test: tools/best11/test_fl8.py):
//   read_file: I/O threads FL8_IOT (default 16; on a cold network file system they only wait); section search on threads; COMPONENTS / net names / pin interning: per-chunk tables merged
//   in chunk order (= first appearance over the whole file), arrays filled on threads; name tables: lock-free insertion (later duplicate wins, as before); def_power_text: the text
//   tools/speed4/act_inproc.power_pins_from_def would scan, without a second read of the file. FL8_TIME=1 prints the phase seconds to stderr.
// Build: g++ -std=c++17 -O3 -fopenmp -fPIC -shared fl8.cc -o libfl8.so
// fastload.cc: chunked multi-threaded (OpenMP) single-pass parsers for the preload of the fast mode:
//   ISPD'25 .net, DEF (COMPONENTS / PINS / NETS -> the StaDesign pin tables), power activity file, and exact-name hash tables.
// Pattern: HeteroSTA style -- read the file in parallel, split the text at record boundaries, parse the chunks on all threads,
// hand flat arrays over in memory (no snapshot, no cache, nothing persistent).
// Build: g++ -std=c++17 -O3 -fopenmp -fPIC -shared fastload.cc -o libfastload.so
// Every parser reproduces the semantics of the Python original (fast_timer.parse_net_fast, ispd25_adapter.parse_net_ispd25,
// fast_sta_v3.parse_def_netlist + the pin loop of StaDesign.__init__, fastmode4.State4.activity_file); anything unusual sets
// `bad` and the Python side falls back to the original function.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string_view>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <omp.h>
using std::vector; using std::string;
static inline double wall(){ return omp_get_wtime(); }
struct PT { const char* tag; double t; bool on; PT(const char* g):tag(g),t(wall()),on(getenv("FL8_TIME")!=nullptr){} void mark(const char* m){ if(on){ double n=wall(); fprintf(stderr,"[fl8] %s %s %.3f\n",tag,m,n-t); t=n; } } };
// first occurrence of pat in [p, p+n): == memmem, on threads
static const char* pmemmem(const char* p, int64_t n, const char* pat, size_t pl, int nt){
  if(n<(int64_t)(16<<20)||nt<=1) return (const char*)memmem(p,n,pat,pl);
  const int64_t CH=8<<20; int64_t nc=n/CH+1; std::atomic<int64_t> best(n);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int64_t c=0;c<nc;c++){
    int64_t o=c*CH; if(o>=best.load(std::memory_order_relaxed)) continue;
    int64_t e=std::min(n,o+CH+(int64_t)pl-1);
    if(e-o<(int64_t)pl) continue;
    const char* q=(const char*)memmem(p+o,e-o,pat,pl);
    if(q){ int64_t v=q-p, b=best.load(); while(v<b&&!best.compare_exchange_weak(b,v)){} }
  }
  int64_t b=best.load(); return b<n? p+b : nullptr;
}

static inline bool isws(unsigned char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }

struct Buf { char* p=nullptr; int64_t n=0; ~Buf(){ free(p); } };
static bool read_file(const char* path, Buf& b, int nt){
  int fd=open(path,O_RDONLY); if(fd<0) return false;
  struct stat st; if(fstat(fd,&st)!=0){ close(fd); return false; }
  int64_t n=st.st_size; b.p=(char*)malloc(n+16); if(!b.p){ close(fd); return false; } b.n=n;
  const int64_t PIECE=32<<20; int64_t np=(n+PIECE-1)/PIECE; bool ok=true;
  int iot=std::max(nt,getenv("FL8_IOT")?atoi(getenv("FL8_IOT")):16);
  #pragma omp parallel for num_threads(iot) schedule(dynamic,1)
  for(int64_t i=0;i<np;i++){
    int64_t o=i*PIECE, e=std::min(n,o+PIECE);
    while(o<e){ ssize_t r=pread(fd,b.p+o,e-o,o); if(r<=0){ ok=false; break; } o+=r; }
  }
  close(fd); memset(b.p+n,0,16); return ok;
}

// ============================================================ .net
struct NPart {
  string names; vector<int32_t> npins;                 // per net
  string pnames; vector<double> slack; vector<int32_t> nparen, ntrip; vector<int32_t> aps;   // per pin
  bool bad=false;
};
struct NetP {
  string names, pnames; int64_t nnets=0, npins=0, naps=0;
  vector<int64_t> pin_off, pname_off, ap_off; vector<double> slack; vector<int32_t> nparen, aps;
  bool bad=false;
};

static bool scan_ap(const char* s, const char* e, const char*& p, int32_t* v){
  const char* q=s;
  if(q>=e||*q!='(') return false; q++;
  auto skipws=[&](){ int k=0; while(q<e&&isws(*q)){q++;k++;} return k; };
  auto num=[&](int32_t& x){ if(q>=e||*q<'0'||*q>'9') return false; long long a=0; while(q<e&&*q>='0'&&*q<='9'){ a=a*10+(*q-'0'); q++; } x=(int32_t)a; return true; };
  skipws();
  if(!num(v[0])) return false;
  for(int k=1;k<=2;k++){
    int w=skipws();
    if(q<e&&*q==','){ q++; } else if(w==0) return false;
    skipws();
    if(!num(v[k])) return false;
  }
  skipws();
  if(q>=e||*q!=')') return false;
  p=q+1; return true;
}

static void parse_net_chunk(const char* b, const char* end, NPart& P){
  int depth=0; bool have=false; int32_t npin=0;
  const char* ls=b;
  while(ls<end){
    const char* le=(const char*)memchr(ls,'\n',end-ls); if(!le) le=end;
    const char* s=ls; const char* e=le;
    while(s<e&&isws(*s)) s++;
    while(e>s&&isws(e[-1])) e--;
    ls=le+1;
    if(s==e) continue;
    if(e-s==1&&*s=='('){ depth++; continue; }
    if(e-s==1&&*s==')'){
      depth--;
      if(depth==0&&have){ P.npins.push_back(npin); have=false; }
      continue;
    }
    if(depth==0){
      const char* t=s; while(t<e&&!isws(*t)) t++;
      P.names.append(s,t-s); P.names.push_back('\n'); have=true; npin=0;
    } else {
      // pin line:  name , slack , [ (l, x, y) ... ]     (fast_timer._PIN)
      const char* q=s; const char* a=q;
      while(q<e&&*q!=','&&!isws(*q)) q++;
      if(q==a) continue;
      const char* n1=a; const char* n2=q;
      while(q<e&&isws(*q)) q++;
      if(q>=e||*q!=',') continue;
      q++; while(q<e&&isws(*q)) q++;
      a=q; while(q<e&&*q!=','&&!isws(*q)) q++;
      if(q==a) continue;
      const char* s1=a; const char* s2=q;
      while(q<e&&isws(*q)) q++;
      if(q>=e||*q!=',') continue;
      q++; while(q<e&&isws(*q)) q++;
      if(q>=e||*q!='[') continue;
      q++;
      const char* r=(const char*)memchr(q,']',e-q); if(!r) continue;
      if(!have) continue;
      P.pnames.append(n1,n2-n1); P.pnames.push_back('\n');
      char tmp[80]; size_t L=std::min<size_t>(s2-s1,79); memcpy(tmp,s1,L); tmp[L]=0; char* ep=nullptr;
      double sv=strtod(tmp,&ep); if(ep==tmp||*ep!=0||(size_t)(s2-s1)>79) sv=__builtin_nan("");
      P.slack.push_back(sv);
      int32_t np=0, nt3=0;
      for(const char* p=q;p<r;){
        if(*p=='('){ np++; const char* np_; int32_t v[3]; if(scan_ap(p,r,np_,v)){ P.aps.push_back(v[0]); P.aps.push_back(v[1]); P.aps.push_back(v[2]); nt3++; p=np_; continue; } }
        p++;
      }
      P.nparen.push_back(np); P.ntrip.push_back(nt3);
      if(np!=nt3) P.bad=true;
      npin++;
    }
  }
}

extern "C" {

void* net_parse2(const char* path, int nt){
  PT pt("net");
  Buf B; if(!read_file(path,B,nt)) return nullptr;
  pt.mark("read");
  int64_t n=B.n; const char* b=B.p;
  int nch=std::max(1,nt*6); if(n<(1<<20)) nch=1;
  vector<int64_t> bd(nch+1); bd[0]=0; bd[nch]=n;
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=1;c<nch;c++){
    int64_t o=(n/nch)*(int64_t)c; const char* ls=b+o; const char* end=b+n;
    // advance to the line after the next line that is just ")"
    const char* nl=(const char*)memchr(ls,'\n',end-ls); if(!nl){ bd[c]=n; continue; }
    ls=nl+1; int64_t pos=n;
    while(ls<end){
      const char* le=(const char*)memchr(ls,'\n',end-ls); if(!le) le=end;
      const char* s=ls; const char* e=le; while(s<e&&isws(*s)) s++; while(e>s&&isws(e[-1])) e--;
      if(e-s==1&&*s==')'){ pos=(le<end)?(le+1-b):n; break; }
      ls=le+1;
    }
    bd[c]=pos;
  }
  for(int c=1;c<=nch;c++) if(bd[c]<bd[c-1]) bd[c]=bd[c-1];
  vector<NPart> parts(nch);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++) if(bd[c+1]>bd[c]) parse_net_chunk(b+bd[c],b+bd[c+1],parts[c]);
  pt.mark("chunks");
  NetP* R=new NetP();
  vector<int64_t> nn(nch+1,0), np(nch+1,0), na(nch+1,0), sn(nch+1,0), sp(nch+1,0);
  for(int c=0;c<nch;c++){ nn[c+1]=nn[c]+parts[c].npins.size(); np[c+1]=np[c]+parts[c].slack.size(); na[c+1]=na[c]+parts[c].aps.size()/3;
    sn[c+1]=sn[c]+parts[c].names.size(); sp[c+1]=sp[c]+parts[c].pnames.size(); if(parts[c].bad) R->bad=true; }
  R->nnets=nn[nch]; R->npins=np[nch]; R->naps=na[nch];
  R->names.resize(sn[nch]); R->pnames.resize(sp[nch]);
  R->pin_off.assign(R->nnets+1,0); R->pname_off.assign(R->npins+1,0); R->ap_off.assign(R->npins+1,0);
  R->slack.resize(R->npins); R->nparen.resize(R->npins); R->aps.resize(R->naps*3);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    NPart& P=parts[c];
    memcpy(&R->names[sn[c]],P.names.data(),P.names.size()); memcpy(&R->pnames[sp[c]],P.pnames.data(),P.pnames.size());
    memcpy(R->slack.data()+np[c],P.slack.data(),P.slack.size()*sizeof(double)); memcpy(R->nparen.data()+np[c],P.nparen.data(),P.nparen.size()*4);
    memcpy(R->aps.data()+na[c]*3,P.aps.data(),P.aps.size()*4);
    int64_t acc=np[c]; for(size_t i=0;i<P.npins.size();i++){ acc+=P.npins[i]; R->pin_off[nn[c]+i+1]=acc; }
    int64_t aa=na[c]; int64_t pp=sp[c];
    const string& pn=P.pnames; size_t k=0;
    for(size_t i=0;i<P.slack.size();i++){ size_t k2=pn.find('\n',k); pp+=(int64_t)(k2-k)+1; k=k2+1; R->pname_off[np[c]+i+1]=pp; aa+=P.ntrip[i]; R->ap_off[np[c]+i+1]=aa; }
  }
  pt.mark("merge");
  return R;
}
int64_t net2_nnets(void* h){ return ((NetP*)h)->nnets; }
int64_t net2_npins(void* h){ return ((NetP*)h)->npins; }
int64_t net2_naps(void* h){ return ((NetP*)h)->naps; }
int net2_bad(void* h){ return ((NetP*)h)->bad; }
const char* net2_names(void* h, int64_t* len){ *len=((NetP*)h)->names.size(); return ((NetP*)h)->names.data(); }
const char* net2_pnames(void* h, int64_t* len){ *len=((NetP*)h)->pnames.size(); return ((NetP*)h)->pnames.data(); }
const int64_t* net2_pin_off(void* h){ return ((NetP*)h)->pin_off.data(); }
const int64_t* net2_pname_off(void* h){ return ((NetP*)h)->pname_off.data(); }
const int64_t* net2_ap_off(void* h){ return ((NetP*)h)->ap_off.data(); }
const double* net2_slack(void* h){ return ((NetP*)h)->slack.data(); }
const int32_t* net2_nparen(void* h){ return ((NetP*)h)->nparen.data(); }
const int32_t* net2_aps(void* h){ return ((NetP*)h)->aps.data(); }
void net2_free(void* h){ delete (NetP*)h; }
// after the .net join nobody reads these: their memory goes now, the rest of the handle stays
void net2_release(void* h){ NetP* R=(NetP*)h; string().swap(R->pnames); vector<int64_t>().swap(R->pname_off); vector<double>().swap(R->slack); vector<int32_t>().swap(R->nparen); }

}  // extern C

// ============================================================ name tables (exact match, later duplicate wins)
static inline uint64_t hstr(const char* s, int64_t n, bool strip){
  uint64_t h=1469598103934665603ULL;
  if(strip){ for(int64_t i=0;i<n;i++){ unsigned char c=s[i]; if(c=='\\') continue; h=(h^c)*1099511628211ULL; } }
  else { for(int64_t i=0;i<n;i++){ h=(h^(unsigned char)s[i])*1099511628211ULL; } }
  h^=h>>29; h*=0xbf58476d1ce4e5b9ULL; h^=h>>32; return h;
}
struct Tab {
  const char* blob; const int64_t* off; int64_t n; vector<int32_t> slot; vector<uint64_t> hv; uint64_t mask;
  bool eq(int64_t e,const char* q,int64_t len,bool strip,string& tmp) const {
    const char* s=blob+off[e]; int64_t l=off[e+1]-off[e]-1;      // blob entries end with '\n'
    if(!strip) return l==len && memcmp(s,q,len)==0;
    if(l<0) return false;
    int64_t k=0; for(int64_t i=0;i<len;i++){ if(q[i]=='\\') continue; if(k>=l||s[k]!=q[i]) return false; k++; }
    return k==l;
  }
};
static Tab* tab_build(const char* blob, const int64_t* off, int64_t n){
  Tab* T=new Tab(); T->blob=blob; T->off=off; T->n=n;
  uint64_t sz=16; while(sz<(uint64_t)n*2+2) sz<<=1; T->mask=sz-1; T->slot.assign(sz,-1); T->hv.resize(n);
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for(int64_t i=0;i<n;i++) T->hv[i]=hstr(blob+off[i],off[i+1]-off[i]-1,false);
    int32_t* slot=T->slot.data(); const uint64_t* hv=T->hv.data(); const uint64_t mask=T->mask;
    // lock-free insertion: a slot is filled once and then only replaced by an equal name with a larger index, so the table maps each distinct
    // name to its LAST index (the serial loop's "later duplicate wins"), and every name stays reachable from its home slot.
    #pragma omp for schedule(dynamic,65536)
    for(int64_t i=0;i<n;i++){
      uint64_t h=hv[i]; uint64_t p=h&mask;
      while(true){
        int32_t e=__atomic_load_n(&slot[p],__ATOMIC_ACQUIRE);
        if(e<0){ int32_t x=-1; if(__atomic_compare_exchange_n(&slot[p],&x,(int32_t)i,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE)) break; e=x; }
        if(hv[e]==h && (off[e+1]-off[e])==(off[i+1]-off[i]) && memcmp(blob+off[e],blob+off[i],off[i+1]-off[i]-1)==0){
          while(e<(int32_t)i){ int32_t x=e; if(__atomic_compare_exchange_n(&slot[p],&x,(int32_t)i,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE)) break; e=x; }
          break;
        }
        p=(p+1)&mask;
      }
    }
  }
  return T;
}
static int64_t tab_find(const Tab* T,const char* q,int64_t len,bool strip){
  uint64_t h=hstr(q,len,strip); uint64_t p=h&T->mask; string tmp;
  while(true){
    int32_t e=T->slot[p]; if(e<0) return -1;
    if(T->hv[e]==h && T->eq(e,q,len,strip,tmp)) return e;
    p=(p+1)&T->mask;
  }
}

extern "C" {
void* nt_build(const char* blob, const int64_t* off, int64_t n){ return tab_build(blob,off,n); }
void nt_free(void* t){ delete (Tab*)t; }
int64_t nt_find(void* t, const char* q, int64_t len, int strip){ return tab_find((Tab*)t,q,len,strip!=0); }
void nt_lookup(void* t, const char* qb, const int64_t* qo, int64_t m, int strip, int nt, int64_t* out){
  Tab* T=(Tab*)t;
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(int64_t i=0;i<m;i++) out[i]=tab_find(T,qb+qo[i],qo[i+1]-qo[i]-1,strip!=0);
}
}

// ============================================================ DEF
struct Span { int64_t off; int32_t len; };
struct Conn { int32_t net; int32_t ilen; int32_t plen; int64_t ioff; int64_t poff; };
struct DefP {
  string inst_names, cell_names, net_names, pin_names, slot_names;
  vector<int64_t> inst_off, net_off, pin_off;
  vector<int32_t> inst_ct, pin_net, pin_inst, pin_slot, slot_cnt; vector<uint8_t> pin_pd;
  int64_t ninst=0, nnet=0, npin=0, ncell=0, nport=0; bool bad=false; string err;
  string power_text;   // records: 1 byte value '1' or '0', 8 bytes length, the text txt[st:pos] of power_pins_from_def, in its order
  Tab* pin_tab=nullptr; Tab* net_tab=nullptr; Tab* inst_tab=nullptr;
  ~DefP(){ delete pin_tab; delete net_tab; delete inst_tab; }
};
static int SNT=8; static int POWER_ON=1;
static const char* find_sec(const Buf& B, const char* name, int64_t& a){
  string pat=string("\n")+name+" "; const char* p=pmemmem(B.p,B.n,pat.data(),pat.size(),SNT);
  if(!p){ a=-1; return nullptr; } a=p-B.p; return p;
}
static int64_t sec_end(const Buf& B, const char* name, int64_t a){
  string pat=string("\nEND ")+name; const char* p=pmemmem(B.p+a,B.n-a,pat.data(),pat.size(),SNT);
  return p? (p-B.p) : B.n;
}
// all occurrences of kw in [a, b), increasing (== the find loop of power_pins_from_def: overlaps cannot occur for its two keywords)
static void find_all(const char* p, int64_t a, int64_t b, const char* kw, size_t kl, int nt, vector<int64_t>& out){
  const int64_t CH=8<<20; int64_t nc=(b-a)/CH+1; vector<vector<int64_t>> part(nc);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int64_t c=0;c<nc;c++){
    int64_t o=a+c*CH, e=std::min(b,o+CH+(int64_t)kl-1);
    const int64_t lim=std::min(b,o+CH);          // a match must START in [o, lim)
    while(o<lim){ int64_t len=e-o; if(len<(int64_t)kl) break; const char* q=(const char*)memmem(p+o,len,kw,kl); if(!q) break; int64_t v=q-p; if(v>=lim) break; part[c].push_back(v); o=v+1; }
  }
  for(auto& v: part) out.insert(out.end(),v.begin(),v.end());
}
static void power_text(const Buf& B, DefP* D, int nt){
  // txt.find("\nNETS "), txt.find("\nEND NETS", a); for each keyword, each position: txt[txt.rfind("\n- ", a, pos):pos]
  int64_t a,b; const char* p=pmemmem(B.p,B.n,"\nNETS ",6,nt); if(!p){ return; } a=p-B.p;
  p=pmemmem(B.p+a,B.n-a,"\nEND NETS",9,nt); b=p?(p-B.p):-1; if(b<0) return;
  const struct { const char* kw; char v; } KW[2]={{"+ USE POWER",'1'},{"+ USE GROUND",'0'}};
  for(int w=0;w<2;w++){
    vector<int64_t> pos; find_all(B.p,a,b,KW[w].kw,strlen(KW[w].kw),nt,pos);
    for(int64_t q: pos){
      int64_t st=-1; for(int64_t r=q-3;r>=a;r--) if(B.p[r]=='\n'&&B.p[r+1]=='-'&&B.p[r+2]==' '){ st=r; break; }
      if(st<0){ D->bad=true; D->err="power record without a start"; return; }
      char v=KW[w].v; int64_t L=q-st; D->power_text.push_back(v); D->power_text.append((const char*)&L,8); D->power_text.append(B.p+st,L);
    }
  }
}
static inline void key_append(string& out,const char* s,int64_t n){ for(int64_t i=0;i<n;i++) if(s[i]!='\\') out.push_back(s[i]); }

extern "C" {

void* def_parse(const char* path, int nt){
  PT pt("def"); SNT=nt;
  Buf B; if(!read_file(path,B,nt)) return nullptr;
  pt.mark("read");
  DefP* D=new DefP();
  int64_t ca,cb, pa,pb, na,nb;
  if(!find_sec(B,"COMPONENTS",ca)){ D->bad=true; D->err="no COMPONENTS"; return D; }
  cb=sec_end(B,"COMPONENTS",ca);
  if(!find_sec(B,"NETS",na)){ D->bad=true; D->err="no NETS"; return D; }
  nb=sec_end(B,"NETS",na);
  bool have_pins = find_sec(B,"PINS",pa)!=nullptr; pb= have_pins? sec_end(B,"PINS",pa):0;
  pt.mark("sections");
  // ---------------- COMPONENTS (line based, parallel)
  {
    int nch=std::max(1,nt*6); int64_t len=cb-ca; if(len<(1<<20)) nch=1;
    vector<int64_t> bd(nch+1); bd[0]=ca+1; bd[nch]=cb;
    for(int c=1;c<nch;c++){ int64_t o=ca+1+(len/nch)*c; const char* nl=(const char*)memchr(B.p+o,'\n',cb-o); bd[c]= nl? (nl+1-B.p) : cb; }
    for(int c=1;c<=nch;c++) if(bd[c]<bd[c-1]) bd[c]=bd[c-1];
    struct CE { int64_t no; int32_t nl; int32_t kl; int32_t tid; };
    struct CP { vector<CE> e; vector<string> tn; int64_t keylen=0; };
    vector<CP> parts(nch); bool bad=false;
    #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
    for(int c=0;c<nch;c++){
      CP& P=parts[c]; std::unordered_map<std::string_view,int32_t> lm;
      const char* ls=B.p+bd[c]; const char* end=B.p+bd[c+1];
      while(ls<end){
        const char* le=(const char*)memchr(ls,'\n',end-ls); if(!le) le=end;
        if(le>ls && ls[0]=='-' && ls+1<le && isws(ls[1])){
          const char* q=ls+1; while(q<le&&isws(*q)) q++;
          const char* a=q; while(q<le&&!isws(*q)) q++; const char* nme=q;
          while(q<le&&isws(*q)) q++;
          const char* t=q; while(q<le&&!isws(*q)) q++;
          if(a==nme||t==q){ bad=true; }
          else {
            std::string_view tv(t,q-t); auto it=lm.find(tv);
            int32_t tid; if(it==lm.end()){ tid=(int32_t)P.tn.size(); lm.emplace(tv,tid); P.tn.emplace_back(t,q-t); } else tid=it->second;
            int32_t kl=0; for(const char* x=a;x<nme;x++) if(*x!='\\') kl++;
            P.e.push_back({a-B.p,(int32_t)(nme-a),kl,tid}); P.keylen+=kl+1;
          }
        }
        ls=le+1;
      }
    }
    if(bad){ D->bad=true; D->err="components line"; return D; }
    // merge in chunk order: cell-type ids by first appearance over the file; names and offsets filled on threads
    std::unordered_map<string,int32_t> ctm; vector<string> ctn;
    vector<vector<int32_t>> tmap(nch); vector<int64_t> ibase(nch+1,0), obase(nch+1,0);
    for(int c=0;c<nch;c++){
      tmap[c].resize(parts[c].tn.size());
      for(size_t k=0;k<parts[c].tn.size();k++){ auto it=ctm.find(parts[c].tn[k]); if(it==ctm.end()){ it=ctm.emplace(parts[c].tn[k],(int32_t)ctn.size()).first; ctn.push_back(parts[c].tn[k]); } tmap[c][k]=it->second; }
      ibase[c+1]=ibase[c]+parts[c].e.size(); obase[c+1]=obase[c]+parts[c].keylen;
    }
    D->ninst=ibase[nch]; D->inst_names.resize(obase[nch]); D->inst_off.resize(D->ninst+1);
    vector<int32_t> rawct(D->ninst);
    #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
    for(int c=0;c<nch;c++){
      int64_t idx=ibase[c], o=obase[c]; char* w=&D->inst_names[0];
      for(const CE& e: parts[c].e){
        D->inst_off[idx]=o; rawct[idx]=tmap[c][e.tid]; idx++;
        const char* x=B.p+e.no; for(int32_t i=0;i<e.nl;i++) if(x[i]!='\\') w[o++]=x[i];
        w[o++]='\n';
      }
    }
    D->inst_off[D->ninst]=obase[nch];
    { vector<CP>().swap(parts); }
    vector<int32_t> ord(ctn.size()); for(size_t i=0;i<ord.size();i++) ord[i]=i;
    std::sort(ord.begin(),ord.end(),[&](int a,int b){ return ctn[a]<ctn[b]; });
    vector<int32_t> rank(ctn.size()); for(size_t i=0;i<ord.size();i++){ rank[ord[i]]=i; D->cell_names+=ctn[ord[i]]; D->cell_names.push_back('\n'); }
    D->ncell=ctn.size(); D->inst_ct.resize(D->ninst);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int64_t i=0;i<D->ninst;i++) D->inst_ct[i]=rank[rawct[i]];
    pt.mark("components");
    D->inst_tab=tab_build(D->inst_names.data(),D->inst_off.data(),D->ninst);
    // duplicate keyed instance names would change dict semantics -> bail out
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int64_t i=0;i<D->ninst;i++){ int64_t f=tab_find(D->inst_tab,D->inst_names.data()+D->inst_off[i],D->inst_off[i+1]-D->inst_off[i]-1,false); if(f!=i) bad=true; }
    if(bad){ D->bad=true; D->err="duplicate instance names"; return D; }
    pt.mark("inst_table");
  }
  // ---------------- PINS (ports): ^-\s+(\S+)(.*?);  DIRECTION\s+(\w+)
  std::unordered_map<string,uint8_t> ports;
  if(have_pins){
    const char* p=B.p+pa+1; const char* end=B.p+pb;
    while(p<end){
      const char* le=(const char*)memchr(p,'\n',end-p); if(!le) le=end;
      if(le>p && p[0]=='-' && p+1<le && isws(p[1])){
        const char* q=p+1; while(q<end&&isws(*q)) q++;
        const char* a=q; while(q<end&&!isws(*q)) q++;
        string key; key_append(key,a,q-a);
        const char* semi=(const char*)memchr(q,';',end-q); if(!semi) semi=end;
        uint8_t pd=1;   // default INPUT
        const char* d=(const char*)memmem(q,semi-q,"DIRECTION",9);
        if(d){ const char* r=d+9; if(r<semi&&isws(*r)){ while(r<semi&&isws(*r)) r++; const char* w=r; while(r<semi&&(isalnum((unsigned char)*r)||*r=='_')) r++;
            string dw(w,r-w); pd=(dw=="INPUT"||dw=="INOUT")?1:0; } }
        ports[key]=pd;
        p=semi+1; continue;
      }
      p=le+1;
    }
  }
  pt.mark("pins_section");
  // ---------------- NETS (parallel by net records)
  struct NC { vector<Span> nets; vector<uint8_t> power; vector<Conn> conns; };
  int nch=std::max(1,nt*6); int64_t len=nb-na; if(len<(1<<20)) nch=1;
  vector<int64_t> bd(nch+1); bd[0]=na+1; bd[nch]=nb;
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(int c=1;c<nch;c++){
    int64_t o=na+1+(len/nch)*c; int64_t pos=nb;
    while(o<nb){ const char* nl=(const char*)memchr(B.p+o,'\n',nb-o); if(!nl) break;
      int64_t s=nl+1-B.p; if(s+1<nb && B.p[s]=='-' && isws(B.p[s+1])){ pos=s; break; } o=s; }
    bd[c]=pos;
  }
  for(int c=1;c<=nch;c++) if(bd[c]<bd[c-1]) bd[c]=bd[c-1];
  vector<NC> parts(nch);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    NC& P=parts[c]; const char* ls=B.p+bd[c]; const char* end=B.p+bd[c+1];
    while(ls<end){
      const char* le=(const char*)memchr(ls,'\n',end-ls); if(!le) le=end;
      const char* q=ls;
      if(le>ls && ls[0]=='-' && ls+1<le && isws(ls[1])){
        q=ls+1; while(q<le&&isws(*q)) q++;
        const char* a=q; while(q<le&&!isws(*q)) q++;
        P.nets.push_back({a-B.p,(int32_t)(q-a)}); P.power.push_back(0);
      }
      while(q<le){
        char ch=*q;
        if(ch=='('){
          const char* p=q+1; while(p<le&&isws(*p)) p++;
          const char* a=p; while(p<le&&!isws(*p)) p++; const char* e1=p;
          bool ok=false; const char* a2=nullptr; const char* e2=nullptr; const char* endp=nullptr;
          if(p>a && p<le){
            while(p<le&&isws(*p)) p++;
            a2=p; while(p<le&&!isws(*p)) p++; e2=p;
            if(e2>a2){ const char* r=p; while(r<le&&isws(*r)) r++;
              if(r<le&&*r==')'){ ok=true; endp=r+1; }
              else if(e2[-1]==')'&&e2-1>a2){ e2--; ok=true; endp=p; } }
          }
          if(ok){
            int32_t il=(int32_t)(e1-a);
            if(!(il==1&&*a=='*') && !P.nets.empty())
              P.conns.push_back({(int32_t)(P.nets.size()-1),il,(int32_t)(e2-a2),(int64_t)(a-B.p),(int64_t)(a2-B.p)});
            q=endp; continue;
          }
          q++; continue;
        }
        if(ch=='+'){
          const char* p=q+1; while(p<le&&isws(*p)) p++;
          if(le-p>=3 && p[0]=='U'&&p[1]=='S'&&p[2]=='E' && p+3<le && isws(p[3])){
            p+=3; while(p<le&&isws(*p)) p++;
            const char* w=p; while(p<le&&(isalnum((unsigned char)*p)||*p=='_')) p++;
            if((p-w==5&&!memcmp(w,"POWER",5))||(p-w==6&&!memcmp(w,"GROUND",6))){ if(!P.nets.empty()) P.power.back()=1; }
            q=p; continue;
          }
        }
        q++;
      }
      ls=le+1;
    }
  }
  pt.mark("nets_chunks");
  vector<int64_t> nbase(nch+1,0); for(int c=0;c<nch;c++) nbase[c+1]=nbase[c]+parts[c].nets.size();
  D->nnet=nbase[nch];
  vector<uint8_t> power(D->nnet,0);
  { vector<int64_t> kl(nch+1,0);
    #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
    for(int c=0;c<nch;c++){ int64_t L=0; for(const Span& sp: parts[c].nets){ const char* x=B.p+sp.off; for(int32_t i=0;i<sp.len;i++) if(x[i]!='\\') L++; L++; } kl[c+1]=L; }
    for(int c=0;c<nch;c++) kl[c+1]+=kl[c];
    D->net_names.resize(kl[nch]); D->net_off.resize(D->nnet+1);
    #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
    for(int c=0;c<nch;c++){
      int64_t o=kl[c]; char* w=&D->net_names[0];
      for(size_t i=0;i<parts[c].nets.size();i++){
        D->net_off[nbase[c]+i]=o; const Span& sp=parts[c].nets[i]; const char* x=B.p+sp.off;
        for(int32_t j=0;j<sp.len;j++) if(x[j]!='\\') w[o++]=x[j];
        w[o++]='\n'; power[nbase[c]+i]=parts[c].power[i];
      }
    }
    D->net_off[D->nnet]=kl[nch];
  }
  pt.mark("net_names");
  // phase 2: instance lookup (parallel)
  vector<vector<int32_t>> ii(nch);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    NC& P=parts[c]; ii[c].resize(P.conns.size()); string tmp;
    for(size_t k=0;k<P.conns.size();k++){
      const Conn& cn=P.conns[k];
      if(power[nbase[c]+cn.net]){ ii[c][k]=-2; continue; }
      if(cn.ilen==3 && !memcmp(B.p+cn.ioff,"PIN",3)){ ii[c][k]=-1; continue; }
      int64_t f=tab_find(D->inst_tab,B.p+cn.ioff,cn.ilen,true);
      ii[c][k]= f<0? -2 : (int32_t)f;
    }
  }
  pt.mark("inst_lookup");
  // phase 3: interning / slot assignment. On threads per chunk: the pin keys and the (cell type, key) pairs in first-appearance order; merged in chunk order (the same ids and
  // slots as one pass over the file); the arrays filled on threads.
  struct IP { vector<string> names; vector<int32_t> lid; vector<std::pair<int32_t,int32_t>> pairs; int64_t kept=0; };
  vector<IP> ip(nch);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    IP& Q=ip[c]; std::unordered_map<string,int32_t> lm; string kn; const NC& P=parts[c];
    Q.lid.resize(P.conns.size()); vector<vector<uint8_t>> seen(D->ncell);
    for(size_t k=0;k<P.conns.size();k++){
      int32_t inst=ii[c][k]; if(inst==-2){ Q.lid[k]=-1; continue; }
      const Conn& cn=P.conns[k];
      kn.clear(); key_append(kn,B.p+cn.poff,cn.plen);
      auto it=lm.find(kn); int32_t id; if(it==lm.end()){ id=(int32_t)Q.names.size(); lm.emplace(kn,id); Q.names.push_back(kn); } else id=it->second;
      Q.lid[k]=id; Q.kept++;
      if(inst>=0){ int32_t ct=D->inst_ct[inst]; auto& sv=seen[ct]; if((int32_t)sv.size()<=id) sv.resize(Q.names.size()+64,0); if(!sv[id]){ sv[id]=1; Q.pairs.push_back({ct,id}); } }
    }
  }
  pt.mark("intern_chunks");
  std::unordered_map<string,int32_t> pkid; vector<string> pks; vector<vector<int32_t>> slot_of(D->ncell); vector<vector<int32_t>> slot_pk(D->ncell);
  vector<vector<int32_t>> gid(nch); vector<int64_t> kbase(nch+1,0);
  for(int c=0;c<nch;c++){
    IP& Q=ip[c]; gid[c].resize(Q.names.size());
    for(size_t k=0;k<Q.names.size();k++){ auto it=pkid.find(Q.names[k]); if(it==pkid.end()){ it=pkid.emplace(Q.names[k],(int32_t)pks.size()).first; pks.push_back(Q.names[k]); } gid[c][k]=it->second; }
    for(auto& pr: Q.pairs){ int32_t ct=pr.first, pid=gid[c][pr.second]; auto& so=slot_of[ct]; if((int32_t)so.size()<=pid) so.resize(pks.size()+64,-1); if(so[pid]<0){ so[pid]=slot_pk[ct].size(); slot_pk[ct].push_back(pid); } }
    kbase[c+1]=kbase[c]+Q.kept;
  }
  for(auto& so: slot_of) so.resize(pks.size(),-1);
  vector<uint8_t> pk_pd(pks.size(),1);
  for(size_t i=0;i<pks.size();i++){ auto pi=ports.find(pks[i]); pk_pd[i]=(pi==ports.end())?1:pi->second; }
  vector<int32_t> kept_pk(kbase[nch]);
  D->pin_net.resize(kbase[nch]); D->pin_inst.resize(kbase[nch]); D->pin_slot.resize(kbase[nch]); D->pin_pd.resize(kbase[nch]);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    const NC& P=parts[c]; const IP& Q=ip[c]; int64_t o=kbase[c];
    for(size_t k=0;k<P.conns.size();k++){
      int32_t inst=ii[c][k]; if(inst==-2) continue;
      int32_t pid=gid[c][Q.lid[k]];
      D->pin_net[o]=(int32_t)(nbase[c]+P.conns[k].net); D->pin_inst[o]=inst; kept_pk[o]=pid;
      if(inst<0){ D->pin_slot[o]=-1; D->pin_pd[o]=pk_pd[pid]; }
      else { D->pin_slot[o]=slot_of[D->inst_ct[inst]][pid]; D->pin_pd[o]=0; }
      o++;
    }
  }
  pt.mark("intern_fill");
  D->npin=D->pin_net.size();
  // pin names: inst + "/" + pk (ports: pk)
  D->pin_off.assign(D->npin+1,0);
  for(int64_t i=0;i<D->npin;i++){ int64_t l=pks[kept_pk[i]].size()+1; if(D->pin_inst[i]>=0) l+=(D->inst_off[D->pin_inst[i]+1]-D->inst_off[D->pin_inst[i]]-1)+1; D->pin_off[i+1]=D->pin_off[i]+l; }
  D->pin_names.resize(D->pin_off[D->npin]);
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(int64_t i=0;i<D->npin;i++){
    char* o=&D->pin_names[D->pin_off[i]];
    if(D->pin_inst[i]>=0){ int64_t il=D->inst_off[D->pin_inst[i]+1]-D->inst_off[D->pin_inst[i]]-1; memcpy(o,D->inst_names.data()+D->inst_off[D->pin_inst[i]],il); o+=il; *o++='/'; }
    const string& s=pks[kept_pk[i]]; memcpy(o,s.data(),s.size()); o+=s.size(); *o='\n';
  }
  pt.mark("pin_names");
  D->slot_cnt.resize(D->ncell);
  for(int64_t c=0;c<D->ncell;c++){ D->slot_cnt[c]=slot_pk[c].size(); for(int32_t p: slot_pk[c]){ D->slot_names+=pks[p]; D->slot_names.push_back('\n'); } }
  D->nport=ports.size();
  if(POWER_ON) power_text(B,D,nt);
  pt.mark("power_text");
  return D;
}
void fl8_set_power(int v){ POWER_ON=v; }
const char* def_power_text(void* h,int64_t* len){ DefP* D=(DefP*)h; *len=D->power_text.size(); return D->power_text.data(); }
int def_bad(void* h){ return ((DefP*)h)->bad; }
const char* def_err(void* h){ return ((DefP*)h)->err.c_str(); }
int64_t def_count(void* h,int which){ DefP* D=(DefP*)h; switch(which){case 0:return D->ninst;case 1:return D->nnet;case 2:return D->npin;case 3:return D->ncell;case 4:return D->nport;} return -1; }
const char* def_blob(void* h,int which,int64_t* len){ DefP* D=(DefP*)h; const string* s=nullptr;
  switch(which){case 0:s=&D->inst_names;break;case 1:s=&D->cell_names;break;case 2:s=&D->net_names;break;case 3:s=&D->pin_names;break;case 4:s=&D->slot_names;break;}
  *len=s->size(); return s->data(); }
const void* def_arr(void* h,int which){ DefP* D=(DefP*)h;
  switch(which){case 0:return D->inst_ct.data();case 1:return D->pin_net.data();case 2:return D->pin_inst.data();case 3:return D->pin_slot.data();
    case 4:return D->pin_pd.data();case 5:return D->slot_cnt.data();case 6:return D->inst_off.data();case 7:return D->net_off.data();case 8:return D->pin_off.data();} return nullptr; }
void* def_pin_table(void* h){ DefP* D=(DefP*)h; if(!D->pin_tab) D->pin_tab=tab_build(D->pin_names.data(),D->pin_off.data(),D->npin); return D->pin_tab; }
void* def_net_table(void* h){ DefP* D=(DefP*)h; if(!D->net_tab) D->net_tab=tab_build(D->net_names.data(),D->net_off.data(),D->nnet); return D->net_tab; }
void* def_inst_table(void* h){ return ((DefP*)h)->inst_tab; }
void def_free(void* h){ delete (DefP*)h; }

// ============================================================ activity file:  name density [input|clock]
struct ActP { string names; vector<int64_t> off; vector<double> val; };
void* act_parse(const char* path, int nt){
  Buf B; if(!read_file(path,B,nt)) return nullptr;
  int nch=std::max(1,nt*4); int64_t n=B.n; if(n<(1<<20)) nch=1;
  vector<int64_t> bd(nch+1); bd[0]=0; bd[nch]=n;
  for(int c=1;c<nch;c++){ int64_t o=(n/nch)*c; const char* nl=(const char*)memchr(B.p+o,'\n',n-o); bd[c]=nl?(nl+1-B.p):n; }
  for(int c=1;c<=nch;c++) if(bd[c]<bd[c-1]) bd[c]=bd[c-1];
  vector<ActP> parts(nch);
  #pragma omp parallel for num_threads(nt) schedule(dynamic,1)
  for(int c=0;c<nch;c++){
    ActP& P=parts[c]; const char* ls=B.p+bd[c]; const char* end=B.p+bd[c+1];
    while(ls<end){
      const char* le=(const char*)memchr(ls,'\n',end-ls); if(!le) le=end;
      const char* q=ls; while(q<le&&isws(*q)) q++;
      const char* a=q; while(q<le&&!isws(*q)) q++;
      const char* an=q; while(q<le&&isws(*q)) q++;
      if(an>a && q<le){
        const char* v=q; while(q<le&&!isws(*q)) q++; const char* ve=q;
        while(q<le&&isws(*q)) q++;
        bool skip=false;
        if(q<le){ const char* t=q; while(q<le&&!isws(*q)) q++; int64_t tl=q-t; if((tl==5&&!memcmp(t,"input",5))||(tl==5&&!memcmp(t,"clock",5))) skip=true; }
        if(!skip){
          char tmp[80]; size_t L=std::min<size_t>(ve-v,79); memcpy(tmp,v,L); tmp[L]=0; char* ep=nullptr; double d=strtod(tmp,&ep);
          if(!(ep==tmp||*ep!=0||(size_t)(ve-v)>79)){ P.off.push_back(P.names.size()); P.names.append(a,an-a); P.names.push_back('\n'); P.val.push_back(d); }
        }
      }
      ls=le+1;
    }
  }
  ActP* R=new ActP(); int64_t tot=0; for(auto& P: parts){ tot+=P.val.size(); }
  R->val.reserve(tot); R->off.reserve(tot+1);
  for(auto& P: parts){ int64_t base=R->names.size(); R->names+=P.names; for(auto o: P.off) R->off.push_back(base+o); R->val.insert(R->val.end(),P.val.begin(),P.val.end()); }
  R->off.push_back(R->names.size());
  return R;
}
int64_t act_n(void* h){ return ((ActP*)h)->val.size(); }
const char* act_names(void* h,int64_t* len){ *len=((ActP*)h)->names.size(); return ((ActP*)h)->names.data(); }
const int64_t* act_off(void* h){ return ((ActP*)h)->off.data(); }
const double* act_val(void* h){ return ((ActP*)h)->val.data(); }
void act_free(void* h){ delete (ActP*)h; }

}  // extern C
