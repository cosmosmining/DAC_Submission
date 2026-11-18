// build2 kernels: tokenizer, strtod batch, ISPD'25 .net parser, demand commit.
// clang++ -std=c++17 -O3 -fPIC -shared build2_kern.cc -o libbuild2_kern.dylib
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <tuple>
using std::vector; using std::string;

static inline bool isws(unsigned char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }

extern "C" {

// ---------- tokenizer (buf must be NUL terminated) ----------
int64_t tok_count(const char* b, int64_t n){
  int64_t c=0; bool in=false;
  for(int64_t i=0;i<n;i++){ bool w=isws(b[i]); if(!w && !in) c++; in=!w; }
  return c;
}
void tok_fill(const char* b, int64_t n, int64_t* st, int64_t* en){
  int64_t c=0; bool in=false;
  for(int64_t i=0;i<n;i++){ bool w=isws(b[i]);
    if(!w && !in) st[c]=i;
    if(w && in) en[c++]=i;
    in=!w; }
  if(in) en[c]=n;
}
void parse_dbl(const char* b, const int64_t* st, int64_t n, double* out){
  for(int64_t i=0;i<n;i++) out[i]=strtod(b+st[i], nullptr);
}

// ---------- .net parser ----------
struct NetP { string names; vector<int64_t> pin_off, ap_off; vector<int32_t> aps; int64_t nnets=0; };

static bool scan_ap(const char* s, const char* e, const char*& p, int32_t* v){
  // regex \(\s*(\d+)\s*[,\s]\s*(\d+)\s*[,\s]\s*(\d+)\s*\)  at s ; on success p=end
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

void* net_parse(const char* path){
  FILE* f=fopen(path,"rb"); if(!f) return nullptr;
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  string buf(n,'\0'); if(fread(&buf[0],1,n,f)!=(size_t)n){ fclose(f); return nullptr; } fclose(f);
  NetP* P=new NetP(); P->pin_off.push_back(0); P->ap_off.push_back(0);
  bool have=false; int depth=0; int64_t npins_cur=0;
  vector<int32_t> cur_aps; vector<int64_t> cur_ap_off; string cur_name;
  const char* b=buf.data(); const char* end=b+n; const char* ls=b;
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
      if(depth==0&&have){
        P->names.append(cur_name); P->names.push_back('\n');
        for(size_t k=0;k<cur_ap_off.size();k++) P->ap_off.push_back(P->ap_off.back()+ (int64_t)((k+1<cur_ap_off.size()?cur_ap_off[k+1]:(int64_t)cur_aps.size()/3)-cur_ap_off[k]));
        P->aps.insert(P->aps.end(),cur_aps.begin(),cur_aps.end());
        P->pin_off.push_back(P->pin_off.back()+(int64_t)cur_ap_off.size());
        P->nnets++; have=false; cur_aps.clear(); cur_ap_off.clear();
      }
      continue;
    }
    if(depth==0){
      const char* t=s; while(t<e&&!isws(*t)) t++;
      cur_name.assign(s,t); have=true; cur_aps.clear(); cur_ap_off.clear();
    } else {
      const char* br=(const char*)memchr(s,'[',e-s);
      const char* a=s; const char* z=e;
      if(br && br+1<e) a=br+1; else if(br) z=br;
      vector<int32_t> loc;
      const char* q=a;
      while(q<z){
        if(*q=='('){ int32_t v[3]; const char* p; if(scan_ap(q,z,p,v)){ loc.insert(loc.end(),v,v+3); q=p; continue; } }
        q++;
      }
      if(loc.empty()) continue;
      cur_ap_off.push_back((int64_t)cur_aps.size()/3);
      cur_aps.insert(cur_aps.end(),loc.begin(),loc.end());
    }
  }
  return P;
}
int64_t net_nnets(void* h){ return ((NetP*)h)->nnets; }
int64_t net_npins(void* h){ return ((NetP*)h)->pin_off.back(); }
int64_t net_naps(void* h){ return ((NetP*)h)->aps.size()/3; }
int64_t net_names_len(void* h){ return ((NetP*)h)->names.size(); }
const char* net_names(void* h){ return ((NetP*)h)->names.data(); }
const int64_t* net_pin_off(void* h){ return ((NetP*)h)->pin_off.data(); }
const int64_t* net_ap_off(void* h){ return ((NetP*)h)->ap_off.data(); }
const int32_t* net_aps(void* h){ return ((NetP*)h)->aps.data(); }
void net_free(void* h){ delete (NetP*)h; }

// ---------- demand commit: to_wires_vias + apply for every net with mask!=0 ----------
static inline int32_t fdiv(int32_t a,int32_t g){ int32_t q=a/g; if((a%g!=0)&&((a<0)!=(g<0))) q--; return q; }

int commit_nets(const int32_t* X0,const int32_t* Y0,const int32_t* Z0,const int32_t* X1,const int32_t* Y1,const int32_t* Z1,
                const int64_t* off,int64_t nn,const uint8_t* mask,int32_t g,
                const int32_t* dirs,int L,int X,int Y,double* d,int sign){
  typedef std::tuple<int,int,int,int> W;  // z, fixed, a, b
  vector<W> raw; vector<std::tuple<int,int,int>> vias; vector<W> mw;
  auto D=[&](int z,int x,int y)->double&{ return d[((int64_t)z*X+x)*Y+y]; };
  for(int64_t i=0;i<nn;i++){
    if(!mask[i]) continue;
    raw.clear(); vias.clear(); mw.clear();
    for(int64_t s=off[i];s<off[i+1];s++){
      int x1=fdiv(X0[s],g),y1=fdiv(Y0[s],g),z1=Z0[s],x2=fdiv(X1[s],g),y2=fdiv(Y1[s],g),z2=Z1[s];
      if(z1!=z2){ int lo=std::min(z1,z2),hi=std::max(z1,z2); for(int z=lo;z<hi;z++) vias.emplace_back(x1,y1,z); }
      else {
        if(z1<0||z1>=L) return -2;
        if(dirs[z1]==0&&y1==y2&&x1!=x2) raw.emplace_back(z1,y1,std::min(x1,x2),std::max(x1,x2));
        else if(dirs[z1]==1&&x1==x2&&y1!=y2) raw.emplace_back(z1,x1,std::min(y1,y2),std::max(y1,y2));
      }
    }
    std::sort(raw.begin(),raw.end());
    for(size_t k=0;k<raw.size();k++){
      if(!mw.empty()&&std::get<0>(mw.back())==std::get<0>(raw[k])&&std::get<1>(mw.back())==std::get<1>(raw[k])&&std::get<2>(raw[k])<=std::get<3>(mw.back()))
        std::get<3>(mw.back())=std::max(std::get<3>(mw.back()),std::get<3>(raw[k]));
      else mw.push_back(raw[k]);
    }
    for(auto& w:mw){
      int z=std::get<0>(w),f=std::get<1>(w),a=std::get<2>(w),b=std::get<3>(w);
      if(dirs[z]==0){ if(a<0||b>X||f<0||f>=Y) return -1; for(int x=a;x<b;x++) D(z,x,f)+=2*sign; }
      else { if(a<0||b>Y||f<0||f>=X) return -1; for(int y=a;y<b;y++) D(z,f,y)+=2*sign; }
    }
    std::sort(vias.begin(),vias.end()); vias.erase(std::unique(vias.begin(),vias.end()),vias.end());
    for(auto& v:vias){
      int x=std::get<0>(v),y=std::get<1>(v),z=std::get<2>(v);
      if(z==0) continue;
      int dz=dirs[z];
      // covered check on layer z: find group (z,f) and the interval with a<=pos
      int f=(dz==0)?y:x, pos=(dz==0)?x:y;
      auto it=std::upper_bound(mw.begin(),mw.end(),W(z,f,pos,INT32_MAX));
      bool cov=false;
      if(it!=mw.begin()){ --it; if(std::get<0>(*it)==z&&std::get<1>(*it)==f&&std::get<2>(*it)<=pos&&pos<=std::get<3>(*it)) cov=true; }
      if(cov) continue;
      if(x<0||y<0||x>=X||y>=Y) return -1;
      if(dz==0){
        if(0<x&&x<X-1){ D(z,x-1,y)+=sign; D(z,x,y)+=sign; }
        else if(x>0) D(z,x-1,y)+=2*sign;
        else if(x<X-1) D(z,x,y)+=2*sign;
      } else if(dz==1){
        if(0<y&&y<Y-1){ D(z,x,y-1)+=sign; D(z,x,y)+=sign; }
        else if(y>0) D(z,x,y-1)+=2*sign;
        else if(y<Y-1) D(z,x,y)+=2*sign;
      }
    }
  }
  return 0;
}
}
