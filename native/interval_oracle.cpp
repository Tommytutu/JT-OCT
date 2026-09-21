// Budgeted interval branch-and-bound for uniform-cost, STOP-enabled subtrees.
// No approximate result is reported as exact. Every returned upper has a tree.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <intrin.h>
using U=uint64_t;
using Bits=std::vector<U>;
constexpr double INF=std::numeric_limits<double>::infinity();
constexpr int UNUSED=(-2147483647-1);
struct Answer { double lo=INF,hi=INF; std::array<int,15> tree; Answer(){tree.fill(UNUSED);} };
struct Context {
 int F,W,K,minleaf; double weight,penalty;
 Bits zero,labels; std::vector<int> reps,masses;
 int budget,expanded,scanned; bool timed; std::chrono::steady_clock::time_point end;
 bool expired(){if(std::chrono::steady_clock::now()>=end)timed=true;return timed;}
 Answer basic(const Bits& s,int depth){
  Answer a;int n=0,best=-1,winner=0;
  for(U w:s)n+=(int)__popcnt64(w);
  if(n<minleaf)return a;
  for(int k=0;k<K;k++){int count=0;for(int w=0;w<W;w++)count+=(int)__popcnt64(s[w]&labels[k*W+w]);
   if(count>best){best=count;winner=k;}}
  a.hi=(n-best)*weight;a.tree[0]=-winner-1;
  double conflict=0;for(size_t i=0;i<reps.size();i++)if(s[reps[i]/64]&(U(1)<<(reps[i]%64)))conflict+=masses[i]*weight;
  a.lo=depth?std::min(a.hi,penalty+conflict):a.hi;return a;
 }
 static void graft(std::array<int,15>& dst,const std::array<int,15>& src,int to,int from=0){
  if(to>=15||from>=15||src[from]==UNUSED)return;dst[to]=src[from];
  if(src[from]>=0){graft(dst,src,2*to+1,2*from+1);graft(dst,src,2*to+2,2*from+2);}
 }
 void improve(Answer& a,int f,const Answer& l,const Answer& r){
  double value=penalty+l.hi+r.hi;if(value<a.hi){a.hi=value;a.tree.fill(UNUSED);a.tree[0]=f;graft(a.tree,l.tree,1);graft(a.tree,r.tree,2);}
 }
 struct Candidate {int f;double lo;};
 void split(const Bits& s,int f,Bits& l,Bits& r){for(int w=0;w<W;w++){l[w]=s[w]&zero[f*W+w];r[w]=s[w]^l[w];}}
 Answer search(const Bits& s,int depth,double cutoff){
  Answer a=basic(s,depth);
  if(!depth||a.lo>=a.hi||a.lo>=cutoff||expanded>=budget||expired())return a;
  ++expanded;const double stop=a.hi;std::vector<Candidate> candidates;
  Bits left(W),right(W);
  for(int f=0;f<F;f++){
   if(expired())return a; // Unscanned features retain the original safe LB.
   ++scanned;split(s,f,left,right);bool hasleft=false,hasright=false;
   for(int w=0;w<W;w++){hasleft|=left[w]!=0;hasright|=right[w]!=0;}
   if(!hasleft||!hasright)continue; // Constant splits are dominated by STOP contraction.
   Answer l=basic(left,depth-1),r=basic(right,depth-1);
   double lb=penalty+l.lo+r.lo;if(!std::isfinite(lb))continue;
   candidates.push_back({f,lb});improve(a,f,l,r);
  }
  auto refresh=[&](){double lower=stop;for(auto& c:candidates)lower=std::min(lower,c.lo);a.lo=std::min(a.hi,std::max(a.lo,lower));};
  refresh();if(a.lo>=std::min(a.hi,cutoff))return a;
  std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate& l,const Candidate& r){return l.lo<r.lo;});
  for(auto& c:candidates){
   if(expired()||expanded>=budget)break;
   if(c.lo>=std::min(a.hi,cutoff))continue;
   split(s,c.f,left,right);Answer r=basic(right,depth-1);
   Answer l=search(left,depth-1,std::min(a.hi,cutoff)-penalty-r.lo);
   r=search(right,depth-1,std::min(a.hi,cutoff)-penalty-l.lo);
   c.lo=std::max(c.lo,penalty+l.lo+r.lo);improve(a,c.f,l,r);refresh();
   if(a.lo>=std::min(a.hi,cutoff))break;
  }
  refresh();return a;
 }
};
extern "C" {
__declspec(dllexport) void* interval_create(int F,int W,int K,int ml,double weight,double penalty,
 const U* z,const U* labels,const int* reps,const int* masses,int nr){
 try {auto c=new Context();c->F=F;c->W=W;c->K=K;c->minleaf=ml;c->weight=weight;c->penalty=penalty;
  c->zero.assign(z,z+F*W);c->labels.assign(labels,labels+K*W);c->reps.assign(reps,reps+nr);c->masses.assign(masses,masses+nr);return c;
 }catch(...){return nullptr;}}
__declspec(dllexport) void interval_free(void* handle){delete static_cast<Context*>(handle);}
__declspec(dllexport) int interval_solve(void* handle,const U* mask,int depth,double cutoff,int budget,double seconds,
 double* bounds,int* tree,int* stats){
 try {auto& c=*static_cast<Context*>(handle);if(depth<0||depth>3||budget<0)return -1;
  c.budget=budget;c.expanded=0;c.scanned=0;c.timed=false;
  c.end=std::chrono::steady_clock::now()+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
  Answer a=c.search(Bits(mask,mask+c.W),depth,cutoff);bounds[0]=a.lo;bounds[1]=a.hi;
  std::copy(a.tree.begin(),a.tree.end(),tree);stats[0]=c.expanded;stats[1]=c.scanned;stats[2]=c.timed;
  return a.lo>=a.hi?1:0;
 }catch(...){return -1;}}
}
