// Native D3 structural experiments: explicit LP and JT-CG under
// base / signature compression / signature compression + endpoint elimination.
// All configuration enumeration, OpenMP cost evaluation, structural reduction,
// Gurobi model maintenance, pricing and certification execute in C++.
#include "gurobi_c++.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#define API extern "C" __declspec(dllexport)
static int pop64(uint64_t x){return int(__popcnt64(x));}
static int low64(uint64_t x){unsigned long p;_BitScanForward64(&p,x);return int(p);}
#else
#define API extern "C"
static int pop64(uint64_t x){return __builtin_popcountll(x);}
static int low64(uint64_t x){return __builtin_ctzll(x);}
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Clock=std::chrono::steady_clock;
using CostProvider=int (*)(int,const uint64_t*,const uint64_t*,const uint64_t*,double*,int32_t*,double);
constexpr double INF=std::numeric_limits<double>::infinity(),TOL=1e-9;
thread_local std::string last_error;
struct Timeout{};struct Capacity{};
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void number(std::ostream& os,double x){if(std::isfinite(x))os<<x;else os<<"null";}

struct Col {
    int q=0,s=0,h=-3,left=0,right=0; // h=-3 prefix-only, -1 private STOP, >=0 private split
    double base=INF,split=0.;
};
struct MCol {
    int q=0,s=0;double cost=INF;std::array<int,4> original{{-1,-1,-1,-1}};
};
struct Pick {double value=INF;std::array<int,4> original{{-1,-1,-1,-1}};};

struct Workspace {
    int n,F,K,W,A,P,min_leaf,early,no_repeat,threads;
    const uint8_t* X;const int32_t* y;const double* weight;
    std::vector<uint8_t> allowed;std::vector<double> extra;
    std::vector<uint64_t> ones,classes;double unit=0.;bool uniform=true,packed=false;
    CostProvider provider=nullptr;std::string output;
    Workspace(const uint8_t* x,const int32_t* yy,const double* w,const uint8_t* a,const double* e,
      int nn,int ff,int kk,int ml,int es,int nr,int th):n(nn),F(ff),K(kk),W((nn+63)/64),A(ff+kk),
      P(ff*(ff+kk)+kk),min_leaf(ml),early(es),no_repeat(nr),threads(std::max(1,th)),X(x),y(yy),weight(w),
      allowed(a,a+7*ff),extra(e,e+7*ff),unit(w[0]){for(int i=1;i<n;++i)if(w[i]!=unit){uniform=false;break;}}
    uint64_t valid(int w)const{return w==W-1&&n%64?(uint64_t(1)<<(n%64))-1:~uint64_t(0);}
    double mass(uint64_t bits,int w)const{if(uniform)return pop64(bits)*unit;double z=0.;while(bits){z+=weight[w*64+low64(bits)];bits&=bits-1;}return z;}
    int root(int s)const{return s<F*A?s/A:F+s-F*A;}
    void pack(Clock::time_point deadline){
        if(packed)return;ones.assign(size_t(F)*W,0);classes.assign(size_t(K)*W,0);
        // A word is owned by exactly one worker, so packing needs no atomics.
        // This was previously serial even when the cost kernel used OpenMP.
        std::atomic<bool> timeout(false);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int w=0;w<W;++w){
          if(timeout.load(std::memory_order_relaxed))continue;
          if((w&15)==0&&Clock::now()>=deadline){timeout.store(true,std::memory_order_relaxed);continue;}
          int first=64*w,last=std::min(n,first+64);
          for(int i=first;i<last;++i){auto bit=uint64_t(1)<<(i-first);
            classes[size_t(y[i])*W+w]|=bit;
            const uint8_t* xi=X+size_t(i)*F;
            for(int f=0;f<F;++f)if(xi[f])ones[size_t(f)*W+w]|=bit;
          }
        }
        if(timeout.load(std::memory_order_relaxed))throw Timeout{};
        packed=true;
    }
    void counts_into(const uint64_t* rows,std::vector<double>& c)const{
        std::fill(c.begin(),c.end(),0.);
        for(int w=0;w<W;++w)for(int k=0;k<K;++k)c[k]+=mass(rows[w]&classes[size_t(k)*W+w],w);
    }
    std::vector<double> counts(const std::vector<uint64_t>& rows)const{
        std::vector<double> c(K);counts_into(rows.data(),c);return c;
    }
    std::pair<int,double> best_leaf(const std::vector<double>& c)const{
        int label=int(std::max_element(c.begin(),c.end())-c.begin());double total=std::accumulate(c.begin(),c.end(),0.);
        return {label,std::max(0.,total-c[label])};
    }
    int row_count(const std::vector<uint64_t>& rows)const{int z=0;for(auto x:rows)z+=pop64(x);return z;}
    int row_count(const uint64_t* rows)const{int z=0;for(int w=0;w<W;++w)z+=pop64(rows[w]);return z;}
    std::vector<Col> enumerate(double penalty,int reduction,int cap,int cost_batch,Clock::time_point deadline,
                               double& prepare_seconds,double& sc_seconds,double& gpu_seconds,long long& raw_count){
        auto started=Clock::now();pack(deadline);
        std::vector<uint64_t> all(W);for(int w=0;w<W;++w)all[w]=valid(w);auto global=counts(all);
        std::vector<std::vector<double>> branch_loss(size_t(2)*F,std::vector<double>(K,INF));
        std::vector<int> branch_n(size_t(2)*F,0);
        #pragma omp parallel num_threads(threads)
        {
          std::vector<uint64_t> rows(W);std::vector<double> c(K);
          #pragma omp for schedule(static)
          for(int fa=0;fa<2*F;++fa){int f=fa/2,a=fa%2;
            for(int w=0;w<W;++w)rows[w]=a?ones[size_t(f)*W+w]:(valid(w)^ones[size_t(f)*W+w]);
            branch_n[fa]=row_count(rows.data());counts_into(rows.data(),c);double total=std::accumulate(c.begin(),c.end(),0.);
            for(int k=0;k<K;++k)branch_loss[fa][k]=std::max(0.,total-c[k]);
          }
        }
        const int states=4*F*F;std::vector<std::vector<Col>> local(states);std::vector<int> multiplicity(states,0);
        std::vector<uint64_t> routed(size_t(states)*W,0);
        std::atomic<bool> timeout(false);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int id=0;id<states;++id){
          if(timeout.load(std::memory_order_relaxed))continue;
          if((id&255)==0&&Clock::now()>=deadline){timeout.store(true,std::memory_order_relaxed);continue;}
          int q=id/(F*F),rem=id%(F*F),f=rem/F,g=rem%F,a=q/2,b=q%2;
          if(!allowed[f]||!allowed[(1+a)*F+g]||(no_repeat&&f==g))continue;
          for(int w=0;w<W;++w){auto rf=a?ones[size_t(f)*W+w]:(valid(w)^ones[size_t(f)*W+w]);auto rg=b?ones[size_t(g)*W+w]:(valid(w)^ones[size_t(g)*W+w]);routed[size_t(id)*W+w]=rf&rg;}
        }
        if(timeout.load(std::memory_order_relaxed))throw Timeout{};
        auto compute_cpu=[&](int id,std::vector<double>& c,std::vector<double>& lc_count,std::vector<double>& rc_count){
          if(Clock::now()>=deadline){timeout=true;return;}int q=id/(F*F),rem=id%(F*F),f=rem/F,g=rem%F;if(!allowed[f]||!allowed[(1+q/2)*F+g]||(no_repeat&&f==g))return;
          const uint64_t* rr=routed.data()+size_t(id)*W;int total=row_count(rr),positive=0;
          std::pair<int,double> stop;
          const bool binary_uniform=uniform&&K==2;
          if(binary_uniform){const uint64_t* positive_mask=classes.data()+W;
            for(int w=0;w<W;++w)positive+=pop64(rr[w]&positive_mask[w]);
            stop={positive>total-positive?1:0,std::min(positive,total-positive)*unit};
          }else{counts_into(rr,c);stop=best_leaf(c);}
          double prefix=.5*extra[(1+q/2)*F+g];double splitw=.75;
          Col best;bool have=false;int candidates=0;
          auto accept=[&](const Col& candidate){++candidates;if(!reduction){local[id].push_back(candidate);return;}
            double value=candidate.base+penalty*candidate.split,best_value=best.base+penalty*best.split;
            if(!have||std::pair<double,int>(value,candidate.h)<std::pair<double,int>(best_value,best.h)){best=candidate;have=true;}};
          if(early&&total>=min_leaf)accept({q,f*A+g,-1,stop.first,0,prefix+stop.second,splitw});
          for(int h=0;h<F;++h){if(!allowed[(3+q)*F+h]||(no_repeat&&(h==f||h==g)))continue;
            const uint64_t* hm=ones.data()+size_t(h)*W;
            if(binary_uniform){int nl=0,left_positive=0;const uint64_t* positive_mask=classes.data()+W;
              for(int w=0;w<W;++w){uint64_t left=rr[w]&~hm[w];nl+=pop64(left);left_positive+=pop64(left&positive_mask[w]);}
              int nr=total-nl;if(nl<min_leaf||nr<min_leaf)continue;int right_positive=positive-left_positive;
              int left_negative=nl-left_positive,right_negative=nr-right_positive;
              int left_label=left_positive>left_negative?1:0,right_label=right_positive>right_negative?1:0;
              double loss=double(std::min(left_positive,left_negative)+std::min(right_positive,right_negative))*unit;
              accept({q,f*A+g,h,left_label,right_label,prefix+extra[(3+q)*F+h]+loss,splitw+1.});continue;
            }
            std::fill(lc_count.begin(),lc_count.end(),0.);std::fill(rc_count.begin(),rc_count.end(),0.);int nl=0,nr=0;
            for(int w=0;w<W;++w){uint64_t left=rr[w]&~hm[w],right=rr[w]&hm[w];nl+=pop64(left);nr+=pop64(right);
              for(int k=0;k<K;++k){const uint64_t cm=classes[size_t(k)*W+w];lc_count[k]+=mass(left&cm,w);rc_count[k]+=mass(right&cm,w);}}
            if(nl<min_leaf||nr<min_leaf)continue;auto lc=best_leaf(lc_count),rc=best_leaf(rc_count);
            accept({q,f*A+g,h,lc.first,rc.first,prefix+extra[(3+q)*F+h]+lc.second+rc.second,splitw+1.});}
          multiplicity[id]=candidates;if(reduction&&have)local[id].push_back(best);
        };
        if(provider){
          const int batch=cost_batch;std::vector<double> loss(size_t(batch)*F);std::vector<int32_t> labels(size_t(batch)*F*2);
          std::vector<double> c(K);
          for(int begin=0;begin<states;begin+=batch){if(Clock::now()>=deadline)throw Timeout{};int count=std::min(batch,states-begin);auto tick=Clock::now();
            int code=provider(count,ones.data(),classes.data(),routed.data()+size_t(begin)*W,loss.data(),labels.data(),
                              std::chrono::duration<double>(deadline-Clock::now()).count());gpu_seconds+=elapsed(tick);
            if(code==-1)throw Timeout{};if(code!=1)throw std::runtime_error("CUDA structural cost callback failed");
            for(int j=0;j<count;++j){int id=begin+j,q=id/(F*F),rem=id%(F*F),f=rem/F,g=rem%F;if(!allowed[f]||!allowed[(1+q/2)*F+g]||(no_repeat&&f==g))continue;
              const uint64_t* rr=routed.data()+size_t(id)*W;int total=row_count(rr);counts_into(rr,c);auto stop=best_leaf(c);
              double prefix=.5*extra[(1+q/2)*F+g];double splitw=.75;
              Col best;bool have=false;int candidates=0;
              auto accept=[&](const Col& candidate){++candidates;if(!reduction){local[id].push_back(candidate);return;}
                double value=candidate.base+penalty*candidate.split,best_value=best.base+penalty*best.split;
                if(!have||std::pair<double,int>(value,candidate.h)<std::pair<double,int>(best_value,best.h)){best=candidate;have=true;}};
              if(early&&total>=min_leaf)accept({q,f*A+g,-1,stop.first,0,prefix+stop.second,splitw});
              for(int h=0;h<F;++h)if(std::isfinite(loss[size_t(j)*F+h])&&allowed[(3+q)*F+h]&&!(no_repeat&&(h==f||h==g)))
                accept({q,f*A+g,h,labels[(size_t(j)*F+h)*2],labels[(size_t(j)*F+h)*2+1],
                  prefix+extra[(3+q)*F+h]+loss[size_t(j)*F+h],splitw+1.});
              multiplicity[id]=candidates;if(reduction&&have)local[id].push_back(best);}
          }
        }else{
          #pragma omp parallel num_threads(threads)
          {
            std::vector<double> c(K),lc_count(K),rc_count(K);
            #pragma omp for schedule(dynamic)
            for(int id=0;id<states;++id)compute_cpu(id,c,lc_count,rc_count);
          }
          if(timeout.load(std::memory_order_relaxed))throw Timeout{};
        }
        std::vector<Col> raw;long long stored_estimate=reduction?4LL*(1LL*F*A+K):4LL*F*F*std::max(1,F);
        raw.reserve(size_t(std::min<long long>(cap,stored_estimate)));
        for(int q=0;q<4;++q){
          if(early&&n>=min_leaf){double total=std::accumulate(global.begin(),global.end(),0.);for(int k=0;k<K;++k){
            raw.push_back({q,F*A+k,-3,k,0,.25*std::max(0.,total-global[k]),0.});++raw_count;}}
          int a=q/2;for(int f=0;f<F;++f)if(allowed[f]){
            double rootbase=.25*extra[f];
            if(early&&branch_n[2*f+a]>=min_leaf)for(int k=0;k<K;++k){
              raw.push_back({q,f*A+F+k,-3,k,0,rootbase+.5*branch_loss[2*f+a][k],.25});++raw_count;}
            for(int g=0;g<F;++g){int id=(q*F+f)*F+g;raw_count+=multiplicity[id];for(auto c:local[id]){c.base+=rootbase;raw.push_back(c);}}
          }
          if((int)raw.size()>cap||(reduction==0&&raw_count>cap))throw Capacity{};
        }
        prepare_seconds=elapsed(started);if(reduction==0)return raw;
        auto st=Clock::now();std::vector<int> best(size_t(4)*P,-1);
        for(int i=0;i<(int)raw.size();++i){int key=raw[i].q*P+raw[i].s;int j=best[key];double ci=raw[i].base+penalty*raw[i].split;
          if(j<0||std::pair<double,int>(ci,i)<std::pair<double,int>(raw[j].base+penalty*raw[j].split,j))best[key]=i;}
        std::vector<Col> compressed;for(int id:best)if(id>=0)compressed.push_back(raw[id]);sc_seconds=elapsed(st);return compressed;
    }
    std::vector<MCol> master_columns(const std::vector<Col>& c,double penalty,int reduction,double& ee_seconds)const{
        std::vector<MCol> out;if(reduction<2){out.reserve(c.size());for(int i=0;i<(int)c.size();++i){MCol m;m.q=c[i].q;m.s=c[i].s;m.cost=c[i].base+penalty*c[i].split;m.original[c[i].q]=i;out.push_back(m);}return out;}
        auto st=Clock::now();std::vector<int> best(size_t(4)*P,-1);for(int i=0;i<(int)c.size();++i)best[c[i].q*P+c[i].s]=i;
        for(int side=0;side<2;++side)for(int s=0;s<P;++s){int i=best[(2*side)*P+s],j=best[(2*side+1)*P+s];if(i<0||j<0)continue;
          MCol m;m.q=side;m.s=s;m.cost=c[i].base+penalty*c[i].split+c[j].base+penalty*c[j].split;
          m.original[2*side]=i;m.original[2*side+1]=j;out.push_back(m);}
        ee_seconds=elapsed(st);return out;
    }
    Pick select(const std::vector<MCol>& mc,const std::vector<unsigned char>* active=nullptr)const{
        int M=mc.empty()?0:(1+std::max_element(mc.begin(),mc.end(),[](auto&a,auto&b){return a.q<b.q;})->q);
        std::vector<double> best(size_t(M)*P,INF);std::vector<int> id(size_t(M)*P,-1);
        for(int j=0;j<(int)mc.size();++j){if(active&&!(*active)[j])continue;auto&m=mc[j];int k=m.q*P+m.s;if(m.cost<best[k]){best[k]=m.cost;id[k]=j;}}
        Pick pick;
        if(M==2){std::vector<double> lv(A,INF),rv(A,INF);std::vector<int> li(A,-1),ri(A,-1);
          for(int s=0;s<P;++s){int r=root(s);if(id[s]>=0&&best[s]<lv[r]){lv[r]=best[s];li[r]=id[s];}if(id[P+s]>=0&&best[P+s]<rv[r]){rv[r]=best[P+s];ri[r]=id[P+s];}}
          for(int r=0;r<A;++r)if(li[r]>=0&&ri[r]>=0){double v=lv[r]+rv[r];if(v<pick.value){pick.value=v;pick.original=mc[li[r]].original;for(int q=0;q<4;++q)if(mc[ri[r]].original[q]>=0)pick.original[q]=mc[ri[r]].original[q];}}}
        else if(M==4){std::vector<double> left(P,INF),right(P,INF);std::vector<std::array<int,2>> li(P),ri(P);
          for(int s=0;s<P;++s){if(id[s]>=0&&id[P+s]>=0){left[s]=best[s]+best[P+s];li[s]={id[s],id[P+s]};}
            if(id[2*P+s]>=0&&id[3*P+s]>=0){right[s]=best[2*P+s]+best[3*P+s];ri[s]={id[2*P+s],id[3*P+s]};}}
          std::vector<double> lv(A,INF),rv(A,INF);std::vector<int> ls(A,-1),rs(A,-1);for(int s=0;s<P;++s){int r=root(s);if(left[s]<lv[r]){lv[r]=left[s];ls[r]=s;}if(right[s]<rv[r]){rv[r]=right[s];rs[r]=s;}}
          for(int r=0;r<A;++r)if(ls[r]>=0&&rs[r]>=0){double v=lv[r]+rv[r];int l=ls[r],s=rs[r];if(v<pick.value){pick.value=v;pick.original={{mc[li[l][0]].original[0],mc[li[l][1]].original[1],mc[ri[s][0]].original[2],mc[ri[s][1]].original[3]}};}}}
        return pick;
    }
    void leaf_json(std::ostream& os,int k)const{os<<"{\"label_index\":"<<k<<"}";}
    void private_json(std::ostream& os,const Col& c)const{if(c.h<0){leaf_json(os,c.left);return;}os<<"{\"feature\":"<<c.h<<",\"left\":";leaf_json(os,c.left);os<<",\"right\":";leaf_json(os,c.right);os<<"}";}
    std::string tree_json(const Pick& pick,const std::vector<Col>& c)const{
        if(!std::isfinite(pick.value))return "null";const Col& c0=c[pick.original[0]];int r=root(c0.s);std::ostringstream os;
        if(r>=F){leaf_json(os,r-F);return os.str();}os<<"{\"feature\":"<<r<<",\"left\":";
        for(int side=0;side<2;++side){const Col&a=c[pick.original[2*side]],&b=c[pick.original[2*side+1]];int action=a.s% A;
          if(action>=F)leaf_json(os,action-F);else{os<<"{\"feature\":"<<action<<",\"left\":";private_json(os,a);os<<",\"right\":";private_json(os,b);os<<"}";}
          if(side==0)os<<",\"right\":";}
        os<<"}";return os.str();
    }
    const char* solve(double penalty,double seconds,int reduction,int coordinator,int cap,int active_cap,int pricing_batch,int method,int cost_batch){
        auto all_start=Clock::now(),deadline=all_start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        double prep=0,sc=0,ee=0,gpu=0,rmp_setup=0,insert=0,rmp_time=0,pricing=0;int iterations=0,peak_cols=0,peak_nz=0;
        long long raw_count=0;auto cols=enumerate(penalty,reduction,cap,cost_batch,deadline,prep,sc,gpu,raw_count);auto mc=master_columns(cols,penalty,reduction,ee);if((int)mc.size()>cap)throw Capacity{};
        int M=reduction==2?2:4,R=reduction==2?A:2*P+A;auto st=Clock::now();GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();GRBModel model(env);
        model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260916);model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        std::vector<GRBConstr> norm,re(R);if(M==4)for(int q=0;q<4;++q)norm.push_back(model.addConstr(GRBLinExpr()==1.));else norm.push_back(model.addConstr(GRBLinExpr()==1.));
        for(int e=0;e<R;++e)re[e]=model.addConstr(GRBLinExpr()==0.);model.update();rmp_setup=elapsed(st);
        std::vector<GRBVar> var(mc.size());std::vector<unsigned char> active(mc.size(),0);std::vector<int> pending;
        auto edge=[&](int q,int s,int slot){if(M==2)return root(s);if(q==0)return s;if(q==1)return slot==0?s:P+root(s);if(q==2)return slot==0?P+root(s):P+A+s;return P+A+s;};
        auto degree=[&](int q){return M==2?1:(q==0||q==3?1:2);};auto sign=[&](int q,int slot){if(M==2)return q==0?1.:-1.;return q==0?1.:(q==3?-1.:(slot==0?-1.:1.));};
        Pick incumbent=select(mc),restricted;double lb=0.;
        if(coordinator==0){pending.resize(mc.size());std::iota(pending.begin(),pending.end(),0);}
        else{
          if(!early)throw std::runtime_error("CG structural experiment requires a common STOP-root seed");
          std::vector<double> global(K,0.);for(int i=0;i<n;++i)global[y[i]]+=weight[i];int k=int(std::max_element(global.begin(),global.end())-global.begin()),s=F*A+k;
          for(int q=0;q<M;++q){int found=-1;for(int j=0;j<(int)mc.size();++j)if(mc[j].q==q&&mc[j].s==s&&(found<0||mc[j].cost<mc[found].cost))found=j;if(found<0)throw std::runtime_error("Missing common STOP seed");pending.push_back(found);}
        }
        int status=GRB_OPTIMAL;
        while(true){if(Clock::now()>=deadline)throw Timeout{};if(std::count(active.begin(),active.end(),1)+(int)pending.size()>active_cap)throw Capacity{};st=Clock::now();
          for(int j:pending)if(!active[j]){auto&m=mc[j];GRBColumn column;if(M==4)column.addTerm(1.,norm[m.q]);else if(m.q==0)column.addTerm(1.,norm[0]);
            for(int slot=0;slot<degree(m.q);++slot)column.addTerm(sign(m.q,slot),re[edge(m.q,m.s,slot)]);var[j]=model.addVar(0.,GRB_INFINITY,m.cost,GRB_CONTINUOUS,column);active[j]=1;}
          model.update();insert+=elapsed(st);peak_cols=std::max(peak_cols,int(model.get(GRB_IntAttr_NumVars)));peak_nz=std::max(peak_nz,int(model.get(GRB_IntAttr_NumNZs)));
          model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,std::chrono::duration<double>(deadline-Clock::now()).count()));st=Clock::now();model.optimize();rmp_time+=elapsed(st);++iterations;status=model.get(GRB_IntAttr_Status);
          if(status!=GRB_OPTIMAL)break;restricted=select(mc,&active);if(!std::isfinite(restricted.value)||std::abs(restricted.value-model.get(GRB_DoubleAttr_ObjVal))>1e-7)throw std::runtime_error("Native structural RMP recovery failed");
          if(restricted.value<incumbent.value)incumbent=restricted;if(coordinator==0){lb=model.get(GRB_DoubleAttr_ObjVal);break;}
          st=Clock::now();std::vector<double> alpha(M,0.),pi(R);if(M==4)for(int q=0;q<4;++q)alpha[q]=norm[q].get(GRB_DoubleAttr_Pi);else alpha[0]=norm[0].get(GRB_DoubleAttr_Pi);
          for(int e=0;e<R;++e)pi[e]=re[e].get(GRB_DoubleAttr_Pi);std::vector<std::pair<double,int>> improving;std::vector<double> minima(M,INF);
          std::vector<int> signature_best(reduction==0?size_t(M)*P:0,-1);std::vector<double> signature_rc(reduction==0?size_t(M)*P:0,INF);
          for(int j=0;j<(int)mc.size();++j){auto&m=mc[j];double rc=m.cost-(M==4?alpha[m.q]:(m.q==0?alpha[0]:0.));for(int slot=0;slot<degree(m.q);++slot)rc-=sign(m.q,slot)*pi[edge(m.q,m.s,slot)];minima[m.q]=std::min(minima[m.q],rc);
            if(!active[j]&&rc<-1e-9){if(reduction==0){int k=m.q*P+m.s;if(rc<signature_rc[k]){signature_rc[k]=rc;signature_best[k]=j;}}else improving.push_back({rc,j});}}
          // A competent base pricer never inserts dominated private completions
          // with identical master coefficients.  It minimizes them on demand;
          // SC materializes exactly this quotient before pricing.
          if(reduction==0)for(size_t k=0;k<signature_best.size();++k)if(signature_best[k]>=0)improving.push_back({signature_rc[k],signature_best[k]});
          lb=(M==4?std::accumulate(alpha.begin(),alpha.end(),0.):alpha[0]);for(double v:minima)lb+=std::min(0.,v);lb=std::min(lb,incumbent.value);
          std::sort(improving.begin(),improving.end());if(pricing_batch>0&&(int)improving.size()>pricing_batch)improving.resize(pricing_batch);pending.clear();for(auto x:improving)pending.push_back(x.second);pricing+=elapsed(st);
          if(incumbent.value-lb<=1e-7)break;if(pending.empty())throw std::runtime_error("Native structural pricing stalled");
        }
        std::string state=status==GRB_OPTIMAL&&incumbent.value-lb<=1e-7?"OPT":(status==GRB_TIME_LIMIT?"TIME":"ERROR");
        std::ostringstream os;os<<std::setprecision(17)<<"{\"status\":\""<<state<<"\",\"LB\":";number(os,lb);os<<",\"UB\":";number(os,incumbent.value);
        os<<",\"tree\":"<<tree_json(incumbent,cols)<<",\"engine\":\"cpp_gurobi_structural_v2\",\"backend\":\""<<(provider?"cuda":"cpp_openmp")<<"\""
          <<",\"raw_columns\":"<<raw_count<<",\"prepared_columns\":"<<cols.size()<<",\"master_domain_columns\":"<<mc.size()
          <<",\"remaining_clusters\":"<<M<<",\"submitted_rows\":"<<model.get(GRB_IntAttr_NumConstrs)<<",\"peak_active_columns\":"<<peak_cols
          <<",\"peak_rmp_nonzeros\":"<<peak_nz<<",\"iterations\":"<<iterations<<",\"cost_preparation_seconds\":"<<prep
          <<",\"sc_seconds\":"<<sc<<",\"ee_seconds\":"<<ee<<",\"gpu_callback_seconds\":"<<gpu<<",\"rmp_setup_seconds\":"<<rmp_setup
          <<",\"rmp_insertion_seconds\":"<<insert<<",\"rmp_seconds\":"<<rmp_time<<",\"pricing_seconds\":"<<pricing
          <<",\"native_seconds\":"<<elapsed(all_start)<<"}";output=os.str();return output.c_str();
    }
};
}

API const char* d3struct_error(){return last_error.c_str();}
API void* d3struct_create(const uint8_t* X,const int32_t* y,const double* weights,const uint8_t* allowed,const double* extras,
 int n,int F,int K,int min_leaf,int early,int no_repeat,int threads){try{last_error.clear();if(!X||!y||!weights||!allowed||!extras||n<1||F<1||K<1)throw std::invalid_argument("Invalid structural workspace");return new Workspace(X,y,weights,allowed,extras,n,F,K,min_leaf,early,no_repeat,threads);}catch(const std::exception&e){last_error=e.what();return nullptr;}}
API void d3struct_cost_provider(void* handle,CostProvider provider){if(handle)static_cast<Workspace*>(handle)->provider=provider;}
API const char* d3struct_solve(void* handle,double penalty,double seconds,int reduction,int coordinator,int cap,int active_cap,int pricing_batch,int method,int cost_batch){
 try{last_error.clear();if(!handle||penalty<0||seconds<0||reduction<0||reduction>2||coordinator<0||coordinator>1||cap<1||active_cap<1||cost_batch<1)throw std::invalid_argument("Invalid structural solve options");return static_cast<Workspace*>(handle)->solve(penalty,seconds,reduction,coordinator,cap,active_cap,pricing_batch,method,cost_batch);}
 catch(const Timeout&){static_cast<Workspace*>(handle)->output="{\"status\":\"TIME\",\"LB\":0,\"UB\":null,\"tree\":null,\"engine\":\"cpp_gurobi_structural_v2\"}";return static_cast<Workspace*>(handle)->output.c_str();}
 catch(const Capacity&){static_cast<Workspace*>(handle)->output="{\"status\":\"RESOURCE\",\"LB\":0,\"UB\":null,\"tree\":null,\"engine\":\"cpp_gurobi_structural_v2\"}";return static_cast<Workspace*>(handle)->output.c_str();}
 catch(const GRBException&e){last_error=e.getMessage();return nullptr;}catch(const std::exception&e){last_error=e.what();return nullptr;}}
API void d3struct_destroy(void* handle){delete static_cast<Workspace*>(handle);}
