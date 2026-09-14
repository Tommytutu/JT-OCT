// D3 JT-CG with four path blocks and complete [2,1,2] separators.
// Transfers the validated D2 numeric-table/incremental-Gurobi design to D3.
// Stump costs are penalty-independent; only O(F^2) messages survive min_h.
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
#include <vector>
#include <deque>
#include <unordered_map>
#ifdef _MSC_VER
#include <intrin.h>
#define API extern "C" __declspec(dllexport)
static int pop(uint64_t x) { return (int)__popcnt64(x); }
static int lowbit(uint64_t x) { unsigned long p; _BitScanForward64(&p,x); return (int)p; }
#else
#define API extern "C"
static int pop(uint64_t x) { return __builtin_popcountll(x); }
static int lowbit(uint64_t x) { return __builtin_ctzll(x); }
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Clock=std::chrono::steady_clock;
using Reporter=void (*)(const char*);
using CostProvider=int (*)(int,const uint64_t*,const uint64_t*,const uint64_t*,double*,int32_t*,double);
using DirectCostProvider=int (*)(int,const uint64_t*,const uint64_t*,const uint8_t*,const double*,const int32_t*,double*,int32_t*,double);
constexpr double INF=std::numeric_limits<double>::infinity(),TOL=1e-9;
thread_local std::string last_error;
struct Timeout {};
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void number(std::ostream& os,double x) { if(std::isfinite(x))os<<x;else os<<"null"; }
struct Tail { double stop=INF,split=INF;int label=0,h=-1,left=0,right=0; };
struct Choice { double cost=INF;int h=-1,left=0,right=0; };
struct Selection { double value=INF;std::array<int,2> prefix{-1,-1}; };
struct Event {
    double seconds,lb,ub,alpha,residual,active_rc;
    std::array<double,4> minima;
    int iteration,columns,rows,new_columns;
};
struct LazyCosts;
struct Workspace {
    int n,F,K,W,A,P,min_leaf,early,no_repeat,threads;
    const uint8_t* X;const int32_t* y;const double* weights;
    std::vector<uint8_t> allowed,ready;
    std::vector<double> extras,counts,branch_loss;
    std::vector<int> branch_size;
    std::vector<uint64_t> ones,classes;
    std::vector<Tail> tails;
    bool packed=false,counted=false,branches=false,uniform=true;
    double unit;
    std::unique_ptr<GRBEnv> env;
    std::string output;
    int cost_mode=0,cost_batch=128,cache_mb=128;
    CostProvider cost_provider=nullptr;
    DirectCostProvider direct_cost_provider=nullptr;
    std::shared_ptr<LazyCosts> lazy;
    Workspace(const uint8_t* x,const int32_t* labels,const double* w,const uint8_t* a,
              const double* c,int nn,int ff,int kk,int ml,int es,int nr,int th):
        n(nn),F(ff),K(kk),W((nn+63)/64),A(ff+kk),P(ff*(ff+kk)+kk),
        min_leaf(ml),early(es),no_repeat(nr),threads(th),X(x),y(labels),weights(w),
        allowed(a,a+7*ff),ready(size_t(ff)*ff,0),extras(c,c+7*ff),
        tails(size_t(4)*ff*ff),unit(w[0]) {
        for(int i=1;i<n;++i)if(w[i]!=unit){uniform=false;break;}
    }
    uint64_t valid(int w) const {return w==W-1&&n%64?(uint64_t(1)<<(n%64))-1:~uint64_t(0);}
    uint64_t branch(int f,int b,int w) const {auto v=ones[size_t(f)*W+w];return b?v:(valid(w)^v);}
    double mass(uint64_t bits,int w) const {
        if(uniform)return pop(bits)*unit;
        double value=0;while(bits){value+=weights[w*64+lowbit(bits)];bits&=bits-1;}return value;
    }
    static std::pair<int,double> leaf(const std::vector<double>& c) {
        int k=(int)(std::max_element(c.begin(),c.end())-c.begin());
        return {k,std::max(0.,std::accumulate(c.begin(),c.end(),0.)-c[k])};
    }
    int root(int s) const {return s<F*A?s/A:F+s-F*A;}
    int edge_count() const {return 2*P+A;}
    // Edge 0: complete left-side prefix; edge 1: root; edge 2: right-side prefix.
    int edge(int q,int s,int slot) const {
        if(q==0)return s;
        if(q==1)return slot==0?s:P+root(s);
        if(q==2)return slot==0?P+root(s):P+A+s;
        return P+A+s;
    }
    int degree(int q) const {return q==0||q==3?1:2;}
    double sign(int q,int slot) const {return q==0?1.:(q==3?-1.:(slot==0?-1.:1.));}
    void count_global() {
        if(counted)return;counts.assign(K,0.);
        for(int i=0;i<n;++i)counts[y[i]]+=weights[i];counted=true;
    }
    void pack(Clock::time_point deadline) {
        if(packed)return;
        ones.assign(size_t(F)*W,0);classes.assign(size_t(K)*W,0);
        for(int i=0;i<n;++i){
            if(i%1024==0&&Clock::now()>=deadline)throw Timeout{};
            auto bit=uint64_t(1)<<(i%64);int w=i/64;classes[size_t(y[i])*W+w]|=bit;
            for(int f=0;f<F;++f)if(X[size_t(i)*F+f])ones[size_t(f)*W+w]|=bit;
        }packed=true;
    }
    void prepare(Clock::time_point deadline) {
        pack(deadline);
        if(!branches){
            branch_size.assign(2*F,0);branch_loss.assign(size_t(2)*F*K,INF);
            for(int f=0;f<F;++f){
                if(Clock::now()>=deadline)throw Timeout{};
                for(int a=0;a<2;++a){
                    std::vector<double> c(K,0.);int total=0;
                    for(int w=0;w<W;++w){auto bits=branch(f,a,w);total+=pop(bits);
                        for(int k=0;k<K;++k)c[k]+=mass(bits&classes[size_t(k)*W+w],w);}
                    branch_size[2*f+a]=total;
                    double sum=std::accumulate(c.begin(),c.end(),0.);
                    for(int k=0;k<K;++k)branch_loss[(size_t(2*f+a)*K)+k]=std::max(0.,sum-c[k]);
                }
            }branches=true;
        }
        std::atomic<bool> interrupted(false);
        const int workers=std::max(1,threads),pairs=F*F;
        #pragma omp parallel for schedule(static) num_threads(workers)
        for(int pair=0;pair<pairs;++pair){
            if(ready[pair])continue;
            if(Clock::now()>=deadline){interrupted=true;continue;}
            int f=pair/F,g=pair%F;
            if(!allowed[f]||(no_repeat&&f==g)){ready[pair]=1;continue;}
            std::array<Tail,4> local;bool complete=true;
            std::vector<uint64_t> rows(W);
            std::vector<double> c(K),lc(K),rc(K);
            for(int q=0;q<4&&complete;++q){
                int a=q/2,b=q%2;
                if(!allowed[(1+a)*F+g])continue;
                int total=0,positive=0;std::fill(c.begin(),c.end(),0.);
                for(int w=0;w<W;++w){
                    auto bits=branch(f,a,w)&branch(g,b,w);rows[w]=bits;total+=pop(bits);
                    if(uniform&&K==2)positive+=pop(bits&classes[size_t(W)+w]);
                    else for(int k=0;k<K;++k)c[k]+=mass(bits&classes[size_t(k)*W+w],w);
                }
                if(uniform&&K==2){c[0]=(total-positive)*unit;c[1]=positive*unit;}
                auto stop=leaf(c);local[q].label=stop.first;
                if(total>=min_leaf)local[q].stop=stop.second;
                // A zero-cost legal STOP dominates every nonnegative-cost split
                // for every penalty. No separator action is removed by this rule.
                if(early&&local[q].stop==0.)continue;
                for(int h=0;h<F;++h){
                    if(h%8==0&&Clock::now()>=deadline){complete=false;break;}
                    if(!allowed[(3+q)*F+h]||(no_repeat&&(h==f||h==g)))continue;
                    int n0=0,p0=0;
                    const uint64_t* feature=ones.data()+size_t(h)*W;
                    // Routed rows already mask the partial last word. Thus
                    // rows & ~feature needs no repeated valid-word check.
                    // Hoist the binary/uniform dispatch out of the hot loop.
                    if(uniform&&K==2){
                        const uint64_t* positive_bits=classes.data()+W;
                        for(int w=0;w<W;++w){
                            auto bits=rows[w]&~feature[w];n0+=pop(bits);p0+=pop(bits&positive_bits[w]);
                        }
                    }else{
                        std::fill(lc.begin(),lc.end(),0.);
                        for(int w=0;w<W;++w){
                            auto bits=rows[w]&~feature[w];n0+=pop(bits);
                            for(int k=0;k<K;++k)lc[k]+=mass(bits&classes[size_t(k)*W+w],w);
                        }
                    }
                    if(n0<min_leaf||total-n0<min_leaf)continue;
                    double loss;int left,right;
                    if(uniform&&K==2){
                        int n1=total-n0,p1=positive-p0;
                        left=p0>n0-p0?1:0;right=p1>n1-p1?1:0;
                        loss=unit*(std::min(p0,n0-p0)+std::min(p1,n1-p1));
                    }else{
                        for(int k=0;k<K;++k)rc[k]=std::max(0.,c[k]-lc[k]);
                        auto l=leaf(lc),r=leaf(rc);loss=l.second+r.second;left=l.first;right=r.first;
                    }
                    double value=loss+extras[(3+q)*F+h];
                    if(value<local[q].split){local[q].split=value;local[q].h=h;local[q].left=left;local[q].right=right;}
                }
            }
            if(complete){for(int q=0;q<4;++q)tails[size_t(q)*pairs+pair]=local[q];ready[pair]=1;}
            else interrupted=true;
        }
        if(interrupted||Clock::now()>=deadline)throw Timeout{};
    }
    std::vector<Choice> table(double penalty) const {
        std::vector<Choice> tab(size_t(4)*P);
        double sum=std::accumulate(counts.begin(),counts.end(),0.);
        for(int q=0;q<4;++q){
            if(early&&n>=min_leaf)for(int k=0;k<K;++k)
                tab[size_t(q)*P+F*A+k]={.25*std::max(0.,sum-counts[k]),-1,k,0};
            if(!branches)continue;
            int a=q/2;
            for(int f=0;f<F;++f)if(allowed[f]){
                double rootcost=.25*(penalty+extras[f]);
                if(early&&branch_size[2*f+a]>=min_leaf)for(int k=0;k<K;++k)
                    tab[size_t(q)*P+f*A+F+k]={rootcost+.5*branch_loss[size_t(2*f+a)*K+k],-1,k,0};
                for(int g=0;g<F;++g)if(allowed[(1+a)*F+g]&&(!no_repeat||f!=g)){
                    const auto& t=tails[(size_t(q)*F+f)*F+g];auto& c=tab[size_t(q)*P+f*A+g];
                    double prefix=rootcost+.5*(penalty+extras[(1+a)*F+g]);
                    if(early&&std::isfinite(t.stop))c={prefix+t.stop,-1,t.label,0};
                    if(prefix+t.split+penalty<c.cost)c={prefix+t.split+penalty,t.h,t.left,t.right};
                }
            }
        }return tab;
    }
    std::string tree(const Selection& selected,const std::vector<Choice>& tab) const {
        if(selected.prefix[0]<0)return "null";
        std::ostringstream os;auto leaf=[&](int k){os<<"{\"label\":"<<k<<"}";};
        int r=root(selected.prefix[0]);
        if(r>=F){leaf(r-F);return os.str();}
        os<<"{\"feature\":"<<r<<",\"left\":";
        for(int a=0;a<2;++a){
            if(a)os<<",\"right\":";
            int s=selected.prefix[a],g=s%A;
            if(g>=F)leaf(g-F);
            else{
                os<<"{\"feature\":"<<g<<",\"left\":";
                for(int b=0;b<2;++b){
                    if(b)os<<",\"right\":";
                    const auto& c=tab[size_t(2*a+b)*P+s];
                    if(c.h<0)leaf(c.left);
                    else{os<<"{\"feature\":"<<c.h<<",\"left\":";leaf(c.left);os<<",\"right\":";leaf(c.right);os<<"}";}
                }os<<"}";
            }
        }os<<"}";return os.str();
    }
    // Gluing only the selected support. For initialization choose the first
    // feasible root/prefix, rather than pre-solving the global objective.
    Selection glue(const std::vector<Choice>& tab,const std::vector<uint8_t>* support=nullptr,bool first=false) const {
        std::array<std::vector<double>,2> values{std::vector<double>(A,INF),std::vector<double>(A,INF)};
        std::array<std::vector<int>,2> ids{std::vector<int>(A,-1),std::vector<int>(A,-1)};
        for(int a=0;a<2;++a)for(int s=0;s<P;++s){
            int i=2*a*P+s,j=(2*a+1)*P+s,r=root(s);
            if(support&&(!(*support)[i]||!(*support)[j]))continue;
            double value=tab[i].cost+tab[j].cost;
            if(value<values[a][r]&&(!first||ids[a][r]<0)){values[a][r]=value;ids[a][r]=s;}
        }
        Selection out;
        for(int r=0;r<A;++r)if(values[0][r]+values[1][r]<out.value){
            out={values[0][r]+values[1][r],{ids[0][r],ids[1][r]}};if(first)break;
        }return out;
    }
};
std::array<double,4> price(const Workspace& w,const std::vector<Choice>& tab,
                         const double* alpha,const double* pi,std::vector<double>& rc) {
    rc.resize(size_t(4)*w.P);std::array<double,4> minima{0.,0.,0.,0.};
    for(int q=0;q<4;++q)for(int s=0;s<w.P;++s){
        int id=q*w.P+s;double v=tab[id].cost-alpha[q];
        for(int slot=0;slot<w.degree(q);++slot)v-=w.sign(q,slot)*pi[w.edge(q,s,slot)];
        rc[id]=v;minima[q]=std::min(minima[q],v);
    }return minima;
}
void solve(Workspace& w,double penalty,double seconds,int batch,int cap,int method,Reporter report){
    auto start=Clock::now(),deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    std::string status="RUNNING";Selection selected;
    int iterations=0,columns=0,rows=0;double lb=0,first_ub=INF,proof=INF;
    double prepare_seconds=0,pricing_seconds=0,rmp_seconds=0,env_seconds=0,master_seconds=0;
    bool reused=std::all_of(w.ready.begin(),w.ready.end(),[](uint8_t x){return x!=0;});
    std::vector<Choice> tab;std::vector<Event> trace;
    auto json=[&](){
        std::ostringstream os;os<<std::setprecision(17);int major,minor,technical;GRBversion(&major,&minor,&technical);
        os<<"{\"status\":\""<<status<<"\",\"native_seconds\":"<<elapsed(start)<<",\"LB\":";number(os,lb);
        os<<",\"UB\":";number(os,selected.value);os<<",\"tree\":"<<w.tree(selected,tab);
        os<<",\"iterations\":"<<iterations<<",\"columns\":"<<columns<<",\"rows\":"<<rows
          <<",\"pricing_setup_seconds\":"<<prepare_seconds<<",\"pricing_seconds\":"<<pricing_seconds
          <<",\"rmp_seconds\":"<<rmp_seconds<<",\"master_update_seconds\":"<<master_seconds
          <<",\"environment_seconds\":"<<env_seconds<<",\"workspace_reused\":"<<(reused?"true":"false")
          <<",\"first_final_ub_seconds\":";number(os,first_ub);os<<",\"proof_seconds\":";number(os,proof);
        os<<",\"gurobi_version\":\""<<major<<"."<<minor<<"."<<technical<<"\",\"trace\":[";
        for(size_t j=0;j<trace.size();++j){
            auto& e=trace[j];if(j)os<<",";
            os<<"{\"iteration\":"<<e.iteration<<",\"seconds\":"<<e.seconds<<",\"LB\":"<<e.lb<<",\"UB\":"<<e.ub
              <<",\"columns\":"<<e.columns<<",\"rows\":"<<e.rows<<",\"new_columns\":"<<e.new_columns
              <<",\"alpha_sum\":"<<e.alpha<<",\"pricing_lower_bounds\":[";
            for(int q=0;q<4;++q){if(q)os<<",";os<<e.minima[q];}
            os<<"],\"max_equality_residual\":"<<e.residual<<",\"minimum_active_reduced_cost\":"<<e.active_rc<<"}";
        }os<<"]}";w.output=os.str();return w.output.c_str();
    };
    auto check=[&](){if(Clock::now()>=deadline)throw Timeout{};};
    auto publish=[&](){if(report)report(json());};
    try{
        check();w.count_global();
        if(w.early&&w.n>=w.min_leaf){
            int k=Workspace::leaf(w.counts).first,s=w.F*w.A+k;tab=w.table(penalty);
            selected={4*tab[s].cost,{s,s}};first_ub=elapsed(start);publish();
        }
        auto t=Clock::now();
        try{w.prepare(deadline);}catch(...){prepare_seconds+=elapsed(t);throw;}
        tab=w.table(penalty);prepare_seconds+=elapsed(t);
        if(selected.prefix[0]<0){selected=w.glue(tab,nullptr,true);first_ub=elapsed(start);}
        if(selected.prefix[0]<0){status="INFEASIBLE";lb=INF;json();return;}
        check();t=Clock::now();
        if(!w.env){auto env=std::make_unique<GRBEnv>(true);env->set(GRB_IntParam_OutputFlag,0);env->start();w.env=std::move(env);}
        env_seconds+=elapsed(t);check();t=Clock::now();
        GRBModel model(*w.env);model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260905);
        model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        std::vector<GRBConstr> norm;for(int q=0;q<4;++q)norm.push_back(model.addConstr(GRBLinExpr()==1.));
        std::vector<GRBConstr> sep(w.edge_count());std::vector<uint8_t> has_row(w.edge_count(),0),active(size_t(4)*w.P,0);
        std::vector<GRBVar> vars(size_t(4)*w.P);std::vector<int> indices;rows=4;master_seconds+=elapsed(t);
        auto add=[&](const std::vector<int>& ids){
            if(columns+(int)ids.size()>cap)throw std::length_error("Restricted column limit");
            auto begin=Clock::now();
            for(int id:ids){int q=id/w.P,s=id%w.P;
                for(int slot=0;slot<w.degree(q);++slot){int e=w.edge(q,s,slot);
                    if(!has_row[e]){sep[e]=model.addConstr(GRBLinExpr()==0.);has_row[e]=1;++rows;}}}
            model.update();
            for(int id:ids)if(!active[id]){
                int q=id/w.P,s=id%w.P;GRBColumn col;col.addTerm(1.,norm[q]);
                for(int slot=0;slot<w.degree(q);++slot)col.addTerm(w.sign(q,slot),sep[w.edge(q,s,slot)]);
                vars[id]=model.addVar(0.,GRB_INFINITY,tab[id].cost,GRB_CONTINUOUS,col);
                active[id]=1;indices.push_back(id);++columns;
            }model.update();master_seconds+=elapsed(begin);
        };
        add({selected.prefix[0],w.P+selected.prefix[0],2*w.P+selected.prefix[1],3*w.P+selected.prefix[1]});
        for(int iteration=0;iteration<4*w.P+2;++iteration){
            check();model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,std::chrono::duration<double>(deadline-Clock::now()).count()));
            t=Clock::now();model.optimize();rmp_seconds+=elapsed(t);check();
            int code=model.get(GRB_IntAttr_Status);
            if(code==GRB_TIME_LIMIT||code==GRB_INTERRUPTED)throw Timeout{};
            if(code!=GRB_OPTIMAL)throw std::runtime_error("Feasible native D3 RMP failed, status "+std::to_string(code));
            ++iterations;
            std::vector<uint8_t> support(size_t(4)*w.P,0);
            for(int id:indices)if(vars[id].get(GRB_DoubleAttr_X)>1e-9)support[id]=1;
            auto candidate=w.glue(tab,&support);
            if(candidate.prefix[0]<0||std::abs(candidate.value-model.get(GRB_DoubleAttr_ObjVal))>1e-7)
                throw std::runtime_error("D3 RMP support cannot be glued into objective-matching tree");
            if(candidate.value<selected.value-1e-12){selected=candidate;first_ub=elapsed(start);}
            std::array<double,4> alpha;for(int q=0;q<4;++q)alpha[q]=norm[q].get(GRB_DoubleAttr_Pi);
            std::vector<double> pi(w.edge_count(),0.),rc;
            for(int e=0;e<w.edge_count();++e)if(has_row[e])pi[e]=sep[e].get(GRB_DoubleAttr_Pi);
            t=Clock::now();auto minima=price(w,tab,alpha.data(),pi.data(),rc);std::vector<int> seeds;
            for(int q=0;q<4;++q){
                std::vector<int> ids;for(int s=0;s<w.P;++s){int id=q*w.P+s;if(!active[id]&&rc[id]<-TOL)ids.push_back(id);}
                if(batch>0&&(int)ids.size()>batch){
                    auto compare=[&](int a,int b){return rc[a]!=rc[b]?rc[a]<rc[b]:a<b;};
                    std::nth_element(ids.begin(),ids.begin()+batch,ids.end(),compare);ids.resize(batch);
                }
                std::sort(ids.begin(),ids.end());seeds.insert(seeds.end(),ids.begin(),ids.end());
            }
            pricing_seconds+=elapsed(t);check();
            double asum=std::accumulate(alpha.begin(),alpha.end(),0.);
            double certificate=asum+std::accumulate(minima.begin(),minima.end(),0.);
            if(certificate>selected.value+1e-7)throw std::runtime_error("D3 pricing certificate exceeds feasible tree");
            lb=std::max(lb,std::min(selected.value,certificate));
            double residual=0,active_rc=INF;
            for(int q=0;q<4;++q)residual=std::max(residual,std::abs(norm[q].get(GRB_DoubleAttr_Slack)));
            for(int e=0;e<w.edge_count();++e)if(has_row[e])residual=std::max(residual,std::abs(sep[e].get(GRB_DoubleAttr_Slack)));
            for(int id:indices)active_rc=std::min(active_rc,vars[id].get(GRB_DoubleAttr_RC));
            if(residual>1e-7||active_rc<-1e-7)throw std::runtime_error("D3 RMP residual/reduced-cost audit failed");
            trace.push_back({elapsed(start),lb,selected.value,asum,residual,active_rc,minima,iteration,columns,rows,(int)seeds.size()});
            if(seeds.empty()){
                if(selected.value-lb>1e-7)throw std::runtime_error("D3 pricing has no columns but gap is open");
                check();status="OPT";proof=elapsed(start);publish();json();return;
            }
            publish();check();add(seeds);
        }throw std::runtime_error("Finite D3 pricing domain did not converge");
    }catch(const Timeout&){status="TIME";}catch(const std::length_error&){status="RESOURCE";}
    json();publish();
}
}
// Included as a separately fingerprinted source, sharing the exact JT topology.
#include "d3_cg_lazy.cpp"
API const char* d3cg_error(){return last_error.c_str();}
API void* d3cg_create(const uint8_t* X,const int32_t* y,const double* weights,const uint8_t* allowed,
                     const double* extras,int n,int F,int K,int min_leaf,int early,int no_repeat,int threads){
    try{
        last_error.clear();
        if(n<1||F<1||K<1||4LL*(1LL*F*(F+1LL*K)+K)>std::numeric_limits<int>::max()-2)
            throw std::invalid_argument("Invalid/oversized D3 dimensions");
        return new Workspace(X,y,weights,allowed,extras,n,F,K,min_leaf,early,no_repeat,threads);
    }catch(const std::exception& e){last_error=e.what();return nullptr;}
}
API void d3cg_destroy(void* handle){delete static_cast<Workspace*>(handle);}
API int d3cg_configure(void* handle,int mode,int batch,int cache_mb){
    if(!handle||mode<0||mode>2||batch<0||cache_mb<0)return 0;
    auto& w=*static_cast<Workspace*>(handle);w.cost_mode=mode;w.cost_batch=batch;w.cache_mb=cache_mb;return 1;
}
API void d3cg_cost_provider(void* handle,CostProvider provider){
    if(handle)static_cast<Workspace*>(handle)->cost_provider=provider;
}
API void d3cg_direct_cost_provider(void* handle,DirectCostProvider provider){
    if(handle)static_cast<Workspace*>(handle)->direct_cost_provider=provider;
}
API const char* d3cg_prepare_costs(void* handle,double seconds){
    try{
        last_error.clear();if(!handle||seconds<0)throw std::invalid_argument("Invalid cost-table preparation");
        auto& w=*static_cast<Workspace*>(handle);auto start=Clock::now();
        auto deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        if(!w.lazy)w.lazy=std::make_shared<LazyCosts>(w);auto& costs=*w.lazy;
        std::string status="PREPARED";double initial=0.;int direct_batch_states=0;
        auto bg=costs.gpu_batches,bh=costs.hits,bm=costs.misses,bs=costs.stump_evaluations,bw=costs.word_visits,bb=costs.batches;
        double gpu_before=costs.gpu_seconds;
        try{
            if(Clock::now()>=deadline)throw Timeout{};
            w.count_global();costs.prepare_metadata(deadline);initial=elapsed(start);
            if(w.direct_cost_provider){
                std::vector<int32_t> pending;
                for(int pair=0;pair<w.F*w.F;++pair)for(int q=0;q<4;++q){int id=q*w.F*w.F+pair;
                    if(!costs.done[id])pending.push_back(id);}
                int memory_cap=std::max(1,int((64ULL*1024*1024)/(size_t(w.W)*8+size_t(w.F)*16+size_t(w.K)*4+32)));
                int batch_size=std::min(memory_cap,w.cost_batch?int(std::min(16384LL,4LL*w.cost_batch)):4096);
                direct_batch_states=batch_size;
                for(size_t offset=0;offset<pending.size();offset+=batch_size){
                    if(Clock::now()>=deadline)throw Timeout{};
                    int count=std::min(size_t(batch_size),pending.size()-offset);
                    std::vector<double> loss(count);std::vector<int32_t> actions(size_t(count)*3);
                    auto t=Clock::now();int code=w.direct_cost_provider(count,w.ones.data(),w.classes.data(),w.allowed.data(),w.extras.data(),
                        pending.data()+offset,loss.data(),actions.data(),std::max(0.,std::chrono::duration<double>(deadline-Clock::now()).count()));
                    costs.gpu_seconds+=elapsed(t);if(code==-1)throw Timeout{};
                    if(code!=1)throw std::runtime_error("Direct GPU cost callback failed");
                    for(int j=0;j<count;++j){int id=pending[offset+j];auto& tail=w.tails[id];
                        tail.split=loss[j];tail.h=actions[3*j];tail.left=actions[3*j+1];tail.right=actions[3*j+2];costs.done[id]=1;}
                    costs.stump_evaluations+=uint64_t(count)*w.F;costs.word_visits+=uint64_t(count)*w.F*w.W;
                    ++costs.gpu_batches;++costs.batches;costs.refresh_ready();
                }
            }else{
              int cursor=0;
              while(cursor<w.F*w.F){
                if(Clock::now()>=deadline)throw Timeout{};
                int count=w.cost_batch?w.cost_batch:costs.adaptive_batch;std::vector<int> pairs;
                while(cursor<w.F*w.F&&(int)pairs.size()<count){if(!w.ready[cursor])pairs.push_back(cursor);++cursor;}
                if(!pairs.empty())costs.compute_pairs(pairs,deadline);
              }
            }
        }catch(const Timeout&){status="TIME";}
        std::ostringstream os;os<<std::setprecision(17)<<"{\"status\":\""<<status<<"\",\"prepare_seconds\":"<<elapsed(start)
          <<",\"initial_setup_seconds\":"<<initial<<",\"cost_states_resolved\":"<<costs.completed()
          <<",\"cost_states_total\":"<<costs.done.size()<<",\"cost_table_complete\":"<<(status=="PREPARED"?"true":"false")
          <<",\"gpu_full_strategy\":\""<<(w.direct_cost_provider?"direct":"cached")<<"\""
          <<",\"direct_batch_states\":"<<direct_batch_states
          <<",\"cost_backend\":\""<<(costs.gpu_batches>bg?"cuda":"cpp_openmp")<<"\",\"gpu_cost_batches\":"<<costs.gpu_batches-bg
          <<",\"gpu_cost_seconds\":"<<costs.gpu_seconds-gpu_before<<",\"cost_cache_hits\":"<<costs.hits-bh
          <<",\"cost_cache_misses\":"<<costs.misses-bm<<",\"cost_cache_peak_bytes\":"<<costs.peak_cache_bytes
          <<",\"stump_evaluations\":"<<costs.stump_evaluations-bs<<",\"cost_word_visits\":"<<costs.word_visits-bw
          <<",\"cost_batches\":"<<costs.batches-bb<<",\"cost_batch_final\":"<<(w.cost_batch?w.cost_batch:costs.adaptive_batch)<<"}";
        w.output=os.str();return w.output.c_str();
    }catch(const std::exception& e){last_error=e.what();return nullptr;}
}
API const char* d3cg_solve(void* handle,double penalty,double seconds,int batch,int cap,int method,Reporter report){
    try{
        last_error.clear();if(!handle||penalty<0||seconds<0||batch<0||cap<1)throw std::invalid_argument("Invalid D3 solve arguments");
        auto& w=*static_cast<Workspace*>(handle);
        const bool lazy=w.cost_mode==1||(w.cost_mode==2&&w.F>=128&&
            4.*w.F*w.F*w.F*w.W*std::max(1,w.K-1)>2e9);
        if(lazy)solve_lazy(w,penalty,seconds,batch,cap,method,report);
        else solve(w,penalty,seconds,batch,cap,method,report);
        return w.output.c_str();
    }catch(const GRBException& e){last_error="Gurobi "+std::to_string(e.getErrorCode())+": "+e.getMessage();return nullptr;}
     catch(const std::exception& e){last_error=e.what();return nullptr;}
}
API const char* d3cg_price(void* handle,double penalty,const double* alpha,const double* dual){
    try{
        auto& w=*static_cast<Workspace*>(handle);
        if(!std::all_of(w.ready.begin(),w.ready.end(),[](uint8_t x){return x!=0;}))throw std::runtime_error("D3 table is incomplete");
        auto tab=w.table(penalty);std::vector<double> rc;auto minima=price(w,tab,alpha,dual,rc);
        std::ostringstream os;os<<std::setprecision(17)<<"{\"costs\":[";
        for(size_t i=0;i<tab.size();++i){if(i)os<<",";number(os,tab[i].cost);}
        os<<"],\"reduced_costs\":[";for(size_t i=0;i<rc.size();++i){if(i)os<<",";number(os,rc[i]);}
        os<<"],\"minima\":[";for(int q=0;q<4;++q){if(q)os<<",";os<<minima[q];}
        os<<"]}";w.output=os.str();return w.output.c_str();
    }catch(const std::exception& e){last_error=e.what();return nullptr;}
}
