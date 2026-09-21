// D2 junction-tree column generation. Python is only the C ABI/data adapter.
// Two normalization rows and union-indexed root-action separator rows.
// Private stump elimination is exact; the global tree is recovered from RMP
// support, and optimality requires a complete reduced-cost pricing certificate.
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
#ifdef _MSC_VER
#include <intrin.h>
#define API extern "C" __declspec(dllexport)
static int pop(uint64_t x) { return (int)__popcnt64(x); }
static int lowbit(uint64_t x) { unsigned long p; _BitScanForward64(&p, x); return (int)p; }
#else
#define API extern "C"
static int pop(uint64_t x) { return __builtin_popcountll(x); }
static int lowbit(uint64_t x) { return __builtin_ctzll(x); }
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double TOL = 1e-9;
struct Timeout {};
thread_local std::string last_error;
using Reporter = void (*)(const char*);
using CostProvider = int (*)(const uint64_t*, const uint64_t*, const uint8_t*, const double*, double*, int32_t*, double);
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void number(std::ostream& os, double x) { if (std::isfinite(x)) os << x; else os << "null"; }
struct Tail {
    double stop = INF, split = INF;
    int stop_label = 0, feature = -1, left_label = 0, right_label = 0;
};
struct Choice { double cost = INF; int feature = -1, left = 0, right = 0; };
struct Event {
    double seconds, lb, ub, alpha0, alpha1, min0, min1, residual, active_rc;
    int iteration, columns, rows, new_columns;
};
struct Workspace {
    int n, F, K, W, min_leaf, early, no_repeat, threads;
    const uint8_t* X; const int32_t* y; const double* weights;
    std::vector<uint8_t> allowed;
    std::vector<double> extras, global_counts;
    std::vector<uint64_t> ones, classes;
    std::vector<Tail> tails;
    std::vector<uint8_t> ready;
    bool packed = false, counted = false, uniform = true;
    CostProvider provider = nullptr;
    double pack_seconds = 0., cost_seconds = 0.;
    double unit;
    std::unique_ptr<GRBEnv> env;
    std::string output;
    Workspace(const uint8_t* x, const int32_t* labels, const double* w,
              const uint8_t* a, const double* c, int nn, int ff, int kk,
              int ml, int es, int nr, int th):
        n(nn), F(ff), K(kk), W((nn+63)/64), min_leaf(ml), early(es),
        no_repeat(nr), threads(th), X(x), y(labels), weights(w),
        allowed(a,a+3*ff), extras(c,c+3*ff), tails(2*ff), ready(ff,0), unit(w[0]) {
        for(int i=1;i<n;++i) if(w[i]!=unit) { uniform=false; break; }
    }
    uint64_t valid(int word) const {
        return word==W-1 && n%64 ? (uint64_t(1)<<(n%64))-1 : ~uint64_t(0);
    }
    uint64_t branch(int f,int b,int w) const {
        const auto v=ones[size_t(f)*W+w]; return b?v:(valid(w)^v);
    }
    double mass(uint64_t bits, int word) const {
        if(uniform) return pop(bits)*unit;
        double v=0; while(bits) { const int j=lowbit(bits); v+=weights[word*64+j]; bits&=bits-1; } return v;
    }
    static std::pair<int,double> leaf(const std::vector<double>& counts) {
        int k=(int)(std::max_element(counts.begin(),counts.end())-counts.begin());
        return {k,std::max(0.,std::accumulate(counts.begin(),counts.end(),0.)-counts[k])};
    }
    void count_global() {
        if(counted) return;
        global_counts.assign(K,0.);
        for(int i=0;i<n;++i) global_counts[y[i]]+=weights[i];
        counted=true;
    }
    void pack(Clock::time_point deadline) {
        if(packed) return;
        ones.assign(size_t(F)*W,0); classes.assign(size_t(K)*W,0);
        for(int i=0;i<n;++i) {
            if(i%1024==0 && Clock::now()>=deadline) throw Timeout{};
            const auto bit=uint64_t(1)<<(i%64); const int w=i/64;
            classes[size_t(y[i])*W+w]|=bit;
            for(int f=0;f<F;++f) if(X[size_t(i)*F+f]) ones[size_t(f)*W+w]|=bit;
        }
        packed=true;
    }
    void prepare_uniform(Clock::time_point deadline) {
        const int workers=std::max(1,threads);
        // Marginal counts plus a single intersection give all four leaf cells.
        std::vector<int> marginal(size_t(F+1)*K,0);
        for(int i=0;i<n;++i)++marginal[size_t(F)*K+y[i]];
        std::atomic<bool> interrupted(false);
        #pragma omp parallel for schedule(static) num_threads(workers)
        for(int f=0;f<F;++f) {
            if(Clock::now()>=deadline){interrupted=true;continue;}
            for(int k=0;k<K;++k) {
                int value=0;for(int w=0;w<W;++w)value+=pop(ones[size_t(f)*W+w]&classes[size_t(k)*W+w]);
                marginal[size_t(f)*K+k]=value;
            }
        }
        if(interrupted)throw Timeout{};
        // Bounded quadratic storage, not a triple table. If too large, the
        // original streaming implementation remains the exact fallback.
        std::vector<double> loss(size_t(F)*F*2,INF);
        std::vector<int> labels(size_t(F)*F*4,0);
        #pragma omp parallel for schedule(dynamic) num_threads(workers)
        for(int f=0;f<F;++f) for(int g=f;g<F;++g) {
            if(Clock::now()>=deadline){interrupted=true;break;}
            if((!allowed[f]&&!allowed[g])||(no_repeat&&f==g))continue;
            int totals[4]={0,0,0,0},maxima[4]={-1,-1,-1,-1},majority[4]={0,0,0,0};
            for(int k=0;k<K;++k) {
                int both=0;
                for(int w=0;w<W;++w)both+=pop(ones[size_t(f)*W+w]&ones[size_t(g)*W+w]&classes[size_t(k)*W+w]);
                int a=marginal[size_t(f)*K+k],b=marginal[size_t(g)*K+k],t=marginal[size_t(F)*K+k];
                int cell[4]={t-a-b+both,b-both,a-both,both};
                for(int j=0;j<4;++j){totals[j]+=cell[j];if(cell[j]>maxima[j]){maxima[j]=cell[j];majority[j]=k;}}
            }
            for(int transpose=0;transpose<(f==g?1:2);++transpose) {
                int root=transpose?g:f,child=transpose?f:g;
                for(int b=0;b<2;++b) {
                    int left=transpose?b:2*b,right=transpose?b+2:2*b+1;
                    size_t id=(size_t(root)*F+child)*2+b;
                    if(totals[left]>=min_leaf&&totals[right]>=min_leaf)
                        loss[id]=(totals[left]+totals[right]-maxima[left]-maxima[right])*unit;
                    labels[2*id]=majority[left];labels[2*id+1]=majority[right];
                }
            }
        }
        if(interrupted||Clock::now()>=deadline)throw Timeout{};
        for(int f=0;f<F;++f) {
            if(Clock::now()>=deadline)throw Timeout{};
            for(int b=0;b<2;++b) {
                Tail tail;int total=0,maximum=-1;
                for(int k=0;k<K;++k) {
                    int value=b?marginal[size_t(f)*K+k]:marginal[size_t(F)*K+k]-marginal[size_t(f)*K+k];
                    total+=value;if(value>maximum){maximum=value;tail.stop_label=k;}
                }
                if(allowed[f]&&total>=min_leaf)tail.stop=(total-maximum)*unit;
                if(allowed[f])for(int g=0;g<F;++g) {
                    if(!allowed[(b+1)*F+g]||(no_repeat&&f==g))continue;
                    size_t id=(size_t(f)*F+g)*2+b;double value=loss[id]+extras[(b+1)*F+g];
                    if(value<tail.split){tail.split=value;tail.feature=g;tail.left_label=labels[2*id];tail.right_label=labels[2*id+1];}
                }
                tails[2*f+b]=tail;
            }
            ready[f]=1;
        }
    }
    void prepare(Clock::time_point deadline) {
        auto pack_start=Clock::now();
        try {pack(deadline);} catch(...) {pack_seconds+=elapsed(pack_start);throw;}
        pack_seconds+=elapsed(pack_start);
        if(std::all_of(ready.begin(),ready.end(),[](uint8_t x){return x!=0;})) return;
        auto cost_start=Clock::now();
        if(provider) {
            std::vector<double> losses(size_t(4)*F,INF);
            std::vector<int32_t> actions(size_t(8)*F,0);
            int code=provider(ones.data(),classes.data(),allowed.data(),extras.data(),losses.data(),actions.data(),
                std::max(0.,std::chrono::duration<double>(deadline-Clock::now()).count()));
            cost_seconds+=elapsed(cost_start);
            if(code<0 || Clock::now()>=deadline) throw Timeout{};
            if(code==0) throw std::runtime_error("D2 GPU cost provider failed");
            if(code==1) {
                for(int j=0;j<2*F;++j) {
                    auto& t=tails[j];t.stop=losses[2*j];t.split=losses[2*j+1];
                    t.stop_label=actions[4*j];t.feature=actions[4*j+1];
                    t.left_label=actions[4*j+2];t.right_label=actions[4*j+3];
                }
                std::fill(ready.begin(),ready.end(),1);return;
            }
            // Only the auto policy may request an explicit CPU fallback.
            cost_start=Clock::now();
        }
        if(uniform && size_t(F)*F <= (128*1024*1024)/32) {
            try {prepare_uniform(deadline);}catch(...){cost_seconds+=elapsed(cost_start);throw;}
            cost_seconds+=elapsed(cost_start);return;
        }
        std::atomic<bool> interrupted(false);
        const int workers=std::max(1,threads);
        #pragma omp parallel for schedule(static) num_threads(workers)
        for(int f=0;f<F;++f) {
            if(ready[f]) continue;
            if(Clock::now()>=deadline) { interrupted=true; continue; }
            if(!allowed[f]) { ready[f]=1; continue; }
            bool complete=true;
            std::array<Tail,2> local;
            for(int b=0;b<2 && complete;++b) {
                std::vector<double> counts(K,0.); int total=0;
                for(int w=0;w<W;++w) {
                    const auto rows=branch(f,b,w); total+=pop(rows);
                    for(int k=0;k<K;++k) counts[k]+=mass(rows&classes[size_t(k)*W+w],w);
                }
                auto stop=leaf(counts); local[b].stop_label=stop.first;
                if(total>=min_leaf) local[b].stop=stop.second;
                for(int g=0;g<F;++g) {
                    if(g%8==0 && Clock::now()>=deadline) { complete=false; break; }
                    if(!allowed[(b+1)*F+g] || (no_repeat && f==g)) continue;
                    std::vector<double> left(K,0.),right(K); int nleft=0;
                    for(int w=0;w<W;++w) {
                        const auto bits=branch(f,b,w)&branch(g,0,w); nleft+=pop(bits);
                        for(int k=0;k<K;++k) left[k]+=mass(bits&classes[size_t(k)*W+w],w);
                    }
                    if(nleft<min_leaf || total-nleft<min_leaf) continue;
                    for(int k=0;k<K;++k) right[k]=std::max(0.,counts[k]-left[k]);
                    auto l=leaf(left),r=leaf(right);
                    double value=l.second+r.second+extras[(b+1)*F+g];
                    if(value<local[b].split) {
                        local[b].split=value; local[b].feature=g;
                        local[b].left_label=l.first; local[b].right_label=r.first;
                    }
                }
            }
            if(complete) { tails[2*f]=local[0];tails[2*f+1]=local[1];ready[f]=1; }
            else interrupted=true;
        }
        cost_seconds+=elapsed(cost_start);
        if(interrupted || Clock::now()>=deadline) throw Timeout{};
    }
    std::vector<Choice> table(double penalty) const {
        int S=F+K; std::vector<Choice> out(2*S);
        double total=std::accumulate(global_counts.begin(),global_counts.end(),0.);
        for(int b=0;b<2;++b) {
            for(int f=0;f<F;++f) if(allowed[f]) {
                const auto& t=tails[2*f+b]; auto& c=out[b*S+f];
                const double root=.5*(penalty+extras[f]);
                if(early && std::isfinite(t.stop)) { c.cost=root+t.stop;c.left=t.stop_label; }
                if(root+t.split+penalty<c.cost) {
                    c.cost=root+t.split+penalty;c.feature=t.feature;c.left=t.left_label;c.right=t.right_label;
                }
            }
            if(early && n>=min_leaf) for(int k=0;k<K;++k) {
                auto& c=out[b*S+F+k]; c.cost=.5*std::max(0.,total-global_counts[k]);c.left=k;
            }
        }
        return out;
    }
    std::string tree(int signature,const std::vector<Choice>& tab) const {
        if(signature<0) return "null";
        std::ostringstream os;
        auto leaf=[&](int k){ os<<"{\"label\":"<<k<<"}"; };
        if(signature>=F) leaf(signature-F);
        else {
            os<<"{\"feature\":"<<signature<<",\"left\":";
            for(int b=0;b<2;++b) {
                if(b) os<<",\"right\":";
                const auto& c=tab[b*(F+K)+signature];
                if(c.feature<0) leaf(c.left);
                else { os<<"{\"feature\":"<<c.feature<<",\"left\":";leaf(c.left);os<<",\"right\":";leaf(c.right);os<<"}"; }
            }
            os<<"}";
        }
        return os.str();
    }
};
std::array<double,2> price(const std::vector<Choice>& tab,int S,
                         const std::array<double,2>& alpha,const std::vector<double>& pi,
                         std::vector<double>& reduced) {
    reduced.resize(2*S); std::array<double,2> minima{0.,0.};
    for(int b=0;b<2;++b) for(int s=0;s<S;++s) {
        const auto rc=tab[b*S+s].cost-alpha[b]-(b?-1.:1.)*pi[s];
        reduced[b*S+s]=rc;minima[b]=std::min(minima[b],rc);
    }
    return minima;
}

void solve(Workspace& w,double penalty,double seconds,int batch,int cap,int method,Reporter report) {
    auto start=Clock::now(),deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    std::string status="RUNNING",reason;
    int signature=-1,iterations=0,columns=0,rows=0,S=w.F+w.K;
    double lb=0,ub=INF,first_ub=INF,proof=INF,prepare_seconds=0,rmp_seconds=0,pricing_seconds=0,env_seconds=0;
    const bool reused=std::all_of(w.ready.begin(),w.ready.end(),[](uint8_t x){return x!=0;});
    w.pack_seconds=0.;w.cost_seconds=0.;
    std::vector<Choice> tab; std::vector<Event> events;
    auto json=[&]() {
        std::ostringstream os;os<<std::setprecision(17);
        int major,minor,technical;GRBversion(&major,&minor,&technical);
        os<<"{\"status\":\""<<status<<"\",\"native_seconds\":"<<elapsed(start)<<",\"LB\":";number(os,lb);
        os<<",\"UB\":";number(os,ub);os<<",\"tree\":"<<w.tree(signature,tab);
        os<<",\"iterations\":"<<iterations<<",\"columns\":"<<columns<<",\"rows\":"<<rows
          <<",\"pricing_setup_seconds\":"<<prepare_seconds<<",\"pricing_seconds\":"<<pricing_seconds
          <<",\"rmp_seconds\":"<<rmp_seconds<<",\"environment_seconds\":"<<env_seconds
          <<",\"native_pack_seconds\":"<<w.pack_seconds<<",\"native_cost_seconds\":"<<w.cost_seconds
          <<",\"workspace_reused\":"<<(reused?"true":"false")<<",\"first_final_ub_seconds\":";number(os,first_ub);
        os<<",\"proof_seconds\":";number(os,proof);
        os<<",\"gurobi_version\":\""<<major<<"."<<minor<<"."<<technical<<"\",\"trace\":[";
        for(size_t j=0;j<events.size();++j) {
            const auto& e=events[j];if(j)os<<",";
            os<<"{\"iteration\":"<<e.iteration<<",\"seconds\":"<<e.seconds<<",\"LB\":"<<e.lb<<",\"UB\":"<<e.ub
              <<",\"columns\":"<<e.columns<<",\"rows\":"<<e.rows<<",\"new_columns\":"<<e.new_columns
              <<",\"alpha_sum\":"<<e.alpha0+e.alpha1<<",\"pricing_lower_bounds\":["<<e.min0<<","<<e.min1
              <<"],\"max_equality_residual\":"<<e.residual<<",\"minimum_active_reduced_cost\":"<<e.active_rc<<"}";
        }
        os<<"]}";w.output=os.str();return w.output.c_str();
    };
    auto check=[&](){if(Clock::now()>=deadline)throw Timeout{};};
    auto publish=[&](){if(report)report(json());};
    try {
        check();w.count_global();
        // Feasible initializer, not a scan for the globally optimal root.
        if(w.early && w.n>=w.min_leaf) {
            int k=Workspace::leaf(w.global_counts).first;signature=w.F+k;
            tab=w.table(penalty);ub=2*tab[signature].cost;first_ub=elapsed(start);publish();
        }
        auto t=Clock::now();
        try {w.prepare(deadline);} catch(...) {prepare_seconds+=elapsed(t);throw;}
        prepare_seconds+=elapsed(t);tab=w.table(penalty);
        if(signature<0) for(int s=0;s<S;++s) if(std::isfinite(tab[s].cost+tab[S+s].cost)) {
            signature=s;ub=tab[s].cost+tab[S+s].cost;first_ub=elapsed(start);break;
        }
        if(signature<0) {status="INFEASIBLE";lb=INF;json();return;}
        check();
        t=Clock::now();
        if(!w.env) {
            auto env=std::make_unique<GRBEnv>(true);env->set(GRB_IntParam_OutputFlag,0);env->start();w.env=std::move(env);
        }
        env_seconds+=elapsed(t);check();
        GRBModel model(*w.env);
        model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260905);
        model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        std::array<GRBConstr,2> norm{model.addConstr(GRBLinExpr()==1.),model.addConstr(GRBLinExpr()==1.)};
        std::vector<GRBConstr> sep(S);std::vector<uint8_t> has_row(S,0),active(2*S,0);
        std::vector<GRBVar> vars(2*S);std::vector<int> indices;rows=2;
        auto add=[&](const std::vector<int>& ids) {
            if(columns+(int)ids.size()>cap) throw std::length_error("Restricted column limit");
            for(int id:ids) if(!has_row[id%S]) {sep[id%S]=model.addConstr(GRBLinExpr()==0.);has_row[id%S]=1;++rows;}
            model.update();
            for(int id:ids) if(!active[id]) {
                int b=id/S,s=id%S;GRBColumn col;col.addTerm(1.,norm[b]);col.addTerm(b?-1.:1.,sep[s]);
                vars[id]=model.addVar(0.,GRB_INFINITY,tab[id].cost,GRB_CONTINUOUS,col);
                active[id]=1;indices.push_back(id);++columns;
            }
            model.update();
        };
        add({signature,S+signature});
        for(int iteration=0;iteration<2*S+2;++iteration) {
            check();model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,std::chrono::duration<double>(deadline-Clock::now()).count()));
            t=Clock::now();model.optimize();rmp_seconds+=elapsed(t);check();
            int code=model.get(GRB_IntAttr_Status);
            if(code==GRB_TIME_LIMIT || code==GRB_INTERRUPTED)throw Timeout{};
            if(code!=GRB_OPTIMAL)throw std::runtime_error("Feasible native RMP failed, status "+std::to_string(code));
            ++iterations;
            double supported=INF;int selected=-1;
            for(int s=0;s<S;++s) if(active[s] && active[S+s] &&
                vars[s].get(GRB_DoubleAttr_X)>1e-9 && vars[S+s].get(GRB_DoubleAttr_X)>1e-9) {
                double value=tab[s].cost+tab[S+s].cost;
                if(value<supported) {supported=value;selected=s;}
            }
            if(selected<0 || std::abs(supported-model.get(GRB_DoubleAttr_ObjVal))>1e-7)
                throw std::runtime_error("RMP support could not be glued into an objective-matching tree");
            if(supported<ub-1e-12) {ub=supported;signature=selected;first_ub=elapsed(start);}
            std::array<double,2> alpha{norm[0].get(GRB_DoubleAttr_Pi),norm[1].get(GRB_DoubleAttr_Pi)};
            std::vector<double> pi(S,0.),reduced;
            for(int s=0;s<S;++s) if(has_row[s])pi[s]=sep[s].get(GRB_DoubleAttr_Pi);
            t=Clock::now();auto minima=price(tab,S,alpha,pi,reduced);std::vector<int> seeds;
            for(int b=0;b<2;++b) {
                std::vector<int> ids;
                for(int s=0;s<S;++s) if(!active[b*S+s] && reduced[b*S+s]<-TOL)ids.push_back(b*S+s);
                if(batch>0 && (int)ids.size()>batch) {
                    std::sort(ids.begin(),ids.end(),[&](int a,int c){return reduced[a]!=reduced[c]?reduced[a]<reduced[c]:a<c;});ids.resize(batch);
                }
                seeds.insert(seeds.end(),ids.begin(),ids.end());
            }
            pricing_seconds+=elapsed(t);check();
            double certificate=alpha[0]+alpha[1]+minima[0]+minima[1];
            if(certificate>ub+1e-7)throw std::runtime_error("Pricing certificate exceeds feasible tree");
            lb=std::max(lb,std::min(ub,certificate));
            double residual=std::max(std::abs(norm[0].get(GRB_DoubleAttr_Slack)),std::abs(norm[1].get(GRB_DoubleAttr_Slack))),active_rc=INF;
            for(int s=0;s<S;++s) if(has_row[s])residual=std::max(residual,std::abs(sep[s].get(GRB_DoubleAttr_Slack)));
            for(int id:indices)active_rc=std::min(active_rc,vars[id].get(GRB_DoubleAttr_RC));
            if(residual>1e-7 || active_rc<-1e-7)throw std::runtime_error("RMP residual/reduced-cost audit failed");
            events.push_back({elapsed(start),lb,ub,alpha[0],alpha[1],minima[0],minima[1],residual,active_rc,iteration,columns,rows,(int)seeds.size()});
            if(seeds.empty()) {
                if(ub-lb>1e-7)throw std::runtime_error("No improving columns but pricing gap is open");
                check();status="OPT";proof=elapsed(start);publish();json();return;
            }
            publish();check();add(seeds);
        }
        throw std::runtime_error("Finite D2 pricing domain did not converge");
    } catch(const Timeout&) {status="TIME";}
      catch(const std::length_error&) {status="RESOURCE";}
    json();publish();
}

// Experiment A at D2: private child decisions are already minimized in
// Workspace::table (SC).  EE joins the two root-signature blocks and therefore
// leaves a single block whose minimum is obtained without an LP.
void solve_structural_lp(Workspace& w,double penalty,double seconds,bool ee,int method) {
    auto start=Clock::now(),deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    w.pack_seconds=0.;w.cost_seconds=0.;int S=w.F+w.K,signature=-1,status_code=GRB_OPTIMAL;
    double prepare_seconds=0.,ee_seconds=0.,rmp_setup=0.,rmp_insert=0.,rmp_seconds=0.,value=INF;
    int prepared=0,common=0,rows=0,nonzeros=0,iterations=0;
    std::vector<Choice> tab;
    auto json=[&](const char* status){std::ostringstream os;os<<std::setprecision(17)
      <<"{\"status\":\""<<status<<"\",\"LB\":";number(os,value);os<<",\"UB\":";number(os,value);
      os<<",\"tree\":"<<(signature>=0?w.tree(signature,tab):"null")
        <<",\"engine\":\"cpp_gurobi_d2_structural_v1\",\"reduction\":\""<<(ee?"sc_ee":"sc")<<"\""
        <<",\"backend\":\""<<(w.provider?"cuda":"cpp_openmp")<<"\",\"raw_columns\":null"
        <<",\"prepared_columns\":"<<prepared<<",\"master_domain_columns\":"<<(ee?common:prepared)
        <<",\"remaining_clusters\":"<<(ee?1:2)<<",\"submitted_rows\":"<<rows
        <<",\"peak_active_columns\":"<<(ee?common:prepared)<<",\"peak_rmp_nonzeros\":"<<nonzeros
        <<",\"iterations\":"<<iterations<<",\"cost_preparation_seconds\":"<<prepare_seconds
        <<",\"sc_seconds\":0,\"ee_seconds\":"<<ee_seconds<<",\"rmp_setup_seconds\":"<<rmp_setup
        <<",\"rmp_insertion_seconds\":"<<rmp_insert<<",\"rmp_seconds\":"<<rmp_seconds
        <<",\"pricing_seconds\":0,\"native_pack_seconds\":"<<w.pack_seconds
        <<",\"native_cost_seconds\":"<<w.cost_seconds<<",\"native_seconds\":"<<elapsed(start)<<"}";
      w.output=os.str();};
    try {
        if(Clock::now()>=deadline)throw Timeout{};w.count_global();auto t=Clock::now();w.prepare(deadline);
        prepare_seconds=elapsed(t);tab=w.table(penalty);
        for(const auto& c:tab)if(std::isfinite(c.cost))++prepared;
        t=Clock::now();for(int s=0;s<S;++s)if(std::isfinite(tab[s].cost)&&std::isfinite(tab[S+s].cost)){
          ++common;double candidate=tab[s].cost+tab[S+s].cost;if(candidate<value){value=candidate;signature=s;}}
        if(signature<0){value=INF;json("INFEASIBLE");return;}if(ee){ee_seconds=elapsed(t);json("OPT");return;}
        if(Clock::now()>=deadline)throw Timeout{};t=Clock::now();
        if(!w.env){auto env=std::make_unique<GRBEnv>(true);env->set(GRB_IntParam_OutputFlag,0);env->start();w.env=std::move(env);}
        GRBModel model(*w.env);model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260916);
        model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        std::array<GRBConstr,2> norm{model.addConstr(GRBLinExpr()==1.),model.addConstr(GRBLinExpr()==1.)};
        std::vector<GRBConstr> sep(S);for(int s=0;s<S;++s)sep[s]=model.addConstr(GRBLinExpr()==0.);model.update();
        rmp_setup=elapsed(t);t=Clock::now();
        for(int b=0;b<2;++b)for(int s=0;s<S;++s){int id=b*S+s;if(!std::isfinite(tab[id].cost))continue;
          GRBColumn col;col.addTerm(1.,norm[b]);col.addTerm(b?-1.:1.,sep[s]);model.addVar(0.,GRB_INFINITY,tab[id].cost,GRB_CONTINUOUS,col);}
        model.update();rmp_insert=elapsed(t);rows=model.get(GRB_IntAttr_NumConstrs);nonzeros=model.get(GRB_IntAttr_NumNZs);
        model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,std::chrono::duration<double>(deadline-Clock::now()).count()));
        t=Clock::now();model.optimize();rmp_seconds=elapsed(t);status_code=model.get(GRB_IntAttr_Status);iterations=1;
        if(status_code==GRB_TIME_LIMIT||status_code==GRB_INTERRUPTED)throw Timeout{};
        if(status_code!=GRB_OPTIMAL)throw std::runtime_error("D2 structural LP failed, status "+std::to_string(status_code));
        if(std::abs(model.get(GRB_DoubleAttr_ObjVal)-value)>1e-7)throw std::runtime_error("D2 SC LP and EE join disagree");
        json("OPT");
    }catch(const Timeout&){value=INF;signature=-1;json("TIME");}
}
}

API const char* d2cg_error() {return last_error.c_str();}
API void* d2cg_create(const uint8_t* X,const int32_t* y,const double* weights,
                     const uint8_t* allowed,const double* extras,int n,int F,int K,
                     int min_leaf,int early_stop,int no_repeat,int threads) {
    try {last_error.clear();if(n<1||F<1||K<1)throw std::invalid_argument("Invalid D2 dimensions");
        return new Workspace(X,y,weights,allowed,extras,n,F,K,min_leaf,early_stop,no_repeat,threads);
    }catch(const std::exception& e){last_error=e.what();return nullptr;}
}
API void d2cg_destroy(void* handle) {delete static_cast<Workspace*>(handle);}
// Reuse already validated, immutable Problem bitsets. Ownership stays native.
API int d2cg_set_packed(void* handle,const uint64_t* features,const uint64_t* classes) {
    try {auto& w=*static_cast<Workspace*>(handle);
        w.ones.assign(features,features+size_t(w.F)*w.W);
        w.classes.assign(classes,classes+size_t(w.K)*w.W);w.packed=true;return 1;
    }catch(const std::exception& e){last_error=e.what();return 0;}
}
API void d2cg_cost_provider(void* handle,CostProvider provider) {
    static_cast<Workspace*>(handle)->provider=provider;
}
API const char* d2cg_solve(void* handle,double penalty,double seconds,int batch,int cap,int method,Reporter report) {
    try {last_error.clear();if(!handle||penalty<0||seconds<0||batch<0||cap<1)throw std::invalid_argument("Invalid D2 solve arguments");
        auto& w=*static_cast<Workspace*>(handle);solve(w,penalty,seconds,batch,cap,method,report);return w.output.c_str();
    }catch(const GRBException& e){last_error="Gurobi "+std::to_string(e.getErrorCode())+": "+e.getMessage();return nullptr;}
     catch(const std::exception& e){last_error=e.what();return nullptr;}
}
API const char* d2cg_structural(void* handle,double penalty,double seconds,int ee,int method) {
    try {last_error.clear();if(!handle||penalty<0||seconds<0||(ee!=0&&ee!=1))throw std::invalid_argument("Invalid D2 structural arguments");
        auto& w=*static_cast<Workspace*>(handle);solve_structural_lp(w,penalty,seconds,ee!=0,method);return w.output.c_str();
    }catch(const GRBException& e){last_error="Gurobi "+std::to_string(e.getErrorCode())+": "+e.getMessage();return nullptr;}
     catch(const std::exception& e){last_error=e.what();return nullptr;}
}
// Diagnostic API uses the same pricing function as the production CG loop.
API const char* d2cg_price(void* handle,double penalty,const double* alpha,const double* dual) {
    try {
        auto& w=*static_cast<Workspace*>(handle);int S=w.F+w.K;
        if(!std::all_of(w.ready.begin(),w.ready.end(),[](uint8_t x){return x!=0;}))throw std::runtime_error("D2 table is incomplete");
        auto tab=w.table(penalty);std::vector<double> pi(dual,dual+S),rc;
        auto minima=price(tab,S,{alpha[0],alpha[1]},pi,rc);
        std::ostringstream os;os<<std::setprecision(17)<<"{\"costs\":[";
        for(int i=0;i<2*S;++i){if(i)os<<",";number(os,tab[i].cost);}
        os<<"],\"reduced_costs\":[";
        for(int i=0;i<2*S;++i){if(i)os<<",";number(os,rc[i]);}
        os<<"],\"minima\":["<<minima[0]<<","<<minima[1]<<"]}";w.output=os.str();return w.output.c_str();
    }catch(const std::exception& e){last_error=e.what();return nullptr;}
}
