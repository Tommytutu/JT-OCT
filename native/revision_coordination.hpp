// Revision controls: immutable exact costs, native EE/MP/LP/CG and tree audit.
// No oracle is reachable from this coordinator. The caller owns input buffers.
#pragma once
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include "gurobi_c++.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace revision {
using Clock=std::chrono::steady_clock;
constexpr double inf=std::numeric_limits<double>::infinity();
constexpr int inactive=INT32_MIN;
inline double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
inline void num(std::ostream& s,double v){if(std::isfinite(v))s<<v;else s<<"null";}
struct Timed{};
struct Limited:std::runtime_error{using std::runtime_error::runtime_error;};
struct Budget {
    Clock::time_point start=Clock::now(); double seconds; uint64_t memory;
    Budget(double s,uint64_t m):seconds(s),memory(m){if(!std::isfinite(s)||s<0||m==0)throw std::invalid_argument("Invalid resource budget");}
    double remaining()const{return std::max(0.,seconds-elapsed(start));}
    void check()const{if(remaining()<=0)throw Timed{};PROCESS_MEMORY_COUNTERS_EX p{};p.cb=sizeof(p);
        if(!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&p),sizeof(p)))throw std::runtime_error("Memory query failed");
        if(p.PrivateUsage>memory)throw Limited("Measured process memory limit");}
};

struct Data {
    const uint8_t* X; int n,F,D,K,threads,min_leaf; bool early,no_repeat;
    std::vector<int64_t> labels;std::vector<int32_t> y;std::vector<double> weights,extras;std::vector<uint8_t> allowed;
    Data(const uint8_t* x,const int64_t* yy,const double* w,const uint8_t* allow,const double* extra,
         int nn,int ff,int d,int es,int nr,int ml,int th):X(x),n(nn),F(ff),D(d),threads(th),min_leaf(ml),early(es!=0),no_repeat(nr!=0){
        if(!x||!yy||n<1||F<1||D<1||D>5||threads<1||threads>256||min_leaf<0)throw std::invalid_argument("Invalid data dimensions");
        labels.assign(yy,yy+n);std::sort(labels.begin(),labels.end());labels.erase(std::unique(labels.begin(),labels.end()),labels.end());K=int(labels.size());
        if(labels.front()<0)throw std::invalid_argument("Labels must be nonnegative");
        if(4LL*(int64_t(F)*(int64_t(F)+K)+K)>INT32_MAX)throw std::invalid_argument("Signature index domain exceeds int32");
        y.resize(n);weights.resize(n);int invalid=0;double mass=0;
        #pragma omp parallel for num_threads(threads) reduction(|:invalid) reduction(+:mass)
        for(int i=0;i<n;++i){y[i]=int(std::lower_bound(labels.begin(),labels.end(),yy[i])-labels.begin());weights[i]=w?w[i]:1./n;
            if(!std::isfinite(weights[i])||weights[i]<0)invalid=1;mass+=weights[i];
            for(int f=0;f<F;++f)if(X[size_t(i)*F+f]>1)invalid=1;}
        if(invalid||std::abs(mass-1)>1e-8)throw std::invalid_argument("Invalid binary data or normalized weights");
        size_t count=size_t((1<<D)-1)*F;allowed.resize(count);extras.resize(count);
        #pragma omp parallel for num_threads(threads) reduction(|:invalid)
        for(int64_t j=0;j<int64_t(count);++j){allowed[j]=allow?allow[j]:1;extras[j]=extra?extra[j]:0.;
            if(allowed[j]>1||!std::isfinite(extras[j])||extras[j]<0)invalid=1;}
        if(invalid)throw std::invalid_argument("Invalid feature restrictions or split costs");
    }
    int nodes()const{return (1<<(D+1))-1;}
    int majority()const{std::vector<double> h(K);for(int i=0;i<n;++i)h[y[i]]+=weights[i];return int(std::max_element(h.begin(),h.end())-h.begin());}
    double audit(const std::vector<int>& tree,double penalty)const{
        if(int(tree.size())!=nodes())throw std::runtime_error("Invalid recovered tree size");
        double splitcost=0;int bad=0;std::vector<int> leaf(nodes(),0),depth(nodes(),0);
        for(int v=0;v<nodes();++v){int a=tree[v];if(v)depth[v]=depth[(v-1)/2]+1;
            bool active=v==0||tree[(v-1)/2]>=0;
            if(!active){if(a!=inactive)throw std::runtime_error("Active node below prediction");continue;}
            if(a==inactive)throw std::runtime_error("Missing active node");
            if(a<0){if(-a-1>=K||(!early&&depth[v]!=D))throw std::runtime_error("Invalid prediction or early stop");leaf[v]=1;}
            else{if(a>=F||depth[v]>=D||!allowed[size_t(v)*F+a])throw std::runtime_error("Forbidden split");
                if(no_repeat)for(int u=v;u;){u=(u-1)/2;if(tree[u]==a)throw std::runtime_error("Repeated path feature");}
                splitcost+=penalty+extras[size_t(v)*F+a];}}
        std::vector<std::vector<int>> support(threads,std::vector<int>(nodes()));double error=0;
        #pragma omp parallel for num_threads(threads) reduction(+:error) reduction(|:bad)
        for(int i=0;i<n;++i){int v=0;while(tree[v]>=0){v=2*v+1+X[size_t(i)*F+tree[v]];if(v>=nodes()){bad=1;break;}}
            if(v<nodes()){support[omp_get_thread_num()][v]++;if(-tree[v]-1!=y[i])error+=weights[i];}}
        if(bad)throw std::runtime_error("Recovery routing overflow");
        for(int v=0;v<nodes();++v)if(leaf[v]){int count=0;for(auto& h:support)count+=h[v];if(count<min_leaf)throw std::runtime_error("Leaf support violation");}
        return error+splitcost;
    }
    void tree_json(std::ostream& s,const std::vector<int>& t,int v=0)const{
        if(t[v]<0){s<<"{\"label\":"<<labels.at(-t[v]-1)<<"}";return;}
        s<<"{\"feature\":"<<t[v]<<",\"left\":";tree_json(s,t,2*v+1);s<<",\"right\":";tree_json(s,t,2*v+2);s<<"}";
    }
};

struct Pick{double value=inf;std::vector<int> ids;};
struct Table {
    int originalM,M,F,K,A,P,threads;bool ee;std::vector<double> cost;double ee_seconds=0;size_t finite_count=0;
    Table(int m,int f,int k,const double* high,const double* low,int th,bool elimination,Budget& budget):
        originalM(m),M(elimination?m/2:m),F(f),K(k),A(f+k),P(m==2?f+k:f*(f+k)+k),threads(th),ee(elimination){
        if((m!=2&&m!=4)||f<1||k<1||!high||th<1||int64_t(m)*P>100000000)throw std::invalid_argument("Unsupported exact table dimensions");
        budget.check();int invalid=0;
        #pragma omp parallel for num_threads(threads) reduction(|:invalid)
        for(int i=0;i<m*P;++i){double h=high[i];if(std::isnan(h)||h<0||(low&&(std::isnan(low[i])||low[i]!=h)))invalid=1;}
        if(invalid)throw std::invalid_argument("Incomplete, negative, or NaN exact-cost table");
        auto t=Clock::now();cost.resize(size_t(M)*P);
        #pragma omp parallel for num_threads(threads)
        for(int i=0;i<M*P;++i){int q=i/P,s=i%P;cost[i]=ee?high[(2*q)*P+s]+high[(2*q+1)*P+s]:high[i];}
        if(ee)ee_seconds=elapsed(t);
        for(double c:cost)finite_count+=std::isfinite(c);budget.check();
    }
    int root(int s)const{return originalM==2?s:(s<F*A?s/A:F+s-F*A);}
    int rows()const{return M==1?0:(M==2?(ee?A:P):2*P+A);}
    int degree(int q)const{return M==1?0:(M==2||q==0||q==3?1:2);}
    int edge(int q,int s,int slot)const{if(M==2)return ee?root(s):s;if(q==0)return s;if(q==1)return slot==0?s:P+root(s);if(q==2)return slot==0?P+root(s):P+A+s;return P+A+s;}
    double sign(int q,int slot)const{return q==0?1.:(q==M-1?-1.:(slot==0?-1.:1.));}
    bool eligible(int i,const std::vector<uint8_t>* active)const{return std::isfinite(cost[i])&&(!active||(*active)[i]);}
    Pick pick(const std::vector<uint8_t>* active=nullptr,bool first=false)const{
        // Root blocks are independent; each worker owns its minima/backpointers.
        int blocks=originalM==2?P:A;std::vector<Pick> per(blocks);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int r=0;r<blocks;++r){int lo=originalM==2?r:(r<F?r*A:F*A+r-F),hi=originalM==2?r+1:(r<F?lo+A:lo+1);
            if(M==1){for(int s=lo;s<hi;++s)if(eligible(s,active)){if(per[r].ids.empty()||(!first&&cost[s]<per[r].value)){per[r]={cost[s],{s}};}}}
            else if(M==2&&!ee){for(int s=lo;s<hi;++s)if(eligible(s,active)&&eligible(P+s,active)){double v=cost[s]+cost[P+s];if(per[r].ids.empty()||(!first&&v<per[r].value))per[r]={v,{s,P+s}};}}
            else{double lv=inf,rv=inf;int ls=-1,rs=-1;
                for(int s=lo;s<hi;++s){bool l=eligible(s,active)&&(M==2||eligible(P+s,active));int ri=(M==2?1:2)*P+s;bool rr=eligible(ri,active)&&(M==2||eligible(3*P+s,active));
                    double lval=l?cost[s]+(M==4?cost[P+s]:0):inf,rval=rr?cost[ri]+(M==4?cost[3*P+s]:0):inf;
                    if(l&&(ls<0||(!first&&lval<lv))){ls=s;lv=lval;}if(rr&&(rs<0||(!first&&rval<rv))){rs=s;rv=rval;}}
                if(ls>=0&&rs>=0)per[r]=M==2?Pick{lv+rv,{ls,P+rs}}:Pick{lv+rv,{ls,P+ls,2*P+rs,3*P+rs}};}}
        Pick out;for(auto& p:per)if(!p.ids.empty()&&(out.ids.empty()||(!first&&p.value<out.value)))out=p;return out;
    }
    std::vector<int> originals(const Pick& p)const{if(!ee)return p.ids;std::vector<int> ids;for(int id:p.ids){int q=id/P,s=id%P;ids.push_back(2*q*P+s);ids.push_back((2*q+1)*P+s);}return ids;}
};

struct Result {
    std::string status="TIME",coordinator;Pick pick;double lb=0,setup=0,update=0,rmp=0,pricing=0,message=0,recovery=0,ee=0,total=0;
    int iterations=0,columns=0,rows=0,nonzeros=0,pricing_calls=0;double residual=0,min_active_rc=0;bool capacity=false;
    std::vector<int> originals;
    void fields(std::ostream& s)const{s<<std::setprecision(17)<<"\"status\":\""<<status<<"\",\"coordinator\":\""<<coordinator<<"\",\"LB\":";num(s,lb);s<<",\"UB\":";num(s,pick.value);
        s<<",\"absolute_gap\":";num(s,pick.value-lb);s<<",\"selected_original_ids\":[";for(size_t i=0;i<originals.size();++i){if(i)s<<",";s<<originals[i];}s<<"]"
         <<",\"ee_seconds\":"<<ee<<",\"rmp_setup_seconds\":"<<setup<<",\"rmp_update_seconds\":"<<update<<",\"rmp_seconds\":"<<rmp
         <<",\"pricing_seconds\":"<<pricing<<",\"message_seconds\":"<<message<<",\"recovery_seconds\":"<<recovery<<",\"coordination_seconds\":"<<total
         <<",\"iterations\":"<<iterations<<",\"pricing_calls\":"<<pricing_calls<<",\"peak_active_columns\":"<<columns<<",\"submitted_rows\":"<<rows
         <<",\"peak_nonzeros\":"<<nonzeros<<",\"max_equality_residual\":"<<residual<<",\"minimum_active_reduced_cost\":"<<min_active_rc
         <<",\"column_capacity_triggered\":"<<(capacity?"true":"false")<<",\"certificate_full_domain\":"<<(status=="OPT"?"true":"false");}
};

inline Result coordinate(Table& t,int method,int seed_label,Budget& budget,int cap=2000000,int presolve=1,int lp_method=0){
    // 0=LP, 1=eager CG, 2=complete MP. EE is fixed in Table, not a second oracle.
    if(method<0||method>2||cap<1)throw std::invalid_argument("Invalid coordinator options");
    auto started=Clock::now();Result o;o.coordinator=method==0?"lp":(method==1?"eager_cg":"mp");o.ee=t.ee_seconds;
    o.pick=t.pick(nullptr,true); // Feasibility-only start, never the full-cost optimum.
    if(o.pick.ids.empty()){o.status="INFEASIBLE";o.lb=inf;o.total=elapsed(started);return o;}
    if(seed_label>=0){int s=t.originalM==2?t.F+seed_label:t.F*t.A+seed_label;Pick seed;seed.value=0;
        if(seed_label<t.K&&s<t.P){for(int q=0;q<t.M;++q){int id=q*t.P+s;seed.ids.push_back(id);seed.value+=t.cost[id];}if(std::isfinite(seed.value))o.pick=seed;}}
    try{budget.check();
        if(method==2||t.M==1){auto tick=Clock::now();o.pick=t.pick();o.message=elapsed(tick);budget.check();o.lb=o.pick.value;o.status="OPT";if(method!=2)o.coordinator="single_cluster_direct_min";}
        else{auto tick=Clock::now();GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();GRBModel model(env);
            model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260917);model.set(GRB_IntParam_Method,lp_method);model.set(GRB_IntParam_Presolve,presolve);
            model.set(GRB_IntParam_DualReductions,0);model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
            std::vector<GRBConstr> norms,sep(t.rows());std::vector<uint8_t> row(t.rows(),0),active(t.cost.size(),0);std::vector<GRBVar> vars(t.cost.size());
            for(int q=0;q<t.M;++q)norms.push_back(model.addConstr(GRBLinExpr()==1.));o.setup=elapsed(tick);
            std::vector<int> pending;if(method==0){for(int i=0;i<int(t.cost.size());++i)if(std::isfinite(t.cost[i]))pending.push_back(i);}else pending=o.pick.ids;
            while(true){budget.check();if(o.columns+int(pending.size())>cap){o.capacity=true;o.status="COLUMN_LIMIT";break;}tick=Clock::now();
                for(int id:pending)for(int slot=0;slot<t.degree(id/t.P);++slot){int e=t.edge(id/t.P,id%t.P,slot);if(!row[e]){sep[e]=model.addConstr(GRBLinExpr()==0.);row[e]=1;}}model.update();
                for(int id:pending)if(!active[id]){int q=id/t.P,s=id%t.P;GRBColumn col;col.addTerm(1.,norms[q]);for(int slot=0;slot<t.degree(q);++slot)col.addTerm(t.sign(q,slot),sep[t.edge(q,s,slot)]);
                    vars[id]=model.addVar(0.,GRB_INFINITY,t.cost[id],GRB_CONTINUOUS,col);active[id]=1;}model.update();o.update+=elapsed(tick);
                o.columns=model.get(GRB_IntAttr_NumVars);o.rows=model.get(GRB_IntAttr_NumConstrs);o.nonzeros=model.get(GRB_IntAttr_NumNZs);budget.check();
                model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,budget.remaining()));tick=Clock::now();model.optimize();o.rmp+=elapsed(tick);o.iterations++;
                int status=model.get(GRB_IntAttr_Status);if(status!=GRB_OPTIMAL){o.status=status==GRB_TIME_LIMIT?"TIME":"ERROR";break;}
                tick=Clock::now();Pick current=t.pick(&active);o.recovery+=elapsed(tick);double z=model.get(GRB_DoubleAttr_ObjVal);
                if(std::abs(current.value-z)>1e-7)throw std::runtime_error("RMP objective/recovery mismatch");if(current.value<o.pick.value)o.pick=current;
                std::vector<double> alpha(t.M),pi(t.rows(),0);for(int q=0;q<t.M;++q){alpha[q]=norms[q].get(GRB_DoubleAttr_Pi);o.residual=std::max(o.residual,std::abs(norms[q].get(GRB_DoubleAttr_Slack)));}
                for(int e=0;e<t.rows();++e)if(row[e]){pi[e]=sep[e].get(GRB_DoubleAttr_Pi);o.residual=std::max(o.residual,std::abs(sep[e].get(GRB_DoubleAttr_Slack)));}
                for(int id=0;id<int(active.size());++id)if(active[id])o.min_active_rc=std::min(o.min_active_rc,vars[id].get(GRB_DoubleAttr_RC));
                if(o.residual>1e-7||o.min_active_rc < -1e-7)throw std::runtime_error("LP primal/dual audit failed");
                // Full-domain pricing certifies both LP and CG; omitted rows have zero dual.
                tick=Clock::now();std::vector<double> rc(t.cost.size(),inf),minimum(t.M,inf);
                #pragma omp parallel for num_threads(t.threads) schedule(static)
                for(int q=0;q<t.M;++q){for(int s=0;s<t.P;++s){int id=q*t.P+s;if(!std::isfinite(t.cost[id]))continue;double v=t.cost[id]-alpha[q];for(int slot=0;slot<t.degree(q);++slot)v-=t.sign(q,slot)*pi[t.edge(q,s,slot)];rc[id]=v;minimum[q]=std::min(minimum[q],v);}}
                double bound=std::accumulate(alpha.begin(),alpha.end(),0.);for(double v:minimum)bound+=std::min(0.,v);
                if(bound>o.pick.value+1e-7)throw std::runtime_error("Full-domain LB exceeds feasible UB");o.lb=std::max(o.lb,bound);
                pending.clear();for(int id=0;id<int(rc.size());++id)if(!active[id]&&rc[id]<-1e-9)pending.push_back(id);o.pricing+=elapsed(tick);o.pricing_calls++;
                budget.check();if(o.pick.value-o.lb<=1e-7){o.status="OPT";break;}
                if(pending.empty())throw std::runtime_error("Pricing stalled with an open certificate");
            }
        }
    }catch(const Timed&){o.status="TIME";}catch(const Limited&){o.status="MEM";}
    o.originals=t.originals(o.pick);o.total=elapsed(started)+t.ee_seconds;return o;
}
} // namespace revision
