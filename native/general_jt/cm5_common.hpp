// CM-JT-CG production core. Python passes data/options and audits the result.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <intrin.h>
#include <omp.h>
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
namespace cm5 {
using U=std::uint64_t;
using Clock=std::chrono::steady_clock;
inline double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
constexpr double INF=std::numeric_limits<double>::infinity();
constexpr int ABSENT=-99, SLOTS=64;
struct Options {
    int version=1, depth=5, metric=0, threads=8, cells=8, refinements=2, cache_entries=256, min_leaf=1;
    int max_nodes=100000, memory_mb=8192;
    double penalty=0, beta=1, seconds=60, tolerance=1e-7, node_seconds=.5, request_seconds=.01, heuristic_seconds=2;
};
struct Tree {
    std::array<int,SLOTS> a;
    Tree(){a.fill(ABSENT);a[1]=-1;}
};
struct Counts {int tp=0,tn=0,k=0;};
struct Box {
    int tl=0,th=0,ul=0,uh=0;
    bool contains(int t,int u)const{return tl<=t&&t<=th&&ul<=u&&u<=uh;}
    bool valid()const{return tl<=th&&ul<=uh;}
    int width()const{return std::max(th-tl,uh-ul);}
    bool operator<(const Box&b)const{return std::tie(tl,th,ul,uh)<std::tie(b.tl,b.th,b.ul,b.uh);}
};
inline Box intersect(Box a,Box b){return {std::max(a.tl,b.tl),std::min(a.th,b.th),std::max(a.ul,b.ul),std::min(a.uh,b.uh)};}
inline Box difference(Box a,Box b){return {a.tl-b.th,a.th-b.tl,a.ul-b.uh,a.uh-b.ul};}
inline std::pair<Box,Box> split(Box b){Box a=b; if(b.th-b.tl>=b.uh-b.ul){int m=(b.tl+b.th)/2;a.th=m;b.tl=m+1;}else{int m=(b.ul+b.uh)/2;a.uh=m;b.ul=m+1;}return {a,b};}
struct Metric {
    int id,P,N;double beta;
    std::array<double,8> weights{};
    double ceiling()const{return id==8?std::accumulate(weights.begin(),weights.end(),0.):1.;}
    double score(int t,int u)const {
        if(id==8){double value=0;for(int i=0;i<8;++i)if(weights[i])value+=weights[i]*Metric{i,P,N,beta}.score(t,u);return value;}
        double p=P,n=N,q=double(t)+N-u;
        switch(id){
        case 0:case 1:{double b=id==1?beta*beta:1.;return (1+b)*t/(b*p+n+t-u);}
        case 2:return q>0&&q<p+n?(n*t+p*u-p*n)/std::sqrt(p*n*q*(p+n-q)):0.;
        case 3:return std::sqrt(double(t)*u/(p*n));
        case 4:return q>0?t/std::sqrt(p*q):0.;
        case 5:return t/(p+n-u);
        case 6:return (t+u)/(p+n);
        default:return .5*(t/p+u/n);
        }
    }
    double upper(Box b)const{return std::min(ceiling(),score(b.th,b.uh)+1e-12*std::max(1.,ceiling()));}
    const char* name()const {static const char* names[]={"f1","fbeta","mcc","gmean","fmi","jaccard","accuracy","balanced_accuracy","weighted_sum"};return names[id];}
};
struct Mask {std::vector<U> bits;int p=0,n=0;std::string key;};
using Rows=std::shared_ptr<Mask>;
struct Data {
    int n,F,W,min_leaf,P=0,N=0,conflict_correct=0;
    const unsigned char *X,*y;
    std::vector<U> zero,pos;
    Rows all;
    std::vector<std::array<Rows,2>> root_rows;
    Data(const unsigned char*x,const unsigned char*labels,int samples,int features,int ml):n(samples),F(features),W((n+63)/64),min_leaf(ml),X(x),y(labels),zero(size_t(F)*W),pos(W){
        std::vector<U> full(W);std::unordered_map<std::string,std::array<int,2>> groups;
        for(int i=0;i<n;++i){
            if(y[i]>1)throw std::invalid_argument("binary y required");
            full[i/64]|=U(1)<<(i%64);if(y[i]){pos[i/64]|=U(1)<<(i%64);++P;}else ++N;
            for(int f=0;f<F;++f){if(X[size_t(i)*F+f]>1)throw std::invalid_argument("binary X required");if(!X[size_t(i)*F+f])zero[size_t(f)*W+i/64]|=U(1)<<(i%64);}
            ++groups[std::string(reinterpret_cast<const char*>(X+size_t(i)*F),F)][y[i]];
        }
        if(!P||!N)throw std::invalid_argument("both classes required");
        for(auto&g:groups)conflict_correct+=std::max(g.second[0],g.second[1]);
        all=mask(full);root_rows.resize(F);
        for(int f=0;f<F;++f)root_rows[f]=partition(all,f);
    }
    Rows mask(const std::vector<U>&v)const {
        auto r=std::make_shared<Mask>();r->bits=v;
        for(int w=0;w<W;++w){r->p+=int(__popcnt64(v[w]&pos[w]));r->n+=int(__popcnt64(v[w]&~pos[w]));}
        r->key.assign(reinterpret_cast<const char*>(v.data()),size_t(W)*sizeof(U));return r;
    }
    std::array<Rows,2> partition(const Rows&r,int f)const {
        std::vector<U> a(W),b(W);for(int w=0;w<W;++w){a[w]=r->bits[w]&zero[size_t(f)*W+w];b[w]=r->bits[w]^a[w];}return {mask(a),mask(b)};
    }
    Counts counts(const Tree&t,const Rows&r,int node=1,int remaining=5)const {
        if(node>=SLOTS)throw std::runtime_error("tree exceeds depth five");
        int f=t.a[node];if(f==-1)return {0,r->n,0};if(f==-2)return {r->p,0,0};
        if(f<0||f>=F||remaining<=0)throw std::runtime_error("invalid tree action");
        auto child=partition(r,f);
        if(child[0]->p+child[0]->n<min_leaf||child[1]->p+child[1]->n<min_leaf)throw std::runtime_error("minimum leaf violation");
        auto a=counts(t,child[0],node*2,remaining-1),b=counts(t,child[1],node*2+1,remaining-1);
        return {a.tp+b.tp,a.tn+b.tn,1+a.k+b.k};
    }
};
// Optional objective-independent support cuts inside the count-CG engine.
// The owner updates these only between frozen-dual pricing batches.
struct GenericHook {
    virtual ~GenericHook()=default;
    virtual void advance(double end)=0;
    virtual bool pending()const=0;
    virtual double lower(Box box)const=0;
    virtual double pricing(const Rows&rows,int depth,Box box,double penalty)const=0;
    virtual std::string diagnostics()const=0;
};
inline void transplant(Tree&target,int dst,const Tree&source,int src=1){
    if(dst>=SLOTS||src>=SLOTS)throw std::runtime_error("tree transplant overflow");
    target.a[dst]=source.a[src];if(source.a[src]>=0){transplant(target,dst*2,source,src*2);transplant(target,dst*2+1,source,src*2+1);}
}
inline std::string tree_json(const Tree&t,int node=1){
    if(t.a[node]<0)return "{\"label\":"+std::to_string(-t.a[node]-1)+"}";
    return "{\"feature\":"+std::to_string(t.a[node])+",\"left\":"+tree_json(t,node*2)+",\"right\":"+tree_json(t,node*2+1)+"}";
}
inline size_t peak_memory(){PROCESS_MEMORY_COUNTERS p{};GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p));return p.PeakWorkingSetSize;}
struct Profile {Counts c;Tree tree;};
struct Counters {U nodes=0,prunes=0,hits=0,misses=0,evictions=0;double search_work=0,order_work=0;};
struct Choice {int f,p0,n0,gain;};
using Order=std::vector<Choice>;
// Each worker owns its caches and counters. Data and root masks are immutable.
struct Context {
    const Data& data;Counters stats;
    std::unordered_map<std::string,std::shared_ptr<Order>> orders;
    bool incremental_cache=false;
    size_t order_capacity=4096;
    std::deque<std::string> order_fifo;
    explicit Context(const Data&d):data(d){}
    std::shared_ptr<Order> order(const Rows&r){
        auto found=orders.find(r->key);if(found!=orders.end()){++stats.hits;return found->second;}
        double started=now();++stats.misses;auto out=std::make_shared<Order>();
        // Equivalence is compared only within this fixed routed mask. Omitting
        // its zero words preserves both partitions and canonical ordering.
        std::vector<int> active;active.reserve(data.W);for(int w=0;w<data.W;++w)if(r->bits[w])active.push_back(w);
        std::vector<U> a(active.size()),b(active.size());std::unordered_set<std::string> seen;
        for(int f=0;f<data.F;++f){int p=0,n=0;
            for(size_t j=0;j<active.size();++j){int w=active[j];a[j]=r->bits[w]&data.zero[size_t(f)*data.W+w];b[j]=r->bits[w]^a[j];p+=int(__popcnt64(a[j]&data.pos[w]));n+=int(__popcnt64(a[j]&~data.pos[w]));}
            int left=p+n,right=r->p+r->n-left;if(!left||!right||std::min(left,right)<data.min_leaf)continue;
            const auto&canonical=std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end())?a:b;
            if(!seen.insert(std::string(reinterpret_cast<const char*>(canonical.data()),canonical.size()*sizeof(U))).second)continue;
            out->push_back({f,p,n,std::min(r->p,r->n)-std::min(p,n)-std::min(r->p-p,r->n-n)});
        }
        std::sort(out->begin(),out->end(),[](auto a,auto b){return a.gain!=b.gain?a.gain>b.gain:a.f<b.f;});
        if(orders.size()>=order_capacity){
            if(incremental_cache){orders.erase(order_fifo.front());order_fifo.pop_front();}
            else orders.clear();
            ++stats.evictions;
        }
        orders.emplace(r->key,out);if(incremental_cache)order_fifo.push_back(r->key);
        stats.order_work+=now()-started;return out;
    }
};
struct Task {Rows rows;int depth,node;};
struct State {std::vector<Task> tasks;Counts c;Tree tree;};
struct Frame {State state;int next=-1;std::shared_ptr<Order> order;explicit Frame(State s):state(std::move(s)){};};
// Exhaustive and resumable count-rectangle pricing, iterative split budgets.
// Its lower bound is the first unexcluded budget, including while interrupted.
struct Search {
    Rows rows;int depth,budget=0,max_k;Box box;bool complete=false,found=false;
    Profile best;std::vector<Frame> stack;
    Search(Rows r,int d,Box b):rows(std::move(r)),depth(d),max_k((1<<d)-1),box(b){start();}
    void start(){State s;s.tasks.push_back({rows,depth,1});stack.emplace_back(std::move(s));}
    bool valid(const State&s)const {
        if(s.c.tp>box.th||s.c.tn>box.uh)return false;
        int p=0,n=0,correct=0;
        for(const auto&t:s.tasks){p+=t.rows->p;n+=t.rows->n;correct+=(t.depth==0||s.c.k==budget)?std::max(t.rows->p,t.rows->n):t.rows->p+t.rows->n;}
        return s.c.tp+p>=box.tl&&s.c.tn+n>=box.ul&&s.c.tp+s.c.tn+correct>=box.tl+box.ul;
    }
    void advance(Context&ctx,double end,U steps=std::numeric_limits<U>::max()){
        double started=now();U visits=0;
        while(!complete&&visits<steps){
            if((visits&255)==0&&now()>=end)break;
            if(stack.empty()){if(++budget>max_k){complete=true;break;}start();}
            Frame&frame=stack.back();State&s=frame.state;
            if(frame.next<0){++visits;++ctx.stats.nodes;
                if(!valid(s)){++ctx.stats.prunes;stack.pop_back();continue;}
                if(s.tasks.empty()){complete=found=true;best={s.c,s.tree};break;}
                if(s.tasks.back().rows->p+s.tasks.back().rows->n<ctx.data.min_leaf){stack.pop_back();continue;}
                frame.next=0;
            }
            Task task=s.tasks.back();
            if(frame.next<2){bool positive=box.tl-s.c.tp>box.ul-s.c.tn;int label=frame.next++==0?int(positive):int(!positive);
                State child=s;child.tasks.pop_back();child.c.tp+=label?task.rows->p:0;child.c.tn+=label?0:task.rows->n;child.tree.a[task.node]=-label-1;stack.emplace_back(std::move(child));continue;}
            if(!task.depth||s.c.k>=budget){stack.pop_back();continue;}
            if(!frame.order)frame.order=ctx.order(task.rows);
            size_t i=size_t(frame.next++-2);if(i>=frame.order->size()){stack.pop_back();continue;}
            int f=(*frame.order)[i].f;auto childrows=ctx.data.partition(task.rows,f);
            State child=s;child.tasks.pop_back();++child.c.k;child.tree.a[task.node]=f;
            child.tasks.push_back({childrows[1],task.depth-1,task.node*2+1});child.tasks.push_back({childrows[0],task.depth-1,task.node*2});stack.emplace_back(std::move(child));
        }
        ctx.stats.search_work+=now()-started;
    }
    double lower(double penalty)const{return complete&&!found?INF:penalty*budget;}
};
}
