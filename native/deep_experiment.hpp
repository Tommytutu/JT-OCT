// End-to-end native campaign. Python passes arrays and dispatches CUDA kernels;
// no prefix enumeration, cost minimization, pricing, or tree search lives there.
// Scope: uniform weights 1/n, lambda=0, STOP, no repeated predicate, min_leaf=0.
#pragma once
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <intrin.h>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <unordered_set>

extern "C" void* d3_opt_create(int,int);
extern "C" void d3_opt_free(void*);
extern "C" int d3_opt_costs_multiclass(void*,const uint64_t*,const uint64_t*,const double*,
    const unsigned char*,const double*,const unsigned char*,const int*,double,double*,int*,unsigned char*);

// The callback does only host/device transfers, kernel launches, and synchronization.
struct DSGpuBatch {
    int phase,B,F,W,K,globalF,n,max_words;
    const uint64_t *zero,*labels,*masks;
    const int *indices,*nw,*features;
    const double *costs,*floors;
    const unsigned char *allowed,*dominated;
    double* output; int* tail; unsigned char* reasons;
    double weight,remaining;
};
using DSGpuProvider=int(*)(DSGpuBatch*);

namespace deep_campaign {
using Bits=std::vector<uint64_t>;
using Clock=std::chrono::steady_clock;
inline double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct TimedOut:std::runtime_error{TimedOut():std::runtime_error("wall-clock deadline") {}};
struct Resource:std::runtime_error{std::string status;Resource(std::string s,std::string m):std::runtime_error(m),status(s){}};
struct Timer{double& value;Clock::time_point begin=Clock::now();explicit Timer(double& v):value(v){}~Timer(){value+=elapsed(begin);}};
struct Node {int f=-1,k=0;std::shared_ptr<Node> l,r;};
using Tree=std::shared_ptr<Node>;
struct Group {Bits rows;double lo=0,hi=0;bool exact=false;Tree tree;};
struct Options {int threads,batch,cap,compact,message,columns_per_block;uint64_t memory;};
struct OracleInput {
    int gid,F=0,n=0;std::vector<int> features;std::vector<double> side;
    std::vector<unsigned char> dominated;
};
struct Solution {int status=0,columns=0,rows=0,nonzeros=0;double objective=INF,residual=0;
    std::vector<double> alpha,pi;std::vector<int> selected;};

struct Engine {
    const uint8_t* X;int n,F,D,K=0,W=0,h=0,M=0,P=0,B=0,first=0;
    bool ee,full;Options o;DSGpuProvider gpu;
    Clock::time_point started=Clock::now();double seconds,lb=0,ub=INF,proof=-1;
    std::string status="TIME",reason;
    std::vector<int64_t> labels;std::vector<int> y;Bits zero,class_bits,all;
    std::vector<std::pair<int,int>> conflicts;
    std::vector<Tree> leaves;Tree incumbent;
    std::unique_ptr<Shape> shape;std::vector<Group> groups;
    std::vector<int> private_id;
    std::vector<double> low,high,cost_low,cost_high;
    std::map<std::string,double> stats;
    struct Event{double seconds,lb,ub;int columns;};std::vector<Event> trace;
    std::mutex group_mutex;std::unordered_map<uint64_t,std::vector<int>> group_lookup;
    uint64_t peak_private=0,peak_rss=0;

    Engine(const uint8_t* x,const int64_t* raw_y,int nn,int ff,int dd,bool elimination,bool eager,
           Options opt,double limit,DSGpuProvider provider):
        X(x),n(nn),F(ff),D(dd),ee(elimination),full(eager),o(opt),gpu(provider),seconds(limit){
        if(!X||!raw_y||n<1||F<1||D<5||D>7||seconds<0||!std::isfinite(seconds)||
           o.threads<1||o.batch<1||o.cap<1||o.compact<0||o.columns_per_block<0||!o.memory)
            throw std::invalid_argument("Invalid native campaign dimensions/options");
        labels.assign(raw_y,raw_y+n);std::sort(labels.begin(),labels.end());
        labels.erase(std::unique(labels.begin(),labels.end()),labels.end());
        if(labels.front()<0)throw std::invalid_argument("Labels must be nonnegative");
        K=int(labels.size());y.resize(n);std::vector<int> hist(K);
        for(int i=0;i<n;++i){y[i]=int(std::lower_bound(labels.begin(),labels.end(),raw_y[i])-labels.begin());++hist[y[i]];}
        for(int k=0;k<K;++k){auto t=std::make_shared<Node>();t->k=k;leaves.push_back(t);}
        int best=int(std::max_element(hist.begin(),hist.end())-hist.begin());
        incumbent=leaves[best];ub=double(n-hist[best])/n;
        for(const auto* key:{"data_preparation_seconds","metadata_seconds","initialization_seconds",
            "backend_initialization_seconds","oracle_seconds","ee_preprocessing_seconds",
            "rmp_setup_seconds","rmp_update_seconds","rmp_seconds","rmp_recovery_seconds",
            "pricing_seconds","message_seconds","recovery_seconds"})stats[key]=0;
        stats["data_preparation_seconds"]=elapsed(started);
        omp_set_num_threads(o.threads);
    }
    void memory_check(){
        PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof(m);
        if(!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof(m)))
            throw std::runtime_error("Cannot read process memory");
        peak_private=std::max(peak_private,uint64_t(m.PrivateUsage));peak_rss=std::max(peak_rss,uint64_t(m.WorkingSetSize));
        if(peak_private>o.memory)throw Resource("MEM","Process private committed memory limit");
    }
    void check()const{if(elapsed(started)>=seconds)throw TimedOut();}
    double remaining()const{check();return std::max(1e-6,seconds-elapsed(started));}
    static int pop(uint64_t v){return int(__popcnt64(v));}
    static uint64_t hash(const Bits& bits){uint64_t v=1469598103934665603ULL;for(auto b:bits){v^=b;v*=1099511628211ULL;}return v;}
    Bits route(const Bits& rows,int f,int side)const{
        Bits out(W);for(int w=0;w<W;++w)out[w]=rows[w]&(side?~zero[size_t(f)*W+w]:zero[size_t(f)*W+w]);return out;
    }
    std::vector<int> counts(const Bits& rows)const{
        std::vector<int> hist(K);for(int k=0;k<K;++k)for(int w=0;w<W;++w)hist[k]+=pop(rows[w]&class_bits[size_t(k)*W+w]);return hist;
    }
    std::pair<Tree,double> leaf(const Bits& rows)const{
        auto hist=counts(rows);int k=int(std::max_element(hist.begin(),hist.end())-hist.begin());
        return {leaves[k],double(std::accumulate(hist.begin(),hist.end(),0)-hist[k])/n};
    }
    int size(const Bits& rows)const{int total=0;for(auto w:rows)total+=pop(w);return total;}
    Tree split(int f,Tree l,Tree r)const{auto t=std::make_shared<Node>();t->f=f;t->l=l;t->r=r;return t;}
    void prepare_data(){
        Timer timer(stats["data_preparation_seconds"]);check();W=(n+63)/64;h=D-3;M=1<<h;
        shape=std::make_unique<Shape>(F,K,h);P=shape->states();B=M-(ee?2:0);first=ee?1:0;
        if(int64_t(M)*P>INT32_MAX)throw Resource("INDEX_LIMIT","Original column ids exceed int32");
        // These are mandatory arrays, not an empirical estimate of a full CG master.
        uint64_t required=uint64_t(M)*P*(2*sizeof(double)+sizeof(int));
        if(required>o.memory)throw Resource("MEM_REQUIRED","Mandatory dense state arrays alone exceed memory limit");
        all.assign(W,~uint64_t(0));if(n%64)all.back()=(uint64_t(1)<<(n%64))-1;
        zero.assign(size_t(F)*W,0);class_bits.assign(size_t(K)*W,0);
        std::atomic<bool> invalid{false};
        #pragma omp parallel for schedule(static) num_threads(o.threads)
        for(int w=0;w<W;++w){
            for(int i=w*64;i<std::min(n,w*64+64);++i){
                class_bits[size_t(y[i])*W+w]|=uint64_t(1)<<(i%64);
                for(int f=0;f<F;++f){if(X[size_t(i)*F+f]>1)invalid=true;
                    if(!X[size_t(i)*F+f])zero[size_t(f)*W+w]|=uint64_t(1)<<(i%64);}
            }
        }
        if(invalid)throw std::invalid_argument("Features must be binary");
        // Exact conflict lower bound, keyed by complete sample feature pattern.
        struct Pattern{int row,total=0;std::vector<int> hist;};
        std::unordered_map<std::string,Pattern> patterns;
        for(int i=0;i<n;++i){if(!(i%1024))check();std::string key(reinterpret_cast<const char*>(X+size_t(i)*F),F);
            auto it=patterns.find(key);if(it==patterns.end())it=patterns.emplace(std::move(key),Pattern{i,0,std::vector<int>(K)}).first;
            ++it->second.total;++it->second.hist[y[i]];
        }
        for(auto& entry:patterns){auto& p=entry.second;int loss=p.total-*std::max_element(p.hist.begin(),p.hist.end());
            if(loss){conflicts.emplace_back(p.row,loss);lb+=double(loss)/n;}}
        std::sort(conflicts.begin(),conflicts.end());stats["conflict_lower_bound"]=lb;
        memory_check();check();
    }
    Tree greedy(const Bits& rows,int depth,std::vector<int>& used){
        check();auto stop=leaf(rows);if(!depth||stop.second==0)return stop.first;
        int feature=-1;double best=stop.second;
        for(int f=0;f<F;++f){if(std::find(used.begin(),used.end(),f)!=used.end())continue;
            double value=leaf(route(rows,f,0)).second+leaf(route(rows,f,1)).second;
            if(value<best-1e-12){best=value;feature=f;}}
        if(feature<0)return stop.first;
        used.push_back(feature);auto l=greedy(route(rows,feature,0),depth-1,used),r=greedy(route(rows,feature,1),depth-1,used);
        used.pop_back();return split(feature,l,r);
    }
    double audit(Tree t,const Bits& rows,int budget,std::vector<int> used={})const{
        if(!t)throw std::runtime_error("Missing tree");
        if(t->f<0){if(t->k<0||t->k>=K)throw std::runtime_error("Invalid leaf label");
            int right=0,total=0;for(int w=0;w<W;++w){total+=pop(rows[w]);right+=pop(rows[w]&class_bits[size_t(t->k)*W+w]);}return double(total-right)/n;}
        if(!budget||t->f>=F||std::find(used.begin(),used.end(),t->f)!=used.end())throw std::runtime_error("Invalid tree depth/repeated predicate");
        used.push_back(t->f);return audit(t->l,route(rows,t->f,0),budget-1,used)+audit(t->r,route(rows,t->f,1),budget-1,used);
    }
    int group_for(Bits rows){
        auto code=hash(rows);
        {std::lock_guard<std::mutex> lock(group_mutex);
            auto found=group_lookup.find(code);if(found!=group_lookup.end())
                for(int id:found->second)if(groups[id].rows==rows)return id;}
        // Expensive counts are parallel; only lookup/insertion is serialized.
        auto hist=counts(rows);int total=std::accumulate(hist.begin(),hist.end(),0);
        int k=int(std::max_element(hist.begin(),hist.end())-hist.begin()),conflict=0;
        for(auto [row,loss]:conflicts)if(rows[row/64]&(uint64_t(1)<<(row%64)))conflict+=loss;
        std::sort(hist.begin(),hist.end(),std::greater<int>());
        int covered=0;for(int c=0;c<std::min(K,8);++c)covered+=hist[c];
        Group g;g.rows=std::move(rows);g.lo=double(std::max(conflict,total-covered))/n;
        g.hi=double(total-hist[0])/n;g.tree=leaves[k];g.exact=g.hi-g.lo<=1e-12;
        std::lock_guard<std::mutex> lock(group_mutex);auto& bucket=group_lookup[code];
        for(int id:bucket)if(groups[id].rows==g.rows)return id;
        int id=int(groups.size());groups.push_back(std::move(g));bucket.push_back(id);
        if(!(id%256))memory_check();
        return id;
    }
    void prepare_states(){
        Timer timer(stats["metadata_seconds"]);size_t N=size_t(M)*P;
        low.assign(N,INF);high.assign(N,INF);private_id.assign(N,-1);
        std::exception_ptr failure;std::mutex error_mutex;std::atomic<bool> failed{false};
        #pragma omp parallel for schedule(dynamic,1) num_threads(o.threads)
        for(int q=0;q<M;++q){try{
            for(int s=0;s<P&&!failed;++s){if(!(s%512))check();
                int residual=s,used[4],nu=0;bool valid=true,stopped=false;Bits rows=all;
                for(int j=0;j<h;++j){int remaining=h-j,cut=F*int(shape->P[remaining-1]);
                    if(residual>=cut){int k=residual-cut;auto hist=counts(rows);double loss=double(std::accumulate(hist.begin(),hist.end(),0)-hist[k])/n;
                        low[size_t(q)*P+s]=high[size_t(q)*P+s]=loss/double(1<<(h-j));stopped=true;break;}
                    int f=residual/int(shape->P[remaining-1]);residual%=int(shape->P[remaining-1]);
                    if(std::find(used,used+nu,f)!=used+nu){valid=false;break;}used[nu++]=f;
                    rows=route(rows,f,(q>>(h-j-1))&1);
                }
                if(valid&&!stopped)private_id[size_t(q)*P+s]=group_for(std::move(rows));
            }
        }catch(...){failed=true;std::lock_guard<std::mutex> lock(error_mutex);if(!failure)failure=std::current_exception();}}
        if(failure)std::rethrow_exception(failure);
        group_lookup.clear();group_lookup.rehash(0);
        // Stable IDs independent of OpenMP insertion order.
        std::vector<int> min_id(groups.size(),INT32_MAX),order(groups.size()),remap(groups.size());
        for(size_t id=0;id<N;++id)if(private_id[id]>=0)min_id[private_id[id]]=std::min(min_id[private_id[id]],int(id));
        std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](int a,int b){return min_id[a]<min_id[b];});
        std::vector<Group> sorted;sorted.reserve(groups.size());
        for(int i=0;i<int(order.size());++i){remap[order[i]]=i;sorted.push_back(std::move(groups[order[i]]));}groups.swap(sorted);
        #pragma omp parallel for schedule(static) num_threads(o.threads)
        for(int64_t id=0;id<int64_t(N);++id)if(private_id[id]>=0){int g=remap[private_id[id]];private_id[id]=g;low[id]=groups[g].lo;high[id]=groups[g].hi;}
        stats["candidate_slots"]=double(N);stats["unique_terminal_states"]=double(groups.size());
        memory_check();check();
    }
    std::vector<int> tree_ids(Tree t,bool inject){
        std::vector<int> ids(M);for(int q=0;q<M;++q){Tree node=t;int s=0;
            for(int j=0;j<h;++j){int remaining=h-j;if(node->f<0){s+=F*int(shape->P[remaining-1])+node->k;break;}
                s+=node->f*int(shape->P[remaining-1]);node=((q>>(h-j-1))&1)?node->r:node->l;}
            int id=q*P+s;ids[q]=id;
            if(inject&&private_id[id]>=0){auto& g=groups[private_id[id]];double value=audit(node,g.rows,3);
                if(value<g.lo-1e-8)throw std::runtime_error("Initial tail violates lower bound");
                if(value<g.hi){g.hi=value;g.tree=node;}}
        }return ids;
    }
    void refresh(){
        // Mandatory costs remain complete, including evicted/unqueried signatures.
        #pragma omp parallel for schedule(static) num_threads(o.threads)
        for(int64_t id=0;id<int64_t(low.size());++id)if(private_id[id]>=0){const auto& g=groups[private_id[id]];low[id]=g.lo;high[id]=g.hi;}
        cost_low.resize(size_t(B)*P);cost_high.resize(size_t(B)*P);
        Timer timer(stats[ee?"ee_preprocessing_seconds":"pricing_seconds"]);
        #pragma omp parallel for schedule(static) num_threads(o.threads)
        for(int64_t id=0;id<int64_t(cost_low.size());++id){int q=int(id/P),s=int(id%P);size_t src=size_t(q+first)*P+s;
            double a=low[src],b=high[src];
            if(ee&&q==0){a+=low[s];b+=high[s];}
            if(ee&&q==B-1){a+=low[size_t(M-1)*P+s];b+=high[size_t(M-1)*P+s];}
            cost_low[id]=a;cost_high[id]=b;
        }
    }
    OracleInput oracle_input(int gid){
        OracleInput out;out.gid=gid;auto& rows=groups[gid].rows;out.n=size(rows);
        std::unordered_map<uint64_t,std::vector<Bits>> seen;
        for(int f=0;f<F;++f){Bits a=route(rows,f,0);int na=size(a);if(!na||na==out.n)continue;
            Bits complement(W);for(int w=0;w<W;++w)complement[w]=rows[w]^a[w];
            Bits key=std::min(a,complement);auto code=hash(key);bool found=false;
            for(auto& previous:seen[code])if(previous==key){found=true;break;}
            if(found)continue;seen[code].push_back(std::move(key));out.features.push_back(f);
        }
        out.F=int(out.features.size());out.side.resize(2*out.F);out.dominated.resize(2*out.F);
        for(int a=0;a<2;++a)for(int f=0;f<out.F;++f){double loss=leaf(route(rows,out.features[f],a)).second;
            out.side[a*out.F+f]=loss;out.dominated[a*out.F+f]=loss<=1e-15;}
        return out;
    }
    Tree join(const OracleInput& in,int stride,const double* cost,const int* tail){
        const auto& rows=groups[in.gid].rows;Tree answer=leaf(rows).first;double best=leaf(rows).second;
        int root=-1,chosen_g[2]={-1,-1};
        for(int f=0;f<in.F;++f){double total=0;int gs[2];
            for(int a=0;a<2;++a){double side=in.side[a*in.F+f];gs[a]=-1;
                for(int g=0;g<in.F;++g){double v=cost[(size_t(2*a)*stride+f)*stride+g]+cost[(size_t(2*a+1)*stride+f)*stride+g];
                    if(v<side-1e-12){side=v;gs[a]=g;}}
                total+=side;
            }
            if(total<best-1e-12){best=total;root=f;chosen_g[0]=gs[0];chosen_g[1]=gs[1];}
        }
        if(root>=0){Tree sides[2];for(int a=0;a<2;++a){Bits ra=route(rows,in.features[root],a);int g=chosen_g[a];
                if(g<0)sides[a]=leaf(ra).first;
                else{Tree children[2];for(int b=0;b<2;++b){Bits rb=route(ra,in.features[g],b);int q=2*a+b;
                        int action=tail[(size_t(q)*stride+root)*stride+g];
                        if(action==-1)children[b]=leaf(rb).first;
                        else if(action>=0&&action<in.F)children[b]=split(in.features[action],leaf(route(rb,in.features[action],0)).first,leaf(route(rb,in.features[action],1)).first);
                        else throw std::runtime_error("Invalid D3 oracle backpointer");}
                    sides[a]=split(in.features[g],children[0],children[1]);}}
            answer=split(in.features[root],sides[0],sides[1]);}
        double audited=audit(answer,rows,3);if(std::abs(audited-best)>1e-8)throw std::runtime_error("D3 join audit mismatch");return answer;
    }
    Tree cpu_oracle(const OracleInput& in){
        if(!in.F)return leaf(groups[in.gid].rows).first;
        int words=(in.n+63)/64;std::vector<uint64_t> z(size_t(in.F)*words),c(size_t(K)*words);
        std::vector<int> rows;rows.reserve(in.n);for(int w=0;w<W;++w){auto bits=groups[in.gid].rows[w];while(bits){unsigned long b;_BitScanForward64(&b,bits);rows.push_back(w*64+int(b));bits&=bits-1;}}
        for(int j=0;j<in.n;++j){int row=rows[j];c[size_t(y[row])*words+j/64]|=uint64_t(1)<<(j%64);
            for(int f=0;f<in.F;++f)if(!X[size_t(row)*F+in.features[f]])z[size_t(f)*words+j/64]|=uint64_t(1)<<(j%64);}
        std::vector<double> costs(7*in.F),floors(7),output(size_t(4)*in.F*in.F);
        std::vector<unsigned char> allowed(7*in.F,1),reasons(output.size());std::vector<int> tail(output.size());
        void* ws=d3_opt_create(words,1);if(!ws)throw std::bad_alloc();
        struct Cleanup{void* p;~Cleanup(){d3_opt_free(p);}}cleanup{ws};
        for(int start=0;start<in.F*in.F;start+=64){check();int ip[]={in.F,words,in.n,1,1,0,1,1,start,std::min(64,in.F*in.F-start),K};
            if(d3_opt_costs_multiclass(ws,z.data(),c.data(),costs.data(),allowed.data(),floors.data(),in.dominated.data(),ip,1./n,output.data(),tail.data(),reasons.data()))
                throw std::runtime_error("Native D3 cost failure");}
        return join(in,in.F,output.data(),tail.data());
    }
    void oracle(const std::vector<int>& ids){
        if(ids.empty())return;Timer timer(stats["oracle_seconds"]);check();memory_check();
        std::vector<OracleInput> inputs(ids.size());std::vector<Tree> answers(ids.size());
        std::exception_ptr failure;std::mutex mutex;
        #pragma omp parallel for schedule(dynamic,1) num_threads(o.threads)
        for(int i=0;i<int(ids.size());++i)try{check();inputs[i]=oracle_input(ids[i]);if(!gpu)answers[i]=cpu_oracle(inputs[i]);}
            catch(...){std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();}
        if(failure)std::rethrow_exception(failure);
        if(gpu){
            int batch=int(ids.size()),f=0;for(auto& in:inputs)f=std::max(f,in.F);
            if(f){
                Bits masks(size_t(batch)*W);std::vector<int> indices(size_t(batch)*W),nw(batch),features(size_t(batch)*f);
                std::vector<double> costs(size_t(batch)*7*f),floors(size_t(batch)*7),out(size_t(batch)*4*f*f);
                std::vector<unsigned char> allowed(size_t(batch)*7*f),dominated(size_t(batch)*2*f),reasons(out.size());std::vector<int> tail(out.size());
                #pragma omp parallel for schedule(static) num_threads(o.threads)
                for(int i=0;i<batch;++i){auto& in=inputs[i];std::copy(groups[in.gid].rows.begin(),groups[in.gid].rows.end(),masks.begin()+size_t(i)*W);
                    for(int w=0;w<W;++w)if(masks[size_t(i)*W+w])indices[size_t(i)*W+nw[i]++]=w;
                    for(int j=0;j<in.F;++j){features[size_t(i)*f+j]=in.features[j];for(int k=0;k<7;++k)allowed[(size_t(i)*7+k)*f+j]=1;
                        for(int a=0;a<2;++a)dominated[(size_t(i)*2+a)*f+j]=in.dominated[a*in.F+j];}}
                int max_words=*std::max_element(nw.begin(),nw.end());
                DSGpuBatch request{1,batch,f,W,K,F,n,max_words,zero.data(),class_bits.data(),masks.data(),indices.data(),nw.data(),features.data(),costs.data(),floors.data(),allowed.data(),dominated.data(),out.data(),tail.data(),reasons.data(),1./n,remaining()};
                int code=gpu(&request);if(code==2)throw TimedOut();if(code)throw std::runtime_error("CUDA provider failed");
                #pragma omp parallel for schedule(dynamic,1) num_threads(o.threads)
                for(int i=0;i<batch;++i)try{answers[i]=inputs[i].F?join(inputs[i],f,out.data()+size_t(i)*4*f*f,tail.data()+size_t(i)*4*f*f):leaf(groups[ids[i]].rows).first;}
                    catch(...){std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();}
                if(failure)std::rethrow_exception(failure);stats["gpu_batches"]+=1;
            }else for(int i=0;i<batch;++i)answers[i]=leaf(groups[ids[i]].rows).first;
        }
        // Atomic batch commit: interrupted/partial batches never become exact.
        check();for(int i=0;i<int(ids.size());++i){auto& g=groups[ids[i]];double value=audit(answers[i],g.rows,3);
            if(value<g.lo-1e-8||value>g.hi+1e-8)throw std::runtime_error("Exact D3 cost outside previous interval");
            g.lo=g.hi=value;g.tree=answers[i];g.exact=true;}
        stats["exact_private_evaluations"]+=double(ids.size());stats["oracle_batches"]+=1;
        memory_check();
    }
    Tree recover(const std::vector<int>& selected){
        std::vector<int> states(M);for(int q=0;q<B;++q)states[q+first]=selected[q];
        if(ee){states[0]=states[1];states[M-1]=states[M-2];}
        std::function<Tree(int,int,int)> rec=[&](int begin,int end,int level)->Tree{
            if(level==h){int gid=private_id[begin*P+states[begin]];if(gid<0)throw std::runtime_error("Missing tail state");return groups[gid].tree;}
            int prefix=shape->project(states[begin],level+1);
            // Decode the full signature, stopping at the requested decision.
            int s=states[begin],action=-1,label=-1;
            for(int j=0;j<=level;++j){int rem=h-j,cut=F*int(shape->P[rem-1]);
                if(s>=cut){label=s-cut;break;}action=s/int(shape->P[rem-1]);s%=int(shape->P[rem-1]);}
            for(int q=begin+1;q<end;++q)if(shape->project(states[q],level+1)!=prefix)throw std::runtime_error("Incompatible glued signature");
            if(label>=0)return leaves[label];int mid=(begin+end)/2;return split(action,rec(begin,mid,level+1),rec(mid,end,level+1));
        };
        return rec(0,M,0);
    }
    void accept(const std::vector<int>& selected,double value){
        if(value>=ub-1e-12)return;Timer timer(stats["recovery_seconds"]);Tree tree=recover(selected);double routed=audit(tree,all,D);
        if(std::abs(routed-value)>1e-7)throw std::runtime_error("Recovered tree differs from master/path cost");incumbent=tree;ub=routed;
    }
    std::unique_ptr<Master> make_master(){Timer timer(stats["rmp_setup_seconds"]);memory_check();check();
        auto model=std::make_unique<Master>(B,F,K,h,first,1,o.memory);memory_check();check();return model;}
    void update(Master& master,const std::vector<int>& ids,const std::vector<double>& values){
        if(ids.empty())return;Timer timer(stats["rmp_update_seconds"]);std::vector<double> costs(ids.size());
        for(size_t j=0;j<ids.size();++j)costs[j]=values[ids[j]];master.update(int(ids.size()),ids.data(),costs.data());check();memory_check();
    }
    Solution optimize(Master& master){
        Solution out;{Timer timer(stats["rmp_seconds"]);master.model.set(GRB_DoubleParam_TimeLimit,remaining());master.model.optimize();}
        ++stats["iterations"];out.status=master.model.get(GRB_IntAttr_Status);out.columns=int(master.vars.size());
        out.rows=master.model.get(GRB_IntAttr_NumConstrs);out.nonzeros=master.model.get(GRB_IntAttr_NumNZs);
        stats["final_columns"]=out.columns;stats["final_rows"]=out.rows;stats["peak_active_columns"]=std::max(stats["peak_active_columns"],double(out.columns));
        stats["peak_rmp_nonzeros"]=std::max(stats["peak_rmp_nonzeros"],double(out.nonzeros));
        if(out.status==9||out.status==11)throw TimedOut();
        if(out.status!=2)throw std::runtime_error("Unexpected native master status "+std::to_string(out.status));
        {Timer timer(stats["rmp_recovery_seconds"]);out.objective=master.model.get(GRB_DoubleAttr_ObjVal);
            for(auto& c:master.norm){out.alpha.push_back(c.get(GRB_DoubleAttr_Pi));out.residual=std::max(out.residual,std::abs(c.get(GRB_DoubleAttr_Slack)));}
            for(auto& c:master.rows){out.pi.push_back(c.get(GRB_DoubleAttr_Pi));out.residual=std::max(out.residual,std::abs(c.get(GRB_DoubleAttr_Slack)));}
            std::vector<double> active(size_t(B)*P,INF);for(auto& entry:master.vars)active[entry.first]=entry.second.get(GRB_DoubleAttr_Obj);
            auto path=path_opt(*shape,B,first,active.data());if(std::abs(path.value-out.objective)>1e-7||out.residual>1e-7)throw std::runtime_error("RMP primal recovery audit failed");out.selected=std::move(path.selected);}
        check();memory_check();return out;
    }
    std::vector<int> retained_tree_ids(){auto ids=tree_ids(incumbent,false);std::vector<int> out(B);for(int q=0;q<B;++q)out[q]=q*P+ids[q+first]%P;return out;}
    std::vector<int> unresolved(const std::vector<double>& rc){
        std::vector<int> order;for(int id=0;id<int(rc.size());++id)if(rc[id]<-1e-9&&cost_low[id]<cost_high[id]-1e-12)order.push_back(id);
        std::sort(order.begin(),order.end(),[&](int a,int b){return rc[a]<rc[b]||(rc[a]==rc[b]&&a<b);});
        std::vector<int> out;std::unordered_set<int> seen;
        auto add=[&](int original){int gid=private_id[original];if(gid>=0&&!groups[gid].exact&&seen.insert(gid).second)out.push_back(gid);};
        for(int id:order){int q=id/P,s=id%P;add((q+first)*P+s);if(ee&&q==0)add(s);if(ee&&q==B-1)add((M-1)*P+s);
            if(out.size()>=size_t(o.batch))break;}
        if(out.size()>size_t(o.batch))out.resize(o.batch);return out;
    }
    std::vector<int> capacity_requests(){
        PathResult path;{Timer timer(stats["message_seconds"]);path=path_opt(*shape,B,first,cost_low.data());}
        std::vector<int> ids;std::unordered_set<int> seen;
        auto add=[&](int original){int gid=private_id[original];if(gid>=0&&!groups[gid].exact&&seen.insert(gid).second)ids.push_back(gid);};
        for(int q=0;q<B;++q){int s=path.selected[q];add((q+first)*P+s);
            if(ee&&q==0)add(s);if(ee&&q==B-1)add((M-1)*P+s);}
        if(ids.size()>size_t(o.batch))ids.resize(o.batch);
        stats["capacity_path_refinements"]+=ids.size();return ids;
    }
    void solve_master(){
        refresh();auto master=make_master();std::vector<unsigned char> active(size_t(B)*P);
        std::vector<int> ids;
        if(full){for(int id=0;id<int(cost_high.size());++id)if(std::isfinite(cost_high[id]))ids.push_back(id);
            stats["full_feasible_columns"]=double(ids.size());if(ids.size()>size_t(o.cap))throw Resource("RESOURCE","Full LP column capacity");}
        else ids=retained_tree_ids();
        if(ids.size()>size_t(o.cap))throw Resource("RESOURCE","Capacity cannot hold a feasible bundle");
        update(*master,ids,cost_high);for(int id:ids)active[id]=1;int iteration=0,last_compact=-20;
        while(true){check();refresh();ids.clear();for(int id=0;id<int(active.size());++id)if(active[id])ids.push_back(id);
            update(*master,ids,cost_high);Solution answer=optimize(*master);accept(answer.selected,answer.objective);
            if(full){lb=answer.objective;status="OPT";proof=elapsed(started);break;}
            std::vector<double> rc(cost_low.size()),upper_rc(cost_low.size()),minima(B);
            {Timer timer(stats["pricing_seconds"]);master->price(cost_low.data(),answer.alpha.data(),answer.pi.data(),rc.data(),minima.data());
                double bound=std::accumulate(answer.alpha.begin(),answer.alpha.end(),0.);for(double m:minima)bound+=std::min(0.,m);lb=std::max(lb,bound);
                master->price(cost_high.data(),answer.alpha.data(),answer.pi.data(),upper_rc.data(),minima.data());stats["full_pricing_calls"]+=2;}
            if(o.message){PathResult lower,upper;{Timer timer(stats["message_seconds"]);lower=path_opt(*shape,B,first,cost_low.data());upper=path_opt(*shape,B,first,cost_high.data());}
                lb=std::max(lb,lower.value);accept(upper.selected,upper.value);stats["message_passes"]+=1;}
            if(lb>ub+1e-7)throw std::runtime_error("Certified LB exceeds incumbent");lb=std::min(lb,ub);
            trace.push_back({elapsed(started),lb,ub,answer.columns});
            if(ub-lb<=1e-7){status="OPT";proof=elapsed(started);break;}
            std::vector<int> requests,enter;
            {Timer pricing_timer(stats["pricing_seconds"]);requests=unresolved(rc);
            for(int q=0;q<B;++q){std::vector<int> local;for(int s=0;s<P;++s){int id=q*P+s;
                    if(!active[id]&&upper_rc[id]<-1e-9&&cost_high[id]-cost_low[id]<=1e-12)local.push_back(id);}
                std::sort(local.begin(),local.end(),[&](int a,int b){return upper_rc[a]<upper_rc[b]||(upper_rc[a]==upper_rc[b]&&a<b);});
                if(o.columns_per_block&&local.size()>size_t(o.columns_per_block))local.resize(o.columns_per_block);enter.insert(enter.end(),local.begin(),local.end());}}
            bool compact=o.compact&&answer.columns>=o.compact&&iteration-last_compact>=20;
            bool capacity=!enter.empty()&&answer.columns>=o.cap;
            // An unknown state can have nonnegative individual reduced cost
            // while belonging to a negative-cost compatible bundle. Under a
            // tight column cap, repeatedly recycling exact negative columns
            // must not starve that unknown state. Shared by base and EE.
            if(requests.empty()&&(capacity||answer.columns+int(enter.size())>o.cap))
                requests=capacity_requests();
            if(compact||capacity){
                std::vector<int> keep=retained_tree_ids();for(int q=0;q<B;++q)keep.push_back(q*P+answer.selected[q]);
                std::sort(keep.begin(),keep.end());keep.erase(std::unique(keep.begin(),keep.end()),keep.end());
                if(keep.size()>=size_t(o.cap))keep=retained_tree_ids();
                if(keep.size()>=size_t(o.cap)&&requests.empty())throw Resource("RESOURCE","Capacity cannot retain support and entering column");
                master.reset();master=make_master();std::fill(active.begin(),active.end(),0);update(*master,keep,cost_high);for(int id:keep)active[id]=1;
                stats[capacity?"capacity_rebuilds":"rmp_compactions"]+=1;last_compact=iteration;
                // Refine before repricing so recycling cannot starve the oracle.
                if(!requests.empty())oracle(requests);
                ++iteration;continue;
            }
            int room=o.cap-answer.columns;
            if(enter.size()>size_t(room)){std::stable_sort(enter.begin(),enter.end(),[&](int a,int b){return upper_rc[a]<upper_rc[b];});enter.resize(room);stats["capacity_throttled_batches"]+=1;}
            update(*master,enter,cost_high);for(int id:enter)active[id]=1;
            if(!requests.empty())oracle(requests);
            else if(enter.empty())throw Resource("RESOURCE","Open gap without admissible column or unresolved negative state");
            ++iteration;
        }
        trace.push_back({elapsed(started),lb,ub,int(stats["final_columns"])});
    }
    void run(){
        try{
            prepare_data();{Timer timer(stats["initialization_seconds"]);std::vector<int> used;incumbent=greedy(all,D,used);ub=audit(incumbent,all,D);}
            prepare_states();{Timer timer(stats["initialization_seconds"]);tree_ids(incumbent,true);}
            if(gpu){Timer timer(stats["backend_initialization_seconds"]);DSGpuBatch init{};init.phase=0;init.W=W;init.K=K;init.globalF=F;init.n=n;
                init.zero=zero.data();init.labels=class_bits.data();init.remaining=remaining();if(gpu(&init))throw std::runtime_error("CUDA initialization failed");}
            if(full){std::vector<int> pending;for(int g=0;g<int(groups.size());++g)if(!groups[g].exact)pending.push_back(g);
                for(size_t j=0;j<pending.size();j+=o.batch)oracle(std::vector<int>(pending.begin()+j,pending.begin()+std::min(pending.size(),j+o.batch)));
                stats["precompute_complete"]=1;}
            solve_master();
        }catch(const TimedOut&){status="TIME";}
         catch(const Resource& e){status=e.status;reason=e.what();}
         catch(const std::bad_alloc&){status="MEM";reason="Native allocation failed";}
         catch(const GRBException& e){if(e.getErrorCode()==GRB_ERROR_OUT_OF_MEMORY){status="MEM";reason="Gurobi allocation failed";}else throw;}
        stats["exact_terminal_states"]=0;for(auto& g:groups)stats["exact_terminal_states"]+=g.exact;
        stats["unresolved_terminal_states"]=groups.size()-stats["exact_terminal_states"];
        if(lb>ub+1e-7)throw std::runtime_error("Final lower bound invalid");lb=std::min(lb,ub);
    }
    void write_tree(std::ostream& os,Tree t)const{
        if(t->f<0)os<<"{\"label\":"<<labels[t->k]<<"}";
        else{os<<"{\"feature\":"<<t->f<<",\"left\":";write_tree(os,t->l);os<<",\"right\":";write_tree(os,t->r);os<<"}";}
    }
    std::string result(){
        std::ostringstream os;os<<std::setprecision(17);
        os<<"{\"method\":\""<<(full?(ee?"JT-LP-SC-EE":"JT-LP-SC"):(ee?"JT-CG-EE":"JT-CG"))<<"\",\"status\":\""<<status<<"\",\"LB\":"<<lb<<",\"UB\":"<<ub;
        os<<",\"seconds\":"<<elapsed(started)<<",\"absolute_gap\":"<<std::max(0.,ub-lb)<<",\"proof_seconds\":";
        if(proof<0)os<<"null";else os<<proof;
        os<<",\"tree\":";write_tree(os,incumbent);
        os<<",\"certificate\":{\"complete_pricing_domain\":true,\"pricing_performed\":"<<(stats["full_pricing_calls"]>0?"true":"false")
          <<",\"upper_tree_independently_audited\":true,\"kind\":\""<<(status!="OPT"?"valid_interval_only":(full?"complete_contracted_LP":"complete_lower_cost_pricing"))<<"\"}";
        os<<",\"stats\":{\"native_core\":true,\"tail_depth\":3,\"original_clusters\":"<<M<<",\"retained_clusters\":"<<B;
        os<<",\"peak_private_bytes\":"<<peak_private<<",\"peak_rss_bytes\":"<<peak_rss<<",\"threads\":"<<o.threads;
        os<<",\"coordinator_policy\":\"incremental_upper_rmp_exact_admission_dual_refinement_v1\",\"message_bounds\":"<<(o.message?"true":"false");
        os<<",\"resource_reason\":\""<<reason<<"\"";for(auto [key,value]:stats)os<<",\""<<key<<"\":"<<value;
        os<<"},\"trace\":[";for(size_t i=0;i<trace.size();++i){if(i)os<<",";auto t=trace[i];os<<"{\"seconds\":"<<t.seconds<<",\"LB\":"<<t.lb<<",\"UB\":"<<t.ub<<",\"columns\":"<<t.columns<<"}";}os<<"]}";
        return os.str();
    }
};
}

API const char* ds_campaign_solve(const uint8_t* X,const int64_t* y,int n,int F,int depth,int ee,int full,
    int threads,int batch,int cap,int compact,int message,int columns_per_block,uint64_t memory,double seconds,DSGpuProvider gpu){
    static thread_local std::string output;
    try{deep_campaign::Options options{threads,batch,cap,compact,message,columns_per_block,memory};
        deep_campaign::Engine engine(X,y,n,F,depth,ee!=0,full!=0,options,seconds,gpu);engine.run();output=engine.result();return output.c_str();
    }catch(const GRBException& e){error=e.getMessage();return nullptr;}
     catch(const std::exception& e){error=e.what();return nullptr;}
}
