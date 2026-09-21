// Complete exact-cost LP baseline, D2 and C3-contracted D4/D5.
// Shared optimized D3 OpenMP/CUDA kernels; no Python state search or reduction.
#pragma once
#include <mutex>
#include <unordered_map>

extern "C" void* d3_opt_create(int,int);
extern "C" void d3_opt_free(void*);
extern "C" int d3_opt_costs_multiclass(void*,const uint64_t*,const uint64_t*,const double*,
    const unsigned char*,const double*,const unsigned char*,const int*,double,double*,int*,unsigned char*);
struct RevisionGpuBatch {
    int phase,B,F,W,K,globalF,n,max_words;
    const uint64_t *zero,*labels,*masks;const int *indices,*nw,*features;
    const double *costs,*floors;const unsigned char *allowed,*dominated;
    double* output;int* tail;unsigned char* reasons;double weight,remaining;
};
using RevisionGpuProvider=int(*)(RevisionGpuBatch*);
using RevisionProgress=int(*)(const char*);

namespace revision {
using Bits=std::vector<uint64_t>;
struct TailAnswer {double value=inf;std::array<int,15> tree;TailAnswer(){tree.fill(inactive);}};
struct Conditional {Bits rows;TailAnswer answer;bool exact=false;};
struct OracleInput {int gid,n,F;std::vector<int> features;std::vector<double> side;std::vector<uint8_t> dominated;};

inline std::vector<int> root_tree(const Data& d){std::vector<int> t(d.nodes(),inactive);t[0]=-1-d.majority();return t;}
inline std::string feasible_seed(const Data& d,double penalty){
    if(!d.early||d.n<d.min_leaf)throw std::invalid_argument("Root seed requires feasible early stopping");
    auto tree=root_tree(d);Result out;out.status="TIME";out.coordinator="initial_root_prediction";
    out.pick.value=d.audit(tree,penalty);std::ostringstream s;finish(s,out,d,tree,penalty);s<<",\"checkpoint_only\":true}";return s.str();
}

struct Eager {
    const Data& d;double penalty;Budget budget;int kind,cap,presolve,lp_method,batch,W,h,private_depth;
    RevisionGpuProvider gpu;CostProvider stump_gpu;RevisionProgress progress;Source source;
    Bits all,zero,ones,classes;std::vector<Conditional> groups;std::vector<double> prefix;
    std::map<std::string,double> stats;std::vector<int> incumbent;double incumbent_value;
    Eager(Data& data,double p,int k,double seconds,uint64_t memory,int c,int ps,int method,
          RevisionGpuProvider g,CostProvider sg,int bs,RevisionProgress cb,int pd=0):d(data),penalty(p),budget(seconds,memory),
          kind(k),cap(c),presolve(ps),lp_method(method),batch(bs),W((d.n+63)/64),h(pd?d.D-pd:(d.D==5?2:1)),private_depth(d.D-h),
          gpu(g),stump_gpu(sg),progress(cb),source(&data,1<<h,p),incumbent(root_tree(d)),incumbent_value(d.audit(incumbent,p)){
        if((d.D!=2&&d.D!=3&&d.D!=4&&d.D!=5)||h<1||h>2||private_depth<1||private_depth>3||!d.early||!d.no_repeat||d.min_leaf!=0||kind<0||kind>4||batch<1||cap<1)
            throw std::invalid_argument("Eager baseline requires D2/D4/D5, early stop, no-repeat, min_leaf=0");
        for(double w:d.weights)if(w!=d.weights[0])throw std::invalid_argument("Eager optimized kernel requires uniform weights");
        for(size_t i=0;i<d.allowed.size();++i)if(d.allowed[i]!=1||d.extras[i]!=0)
            throw std::invalid_argument("Eager baseline requires unrestricted features and uniform split penalty");
        if((private_depth==1&&gpu)||(private_depth!=1&&stump_gpu))throw std::invalid_argument("Wrong GPU provider for private depth");
    }
    int count(const Bits& r)const{int n=0;for(auto b:r)n+=pop64(b);return n;}
    Bits route(const Bits& r,int f,int side)const{Bits t(W);for(int w=0;w<W;++w)t[w]=r[w]&(side?~zero[size_t(f)*W+w]:zero[size_t(f)*W+w]);return t;}
    std::pair<int,double> leaf(const Bits& r)const{
        int total=0,best=-1,label=0;for(int k=0;k<d.K;++k){int c=0;for(int w=0;w<W;++w)c+=pop64(r[w]&classes[size_t(k)*W+w]);total+=c;if(c>best){best=c;label=k;}}
        return {label,(total-best)*d.weights[0]};
    }
    void checkpoint(){if(progress){Result out;out.pick.value=incumbent_value;out.coordinator="native_initial_feasible_tree";std::ostringstream s;
        finish(s,out,d,incumbent,penalty);s<<",\"checkpoint_only\":true}";std::string value=s.str();if(progress(value.c_str()))throw std::runtime_error("Checkpoint write failed");}}
    void greedy(const Bits& rows,int v,int remaining,std::vector<int> used){
        budget.check();auto stop=leaf(rows);incumbent[v]=-1-stop.first;if(!remaining||stop.second==0)return;
        std::vector<double> scores(d.F,inf);
        #pragma omp parallel for num_threads(d.threads) schedule(static)
        for(int f=0;f<d.F;++f)if(std::find(used.begin(),used.end(),f)==used.end())
            scores[f]=penalty+leaf(route(rows,f,0)).second+leaf(route(rows,f,1)).second;
        int f=int(std::min_element(scores.begin(),scores.end())-scores.begin());if(scores[f]>=stop.second)return;
        incumbent[v]=f;used.push_back(f);incumbent[2*v+1]=-1-leaf(route(rows,f,0)).first;incumbent[2*v+2]=-1-leaf(route(rows,f,1)).first;
        greedy(route(rows,f,0),2*v+1,remaining-1,used);greedy(route(rows,f,1),2*v+2,remaining-1,used);
    }
    void prepare(){
        budget.check();auto tick=Clock::now();all.assign(W,~uint64_t(0));if(d.n%64)all.back()=(uint64_t(1)<<(d.n%64))-1;
        zero.assign(size_t(d.F)*W,0);classes.assign(size_t(d.K)*W,0);
        #pragma omp parallel for num_threads(d.threads) schedule(static)
        for(int w=0;w<W;++w)for(int i=64*w;i<std::min(d.n,64*w+64);++i){uint64_t bit=uint64_t(1)<<(i%64);classes[size_t(d.y[i])*W+w]|=bit;
            for(int f=0;f<d.F;++f)if(!d.X[size_t(i)*d.F+f])zero[size_t(f)*W+w]|=bit;}
        stats["packing_seconds"]=elapsed(tick);budget.check();tick=Clock::now();
        try{greedy(all,0,d.D,{});}catch(...){incumbent_value=d.audit(incumbent,penalty);checkpoint();throw;}
        incumbent_value=d.audit(incumbent,penalty);stats["initialization_seconds"]=elapsed(tick);checkpoint();budget.check();tick=Clock::now();
        size_t N=size_t(source.M)*source.P;source.high.assign(N,inf);source.private_ids.assign(N,-1);prefix.assign(N,0);
        // Row sets are an exact cache key here: ancestors are constant on their
        // routed rows and constant tests cannot improve this sparse nonnegative-cost domain.
        std::unordered_map<std::string,int> cache;std::mutex cache_mutex;std::exception_ptr failure;std::atomic<bool> failed{false};
        #pragma omp parallel for num_threads(d.threads) schedule(dynamic,16)
        for(int id=0;id<int(N);++id)try{
            if(failed.load(std::memory_order_relaxed))continue;
            if(!(id%64))budget.check();int q=id/source.P,s=id%source.P,A=d.F+d.K;
            int root=h==1?s:(s<d.F*A?s/A:d.F+s-d.F*A);int actions[2]={root,s%A};Bits rows=all;bool stopped=false,valid=true;int used=-1;double base=0;
            for(int j=0;j<h;++j){int a=actions[j];double divisor=double(source.M>>j);
                if(a>=d.F){int label=a-d.F,total=count(rows),correct=0;for(int w=0;w<W;++w)correct+=pop64(rows[w]&classes[size_t(label)*W+w]);
                    source.high[id]=base+(total-correct)*d.weights[0]/divisor;stopped=true;break;}
                if(a==used){valid=false;break;}used=a;base+=penalty/divisor;rows=route(rows,a,(q>>(h-j-1))&1);}
            if(!valid||stopped)continue;auto stop=leaf(rows);std::string key(reinterpret_cast<const char*>(rows.data()),rows.size()*sizeof(uint64_t));
            std::lock_guard<std::mutex> lock(cache_mutex);auto found=cache.find(key);int gid;
            if(found==cache.end()){gid=int(groups.size());Conditional g;g.rows=std::move(rows);g.answer.value=stop.second;g.answer.tree[0]=-1-stop.first;g.exact=stop.second==0.;groups.push_back(std::move(g));cache.emplace(std::move(key),gid);}else gid=found->second;
            source.private_ids[id]=gid;prefix[id]=base;
        }catch(...){failed.store(true,std::memory_order_relaxed);std::lock_guard<std::mutex> lock(cache_mutex);if(!failure)failure=std::current_exception();}
        if(failure)std::rethrow_exception(failure);
        // Stable cache IDs, independent of worker insertion order.
        std::vector<int> first(groups.size(),INT32_MAX),order(groups.size()),remap(groups.size());
        for(int id=0;id<int(N);++id)if(source.private_ids[id]>=0)first[source.private_ids[id]]=std::min(first[source.private_ids[id]],id);
        std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](int a,int b){return first[a]<first[b];});
        std::vector<Conditional> stable;stable.reserve(groups.size());for(int i=0;i<int(order.size());++i){remap[order[i]]=i;stable.push_back(std::move(groups[order[i]]));}groups.swap(stable);
        for(auto& id:source.private_ids)if(id>=0)id=remap[id];
        stats["state_preparation_seconds"]=elapsed(tick);stats["unique_private_states"]=double(groups.size());budget.check();
    }
    OracleInput input(int gid){
        OracleInput in;in.gid=gid;const auto& rows=groups[gid].rows;in.n=count(rows);
        std::unordered_map<std::string,bool> seen;
        for(int f=0;f<d.F;++f){if(!(f%32))budget.check();Bits a=route(rows,f,0);int n=count(a);if(!n||n==in.n)continue;Bits b(W);for(int w=0;w<W;++w)b[w]=rows[w]^a[w];if(b<a)a.swap(b);
            std::string key(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(uint64_t));if(!seen.emplace(std::move(key),true).second)continue;in.features.push_back(f);}
        in.F=int(in.features.size());in.side.resize(2*in.F);in.dominated.resize(2*in.F);
        for(int side=0;side<2;++side)for(int f=0;f<in.F;++f){double z=leaf(route(rows,in.features[f],side)).second;in.side[side*in.F+f]=z;in.dominated[side*in.F+f]=z==0;}
        return in;
    }
    TailAnswer join(const OracleInput& in,int stride,const double* costs,const int* tail){
        TailAnswer out=groups[in.gid].answer;if(!in.F)return out;int root=-1,gs[2]={-1,-1};
        for(int f=0;f<in.F;++f){double total=0;int choices[2];for(int side=0;side<2;++side){double z=.5*penalty+in.side[side*in.F+f];choices[side]=-1;
                for(int g=0;g<in.F;++g){double value=costs[(size_t(2*side)*stride+f)*stride+g]+costs[(size_t(2*side+1)*stride+f)*stride+g];if(value<z){z=value;choices[side]=g;}}total+=z;}
            if(total<out.value){out.value=total;root=f;gs[0]=choices[0];gs[1]=choices[1];}}
        if(root<0)return out;out.tree.fill(inactive);out.tree[0]=in.features[root];
        for(int side=0;side<2;++side){int v=1+side;Bits rows=route(groups[in.gid].rows,in.features[root],side);int g=gs[side];
            if(g<0){out.tree[v]=-1-leaf(rows).first;continue;}out.tree[v]=in.features[g];
            for(int b=0;b<2;++b){int q=2*side+b,node=3+q;Bits r=route(rows,in.features[g],b);int a=tail[(size_t(q)*stride+root)*stride+g];
                if(a==-1)out.tree[node]=-1-leaf(r).first;else if(a>=0&&a<in.F){out.tree[node]=in.features[a];out.tree[2*node+1]=-1-leaf(route(r,in.features[a],0)).first;out.tree[2*node+2]=-1-leaf(route(r,in.features[a],1)).first;}
                else throw std::runtime_error("Invalid exact oracle backpointer");}}
        return out;
    }
    TailAnswer cpu(const OracleInput& in){
        if(!in.F)return groups[in.gid].answer;int words=(in.n+63)/64;Bits z(size_t(in.F)*words),c(size_t(d.K)*words);int j=0;
        for(int w=0;w<W;++w){uint64_t bits=groups[in.gid].rows[w];while(bits){int row=64*w+low64(bits);bits&=bits-1;c[size_t(d.y[row])*words+j/64]|=uint64_t(1)<<(j%64);
                for(int f=0;f<in.F;++f)if(!d.X[size_t(row)*d.F+in.features[f]])z[size_t(f)*words+j/64]|=uint64_t(1)<<(j%64);++j;}}
        std::vector<double> costs(7*in.F,penalty),floors(7,0),out(size_t(4)*in.F*in.F);std::vector<uint8_t> allowed(7*in.F,1),reasons(out.size());std::vector<int> tail(out.size());
        if(private_depth==2)std::fill(allowed.begin()+3*in.F,allowed.end(),0);
        void* ws=d3_opt_create(words,1);if(!ws)throw std::bad_alloc();struct Cleanup{void* p;~Cleanup(){d3_opt_free(p);}}cleanup{ws};
        for(int start=0;start<in.F*in.F;start+=64){budget.check();int ip[]={in.F,words,in.n,1,1,0,1,1,start,std::min(64,in.F*in.F-start),d.K};
            if(d3_opt_costs_multiclass(ws,z.data(),c.data(),costs.data(),allowed.data(),floors.data(),in.dominated.data(),ip,d.weights[0],out.data(),tail.data(),reasons.data()))throw std::runtime_error("D3 optimized CPU failure");}
        return join(in,in.F,out.data(),tail.data());
    }
    void d3_batch(const std::vector<int>& ids){
        std::vector<OracleInput> ins(ids.size());std::vector<TailAnswer> answers(ids.size());std::exception_ptr failure;std::mutex mutex;
        #pragma omp parallel for num_threads(d.threads) schedule(dynamic,1)
        for(int i=0;i<int(ids.size());++i)try{budget.check();ins[i]=input(ids[i]);if(!gpu)answers[i]=cpu(ins[i]);}catch(...){std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();}
        if(failure)std::rethrow_exception(failure);
        if(gpu){int B=int(ids.size()),F=0;for(auto& in:ins)F=std::max(F,in.F);
            if(F){Bits masks(size_t(B)*W);std::vector<int> indices(size_t(B)*W),nw(B),features(size_t(B)*F);
                std::vector<double> costs(size_t(B)*7*F,penalty),floors(size_t(B)*7),out(size_t(B)*4*F*F);
                std::vector<uint8_t> allowed(size_t(B)*7*F),dominated(size_t(B)*2*F),reasons(out.size());std::vector<int> tail(out.size());
                #pragma omp parallel for num_threads(d.threads) schedule(static)
                for(int i=0;i<B;++i){const auto& in=ins[i];std::copy(groups[ids[i]].rows.begin(),groups[ids[i]].rows.end(),masks.begin()+size_t(i)*W);
                    for(int w=0;w<W;++w)if(masks[size_t(i)*W+w])indices[size_t(i)*W+nw[i]++]=w;
                    for(int f=0;f<in.F;++f){features[size_t(i)*F+f]=in.features[f];for(int v=0;v<(private_depth==2?3:7);++v)allowed[(size_t(i)*7+v)*F+f]=1;for(int side=0;side<2;++side)dominated[(size_t(i)*2+side)*F+f]=in.dominated[side*in.F+f];}}
                RevisionGpuBatch req{1,B,F,W,d.K,d.F,d.n,*std::max_element(nw.begin(),nw.end()),zero.data(),classes.data(),masks.data(),indices.data(),nw.data(),features.data(),costs.data(),floors.data(),allowed.data(),dominated.data(),out.data(),tail.data(),reasons.data(),d.weights[0],budget.remaining()};
                int code=gpu(&req);if(code==2)throw Timed{};if(code)throw std::runtime_error("Resident CUDA cost failure");++stats["gpu_batches"];
                #pragma omp parallel for num_threads(d.threads) schedule(dynamic,1)
                for(int i=0;i<B;++i)try{answers[i]=join(ins[i],F,out.data()+size_t(i)*4*F*F,tail.data()+size_t(i)*4*F*F);}catch(...){std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();}
                if(failure)std::rethrow_exception(failure);
            }else for(int i=0;i<B;++i)answers[i]=groups[ids[i]].answer;
        }
        budget.check();for(int i=0;i<int(ids.size());++i){groups[ids[i]].answer=answers[i];groups[ids[i]].exact=true;}stats["exact_kernel_evaluations"]+=double(ids.size());
    }
    void d2_batch(const std::vector<int>& ids){
        int B=int(ids.size());Bits masks(size_t(B)*W);std::vector<double> loss(size_t(B)*d.F);std::vector<int32_t> labels(size_t(B)*d.F*2);
        if(stump_gpu){if(ones.empty()){ones.resize(zero.size());for(int f=0;f<d.F;++f)for(int w=0;w<W;++w)ones[size_t(f)*W+w]=all[w]^zero[size_t(f)*W+w];}
            for(int i=0;i<B;++i)std::copy(groups[ids[i]].rows.begin(),groups[ids[i]].rows.end(),masks.begin()+size_t(i)*W);
            int code=stump_gpu(B,ones.data(),classes.data(),masks.data(),loss.data(),labels.data(),budget.remaining());if(code==-1)throw Timed{};if(code!=1)throw std::runtime_error("D2 CUDA statistics failure");++stats["gpu_batches"];}
        std::exception_ptr failure;std::mutex mutex;
        #pragma omp parallel for num_threads(d.threads) schedule(dynamic,1)
        for(int i=0;i<B;++i)try{budget.check();auto& group=groups[ids[i]];TailAnswer out=group.answer;
            for(int f=0;f<d.F;++f){auto l=stump_gpu?std::pair<int,double>{labels[(size_t(i)*d.F+f)*2],0.}:leaf(route(group.rows,f,0));
                auto r=stump_gpu?std::pair<int,double>{labels[(size_t(i)*d.F+f)*2+1],0.}:leaf(route(group.rows,f,1));
                double value=penalty+(stump_gpu?loss[size_t(i)*d.F+f]:l.second+r.second);
                if(value<out.value){out.value=value;out.tree.fill(inactive);out.tree[0]=f;out.tree[1]=-1-l.first;out.tree[2]=-1-r.first;}}
            group.answer=out;group.exact=true;
        }catch(...){std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();}
        if(failure)std::rethrow_exception(failure);budget.check();stats["exact_kernel_evaluations"]+=B;
    }
    std::string run(){
        Result out;out.pick.value=incumbent_value;out.coordinator="lp_sc_ee";checkpoint();
        try{prepare();if(gpu){auto tick=Clock::now();RevisionGpuBatch init{};init.phase=0;init.W=W;init.K=d.K;init.globalF=d.F;init.n=d.n;init.zero=zero.data();init.labels=classes.data();init.remaining=budget.remaining();
                if(gpu(&init))throw std::runtime_error("Resident CUDA initialization failure");stats["backend_initialization_seconds"]=elapsed(tick);budget.check();}
            auto tick=Clock::now();std::vector<int> pending;for(int g=0;g<int(groups.size());++g)if(!groups[g].exact)pending.push_back(g);
            for(size_t start=0;start<pending.size();start+=batch){budget.check();std::vector<int> ids(pending.begin()+start,pending.begin()+std::min(pending.size(),start+batch));if(private_depth==1)d2_batch(ids);else d3_batch(ids);++stats["oracle_batches"];}
            stats["conditional_cost_seconds"]=elapsed(tick);source.groups=int(groups.size());source.tails.resize(groups.size()*15);
            for(int g=0;g<int(groups.size());++g){if(!groups[g].exact)throw std::runtime_error("Unresolved state in explicit LP");std::copy(groups[g].answer.tree.begin(),groups[g].answer.tree.end(),source.tails.begin()+size_t(g)*15);}
            #pragma omp parallel for num_threads(d.threads) schedule(static)
            for(int id=0;id<int(source.high.size());++id)if(source.private_ids[id]>=0)source.high[id]=prefix[id]+groups[source.private_ids[id]].answer.value;
            source.low=source.high;stats["complete_cost_table"]=1;stats["exact_private_states"]=double(groups.size());budget.check();
            Table t(source.M,d.F,d.K,source.high.data(),source.low.data(),d.threads,kind==1||kind==4,budget);out=coordinate(t,kind==2?2:(kind>=3?1:0),d.majority(),budget,cap,presolve,lp_method);
            stats["master_domain_columns"]=double(t.finite_count);stats["remaining_clusters"]=t.M;
            if(!out.originals.empty()&&out.pick.value<=incumbent_value){tick=Clock::now();incumbent=source.recover(out.originals);double rt=elapsed(tick);out.recovery+=rt;out.total+=rt;incumbent_value=out.pick.value;}
            else if(out.pick.value>incumbent_value)out.originals.clear();
        }catch(const Timed&){out.status="TIME";}catch(const Limited&){out.status="MEM";}catch(const std::bad_alloc&){out.status="MEM";}
        double total=elapsed(budget.start);out.pick.value=incumbent_value;std::ostringstream s;finish(s,out,d,incumbent,penalty);
        s<<",\"native_seconds\":"<<total<<",\"backend\":\""<<((gpu||stump_gpu)?"cuda":"cpp_openmp")<<"\",\"private_depth\":"<<private_depth<<",\"threads\":"<<d.threads;
        for(auto& kv:stats)s<<",\""<<kv.first<<"\":"<<kv.second;close_stats(s);return s.str();
    }
};
}
API const char* revision_seed(void* p,double penalty){return revision::result_call([&]{if(!p||!std::isfinite(penalty)||penalty<0)throw std::invalid_argument("Invalid seed request");return revision::feasible_seed(*static_cast<revision::Data*>(p),penalty);});}
API const char* revision_eager(void* p,double penalty,int kind,double seconds,uint64_t memory,int cap,int presolve,int lp_method,
    RevisionGpuProvider gpu,CostProvider stump_gpu,int batch,RevisionProgress progress){return revision::result_call([&]{if(!p)throw std::invalid_argument("Missing data");revision::Eager e(*static_cast<revision::Data*>(p),penalty,kind,seconds,memory,cap,presolve,lp_method,gpu,stump_gpu,batch,progress);return e.run();});}
