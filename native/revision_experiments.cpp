// Build as a separate DLL: archived d3_structural.cpp and its DLL stay untouched.
#include "revision_coordination.hpp"
#include "d3_structural.cpp"
#include <tuple>

namespace revision {
thread_local std::string error,output;
struct Answer{double value=inf;std::vector<int> tree;};
// Independent small-data exhaustive recursion, not a cost-table/SC/MP routine.
struct Reference {
    const Data& d;double penalty;using Key=std::tuple<int,uint64_t,uint64_t>;
    std::map<Key,Answer> cache;
    Reference(const Data& data,double p):d(data),penalty(p){if(!std::isfinite(p)||p<0)throw std::invalid_argument("Invalid penalty");if(d.n>32||d.F>6)throw std::invalid_argument("Independent reference restricted to n<=32,F<=6");}
    Answer solve(int v,uint64_t rows,uint64_t used){
        Key key{v,rows,used};auto found=cache.find(key);if(found!=cache.end())return found->second;
        int depth=0;for(int u=v;u;u=(u-1)/2)depth++;int count=0;std::vector<double> hist(d.K);
        for(int i=0;i<d.n;++i)if(rows&(uint64_t(1)<<i)){++count;hist[d.y[i]]+=d.weights[i];}
        Answer best;best.tree.assign(d.nodes(),inactive);
        if(count>=d.min_leaf&&(d.early||depth==d.D)){int k=int(std::max_element(hist.begin(),hist.end())-hist.begin());best.value=std::accumulate(hist.begin(),hist.end(),0.)-hist[k];best.tree[v]=-1-k;}
        if(depth<d.D)for(int f=0;f<d.F;++f){if(!d.allowed[size_t(v)*d.F+f]||(d.no_repeat&&(used&(uint64_t(1)<<f))))continue;
            uint64_t left=0,right=0;for(int i=0;i<d.n;++i)if(rows&(uint64_t(1)<<i)){if(d.X[size_t(i)*d.F+f])right|=uint64_t(1)<<i;else left|=uint64_t(1)<<i;}
            auto a=solve(2*v+1,left,used|(uint64_t(1)<<f)),b=solve(2*v+2,right,used|(uint64_t(1)<<f));double val=penalty+d.extras[size_t(v)*d.F+f]+a.value+b.value;
            if(val<best.value){best.value=val;best.tree.assign(d.nodes(),inactive);best.tree[v]=f;for(int j=0;j<d.nodes();++j){if(a.tree[j]!=inactive)best.tree[j]=a.tree[j];if(b.tree[j]!=inactive)best.tree[j]=b.tree[j];}}}
        cache.emplace(key,best);return best;
    }
};
struct Source {
    Data* d;int M,P;double penalty;std::vector<double> high,low;std::vector<int> private_ids,tails;
    int groups=0;
    size_t import_roundoff_states=0;double import_max_roundoff=0;
    Source(Data* data,int m,double p):d(data),M(m),P(m==2?data->F+data->K:data->F*(data->F+data->K)+data->K),penalty(p){
        if((m!=2&&m!=4)||d->D<(m==2?2:3)||d->D>(m==2?4:5)||!std::isfinite(p)||p<0)throw std::invalid_argument("Table requires one/two explicit levels and private depth <=3");}
    void copy_tail(std::vector<int>& full,int gv,const int* tail,int lv=0)const{
        if(lv>=15||gv>=d->nodes())throw std::runtime_error("Tail exceeds D3 boundary");int a=tail[lv];if(a==inactive)throw std::runtime_error("Incomplete realizing tail");full[gv]=a;
        if(a>=0){copy_tail(full,2*gv+1,tail,2*lv+1);copy_tail(full,2*gv+2,tail,2*lv+2);}
    }
    std::vector<int> recover(const std::vector<int>& ids)const{
        if(int(ids.size())!=M)throw std::runtime_error("Missing recovered groups");std::vector<int> full(d->nodes(),inactive);int A=d->F+d->K;
        auto set=[&](int v,int a){if(full[v]!=inactive&&full[v]!=a)throw std::runtime_error("Inconsistent separator recovery");full[v]=a;};
        for(int q=0;q<M;++q){int id=ids[q];if(id/P!=q||!std::isfinite(high.at(id)))throw std::runtime_error("Invalid selected state");int s=id%P;
            int root=M==2?s:(s<d->F*A?s/A:d->F+s-d->F*A);set(0,root<d->F?root:-1-(root-d->F));if(root>=d->F)continue;
            if(M==4){int action=s%A,v=1+q/2;set(v,action<d->F?action:-1-(action-d->F));if(action>=d->F)continue;}
            int gid=private_ids.at(id);if(gid<0||gid>=groups)throw std::runtime_error("Missing exact private representative");copy_tail(full,M-1+q,tails.data()+size_t(gid)*15);
        }return full;
    }
};
inline void finish(std::ostream& s,Result& out,const Data& d,const std::vector<int>& tree,double penalty){
    auto t=Clock::now();double audit=d.audit(tree,penalty);double audit_seconds=elapsed(t);
    if(std::abs(audit-out.pick.value)>1e-7||out.lb>audit+1e-7||(out.status=="OPT"&&audit-out.lb>1e-7))throw std::runtime_error("Independent recovered-tree objective/bounds audit failed");
    s<<"{\"engine\":\"revision_native_v3\",";out.fields(s);s<<",\"tree_audit_passed\":true,\"tree_objective\":"<<audit<<",\"post_audit_seconds\":"<<audit_seconds<<",\"tree\":";d.tree_json(s,tree);
}
inline void close_stats(std::ostream& s){PROCESS_MEMORY_COUNTERS_EX p{};p.cb=sizeof(p);GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&p),sizeof(p));s<<",\"peak_host_working_set_bytes\":"<<p.PeakWorkingSetSize<<",\"process_private_bytes\":"<<p.PrivateUsage<<"}";}
inline std::vector<int> d3_tree(const Data& d,const std::vector<Col>& cols,const std::vector<int>& lookup,const std::vector<int>& ids,int P){
    if(ids.size()!=4)throw std::runtime_error("D3 recovery lacks groups");std::vector<int> tree(15,inactive);int A=d.F+d.K,s=ids[0]%P,root=s<d.F*A?s/A:d.F+s-d.F*A;
    tree[0]=root<d.F?root:-1-(root-d.F);if(root>=d.F)return tree;
    for(int side=0;side<2;++side){int action=(ids[2*side]%P)%A;tree[1+side]=action<d.F?action:-1-(action-d.F);if(action>=d.F)continue;
        for(int j=0;j<2;++j){int q=2*side+j,index=lookup.at(ids[q]);if(index<0)throw std::runtime_error("D3 representative missing");const Col& c=cols[index];int v=3+q;
            if(c.h<0)tree[v]=-1-c.left;else{tree[v]=c.h;tree[2*v+1]=-1-c.left;tree[2*v+2]=-1-c.right;}}}
    return tree;
}
template<class Fn>const char* result_call(Fn fn){try{error.clear();output=fn();return output.c_str();}
    catch(const Timed&){output="{\"status\":\"TIME\",\"LB\":0,\"UB\":null,\"tree_audit_passed\":false}";return output.c_str();}
    catch(const Timeout&){output="{\"status\":\"TIME\",\"LB\":0,\"UB\":null,\"tree_audit_passed\":false}";return output.c_str();}
    catch(const Capacity&){output="{\"status\":\"COLUMN_LIMIT\",\"LB\":0,\"UB\":null,\"column_capacity_triggered\":true}";return output.c_str();}
    catch(const Limited& e){output="{\"status\":\"MEM\",\"LB\":0,\"UB\":null}";return output.c_str();}
    catch(const std::bad_alloc&){output="{\"status\":\"MEM\",\"LB\":0,\"UB\":null}";return output.c_str();}
    catch(const GRBException& e){if(e.getErrorCode()==10001){output="{\"status\":\"MEM\",\"LB\":0,\"UB\":null}";return output.c_str();}error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}}
}

API const char* revision_error(){return revision::error.c_str();}
API int revision_unpack(const int64_t* matrix,uint8_t* X,int64_t* y,int n,int cols,int threads){
    try{if(!matrix||!X||!y||n<1||cols<2||threads<1||threads>256)throw std::invalid_argument("Invalid cache dimensions");int bad=0;
        #pragma omp parallel for num_threads(threads) reduction(|:bad)
        for(int i=0;i<n;++i){y[i]=matrix[size_t(i)*cols];if(y[i]<0)bad=1;for(int f=1;f<cols;++f){int64_t value=matrix[size_t(i)*cols+f];if(value<0||value>1)bad=1;X[size_t(i)*(cols-1)+f-1]=uint8_t(value);}}
        if(bad)throw std::invalid_argument("Frozen cache has nonbinary features or negative labels");return 0;
    }catch(const std::exception& e){revision::error=e.what();return 1;}}
API void* revision_data(const uint8_t* X,const int64_t* y,const double* weights,const uint8_t* allowed,const double* extras,int n,int F,int D,int early,int no_repeat,int min_leaf,int threads){
    try{revision::error.clear();return new revision::Data(X,y,weights,allowed,extras,n,F,D,early,no_repeat,min_leaf,threads);}catch(const std::exception& e){revision::error=e.what();return nullptr;}}
API void revision_data_free(void* p){delete static_cast<revision::Data*>(p);}
API const char* revision_metadata(void* p){return revision::result_call([&]{auto& d=*static_cast<revision::Data*>(p);bool uniform=true;for(auto w:d.weights)if(w!=d.weights[0])uniform=false;std::ostringstream s;s<<std::setprecision(17)<<"{\"n\":"<<d.n<<",\"F\":"<<d.F<<",\"K\":"<<d.K<<",\"min_leaf\":"<<d.min_leaf<<",\"uniform_weight\":";revision::num(s,uniform?d.weights[0]:revision::inf);s<<"}";return s.str();});}
API const char* revision_reference(void* p,double penalty){return revision::result_call([&]{auto& d=*static_cast<revision::Data*>(p);revision::Reference r(d,penalty);auto a=r.solve(0,(uint64_t(1)<<d.n)-1,0);std::ostringstream s;s<<std::setprecision(17)<<"{\"objective\":";revision::num(s,a.value);s<<",\"states\":"<<r.cache.size()<<",\"tree\":";if(std::isfinite(a.value)){if(std::abs(d.audit(a.tree,penalty)-a.value)>1e-8)throw std::runtime_error("Reference tree audit failed");d.tree_json(s,a.tree);}else s<<"null";s<<"}";return s.str();});}

API const char* revision_d3(void* p,double penalty,int kind,double seconds,uint64_t memory,int cap,int presolve,int lp_method,CostProvider provider,int batch){return revision::result_call([&]{
    using namespace revision;auto& d=*static_cast<Data*>(p);if(d.D!=3||kind<0||kind>2||!std::isfinite(penalty)||penalty<0||batch<1||cap<1)throw std::invalid_argument("Invalid D3 request");
    Budget budget(seconds,memory);auto started=Clock::now();budget.check();Workspace w(d.X,d.y.data(),d.weights.data(),d.allowed.data(),d.extras.data(),d.n,d.F,d.K,d.min_leaf,d.early,d.no_repeat,d.threads);w.provider=provider;
    double prepare=0,sc=0,gpu=0;long long raw=0;auto deadline=budget.start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    auto cols=w.enumerate(penalty,1,cap,batch,deadline,prepare,sc,gpu,raw);std::vector<double> costs(size_t(4)*w.P,revision::inf);std::vector<int> lookup(costs.size(),-1);
    #pragma omp parallel for num_threads(d.threads)
    for(int j=0;j<int(cols.size());++j){int id=cols[j].q*w.P+cols[j].s;costs[id]=cols[j].base+penalty*cols[j].split;lookup[id]=j;}
    Table t(4,d.F,d.K,costs.data(),costs.data(),d.threads,kind==1,budget);Result out=coordinate(t,kind==2?2:0,d.early?d.majority():-1,budget,cap,presolve,lp_method);
    if(out.pick.ids.empty()){std::ostringstream s;s<<"{";out.fields(s);s<<",\"raw_columns\":"<<raw<<"}";return s.str();}
    auto recovery=Clock::now();auto tree=d3_tree(d,cols,lookup,out.originals,w.P);double recovery_time=revision::elapsed(recovery);out.recovery+=recovery_time;out.total+=recovery_time;double total=revision::elapsed(started);
    std::ostringstream s;finish(s,out,d,tree,penalty);s<<",\"backend\":\""<<(provider?"cuda":"cpp_openmp")<<"\",\"raw_columns\":"<<raw<<",\"signature_columns\":"<<cols.size()<<",\"master_domain_columns\":"<<t.finite_count
        <<",\"complete_cost_table\":true,\"cost_preparation_seconds\":"<<prepare<<",\"sc_seconds\":"<<sc<<",\"gpu_callback_seconds\":"<<gpu<<",\"native_seconds\":"<<total<<",\"threads\":"<<d.threads;close_stats(s);return s.str();});}

API void* revision_fixed(void* data,int M,double penalty,const double* high,const double* low,const int32_t* private_ids,const int32_t* tails,const uint8_t* exact,int groups){
    try{revision::error.clear();auto s=std::make_unique<revision::Source>(static_cast<revision::Data*>(data),M,penalty);if(!high||!low||!private_ids||groups<0||(groups&&(!tails||!exact)))throw std::invalid_argument("Missing complete table buffers");
        for(int i=0;i<groups;++i)if(exact[i]!=1)throw std::invalid_argument("Unresolved conditional table record");size_t N=size_t(M)*s->P;s->high.assign(high,high+N);s->low.assign(low,low+N);s->private_ids.assign(private_ids,private_ids+N);s->groups=groups;if(groups)s->tails.assign(tails,tails+size_t(groups)*15);
        for(size_t i=0;i<N;++i){
            if(std::isnan(high[i])||std::isnan(low[i])||high[i]<0||low[i]<0||private_ids[i]>=groups||private_ids[i]<-1)
                throw std::invalid_argument("Invalid exact table state");
            if(high[i]!=low[i]){
                // All conditional records above must independently be marked exact.
                // Reconcile ONLY floating-point roundoff in a private copy; the
                // feasible representative cost (high) and the archive stay intact.
                if(!std::isfinite(high[i])||!std::isfinite(low[i]))throw std::invalid_argument("Invalid exact table state: finite/infinite mismatch");
                double delta=std::abs(high[i]-low[i]);
                double tol=std::min(1e-12,32*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(high[i]),std::abs(low[i])}));
                if(delta>tol)throw std::invalid_argument("Invalid exact table state: unresolved cost gap exceeds roundoff tolerance");
                ++s->import_roundoff_states;s->import_max_roundoff=std::max(s->import_max_roundoff,delta);
                s->low[i]=high[i];
            }
        }return s.release();
    }catch(const std::exception& e){revision::error=e.what();return nullptr;}}
API void revision_fixed_free(void* p){delete static_cast<revision::Source*>(p);}
API const char* revision_fixed_run(void* ptr,int kind,double seconds,uint64_t memory,int cap,int presolve,int lp_method){return revision::result_call([&]{
    using namespace revision;if(!ptr||kind<0||kind>4)throw std::invalid_argument("Invalid fixed-table request");auto& src=*static_cast<Source*>(ptr);auto& d=*src.d;Budget budget(seconds,memory);auto started=Clock::now();
    Table t(src.M,d.F,d.K,src.high.data(),src.low.data(),d.threads,kind>=3,budget);Result out=coordinate(t,kind>=3?kind-3:kind,d.majority(),budget,cap,presolve,lp_method);
    if(out.pick.ids.empty()){std::ostringstream s;s<<"{";out.fields(s);s<<"}";return s.str();}auto tick=Clock::now();auto tree=src.recover(out.originals);double recovery_time=revision::elapsed(tick);out.recovery+=recovery_time;out.total+=recovery_time;double total=revision::elapsed(started);
    std::ostringstream s;finish(s,out,d,tree,src.penalty);s<<",\"native_seconds\":"<<total<<",\"master_domain_columns\":"<<t.finite_count<<",\"remaining_clusters\":"<<t.M<<",\"threads\":"<<d.threads
        <<",\"sc_reapplication_is_identity\":true,\"conditional_oracle_calls\":0,\"backend\":\"cpp_openmp\""
        <<",\"exact_table_import_policy\":\"exact_flag_32eps_capped_1e12\",\"import_roundoff_states\":"<<src.import_roundoff_states<<",\"import_max_roundoff\":"<<src.import_max_roundoff;close_stats(s);return s.str();});}

// Generates only bounded correctness fixtures, using the independent enumerator.
API void* revision_fixture(void* data,double penalty){
    try{using namespace revision;error.clear();auto& d=*static_cast<Data*>(data);if(d.D!=4&&d.D!=5)throw std::invalid_argument("Fixture requires D4/D5");auto s=std::make_unique<Source>(&d,1<<(d.D-3),penalty);Reference ref(d,penalty);
        int N=s->M*s->P,A=d.F+d.K;s->high.assign(N,revision::inf);s->private_ids.assign(N,-1);
        for(int id=0;id<N;++id){int q=id/s->P,sig=id%s->P,root=s->M==2?sig:(sig<d.F*A?sig/A:d.F+sig-d.F*A);std::array<int,2> actions{root,sig%A};uint64_t rows=(uint64_t(1)<<d.n)-1,used=0;double cost=0;int v=0;bool valid=true,stopped=false;
            for(int j=0;j<d.D-3;++j){int a=actions[j];double mult=double(s->M>>j);
                if(a>=d.F){int k=a-d.F,count=0;double loss=0;for(int i=0;i<d.n;++i)if(rows&(uint64_t(1)<<i)){++count;if(d.y[i]!=k)loss+=d.weights[i];}if(!d.early||count<d.min_leaf)valid=false;cost+=loss/mult;stopped=true;break;}
                if(!d.allowed[size_t(v)*d.F+a]||(d.no_repeat&&(used&(uint64_t(1)<<a)))){valid=false;break;}
                cost+=(penalty+d.extras[size_t(v)*d.F+a])/mult;int side=(q>>(d.D-4-j))&1;uint64_t routed=0;for(int i=0;i<d.n;++i)if((rows&(uint64_t(1)<<i))&&d.X[size_t(i)*d.F+a]==side)routed|=uint64_t(1)<<i;rows=routed;used|=uint64_t(1)<<a;v=2*v+1+side;}
            if(!valid)continue;if(stopped){s->high[id]=cost;continue;}auto answer=ref.solve(v,rows,used);if(!std::isfinite(answer.value))continue;s->high[id]=cost+answer.value;s->private_ids[id]=s->groups++;size_t start=s->tails.size();s->tails.resize(start+15,inactive);
            std::function<void(int,int)> copy=[&](int gv,int lv){s->tails[start+lv]=answer.tree[gv];if(answer.tree[gv]>=0){copy(2*gv+1,2*lv+1);copy(2*gv+2,2*lv+2);}};copy(v,0);
        }s->low=s->high;return s.release();
    }catch(const std::exception& e){revision::error=e.what();return nullptr;}}

#include "revision_eager.hpp"
#include "revision_ondemand.hpp"
