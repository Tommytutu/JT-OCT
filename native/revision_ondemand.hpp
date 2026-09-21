// Native bound-refinement CG control. An optimistic complete-domain CG problem
// supplies the global LB; a realizing completion supplies the UB, never a message.
// Exact endpoint costs are obtained BEFORE EE. RMP is rebuilt after cost updates.
#pragma once
namespace revision {
struct OnDemand : Eager {
    using Eager::Eager;
    void evaluate(const std::vector<int>& pending){
        auto tick=Clock::now();
        try{for(size_t start=0;start<pending.size();start+=batch){
            budget.check();std::vector<int> ids(pending.begin()+start,pending.begin()+std::min(pending.size(),start+batch));
            if(private_depth==1)d2_batch(ids);else d3_batch(ids);++stats["oracle_batches"];
        }}catch(...){stats["conditional_cost_seconds"]+=elapsed(tick);throw;}
        stats["conditional_cost_seconds"]+=elapsed(tick);
    }
    void refresh(){
        source.groups=int(groups.size());source.tails.resize(groups.size()*15);
        for(int g=0;g<int(groups.size());++g)std::copy(groups[g].answer.tree.begin(),groups[g].answer.tree.end(),source.tails.begin()+size_t(g)*15);
        source.low=source.high;
        #pragma omp parallel for num_threads(d.threads) schedule(static)
        for(int id=0;id<int(source.high.size());++id)if(source.private_ids[id]>=0){
            const auto& g=groups[source.private_ids[id]];
            source.high[id]=prefix[id]+g.answer.value;
            source.low[id]=prefix[id]+(g.exact?g.answer.value:0.);
        }
    }
    std::string run_lazy(){
        bool prepared=false,endpoints_ready=kind!=4;
        std::vector<std::pair<double,double>> bounds;
        Result out;out.pick.value=incumbent_value;out.coordinator=kind==4?(source.M==2?"native_bound_refinement_single_cluster":"native_bound_refinement_cg_ee"):"native_bound_refinement_cg";checkpoint();
        try{
            prepare();prepared=true;
            if(gpu){auto tick=Clock::now();RevisionGpuBatch init{};init.phase=0;init.W=W;init.K=d.K;init.globalF=d.F;init.n=d.n;init.zero=zero.data();init.labels=classes.data();init.remaining=budget.remaining();
                if(gpu(&init))throw std::runtime_error("Resident CUDA initialization failure");stats["backend_initialization_seconds"]=elapsed(tick);budget.check();}
            if(kind==4){
                // M2: eliminate cluster 0; M4: original endpoints 0 and 3.
                // Paired columns share their entire endpoint separator, so adding
                // the exact endpoint value is the exact elimination message.
                auto tick=Clock::now();std::vector<uint8_t> seen(groups.size());std::vector<int> ids;
                for(int q=0;q<source.M;++q)if(q==0||(source.M==4&&q==3))for(int s=0;s<source.P;++s){
                    int g=source.private_ids[q*source.P+s];if(g>=0&&!groups[g].exact&&!seen[g]){seen[g]=1;ids.push_back(g);}}
                stats["endpoint_requested_private_states"]=double(ids.size());
                evaluate(ids);endpoints_ready=true;stats["endpoint_exact_preparation_seconds"]=elapsed(tick);
            }
            for(;;){
                budget.check();refresh();
                Table optimistic(source.M,d.F,d.K,source.low.data(),source.low.data(),d.threads,kind==4,budget);
                auto step=coordinate(optimistic,1,d.majority(),budget,cap,presolve,lp_method);
                ++stats["bound_refinement_rounds"];
                out.setup+=step.setup;out.update+=step.update;out.rmp+=step.rmp;out.pricing+=step.pricing;out.message+=step.message;out.ee+=step.ee;
                out.total+=step.total;out.recovery+=step.recovery;out.residual=std::max(out.residual,step.residual);out.min_active_rc=std::min(out.min_active_rc,step.min_active_rc);
                out.iterations+=step.iterations;out.pricing_calls+=step.pricing_calls;out.columns=std::max(out.columns,step.columns);out.rows=std::max(out.rows,step.rows);out.nonzeros=std::max(out.nonzeros,step.nonzeros);
                out.lb=std::max(out.lb,step.lb);out.capacity=out.capacity||step.capacity;
                if(out.lb>incumbent_value+1e-7)throw std::runtime_error("Optimistic CG bound exceeds audited incumbent");
                if(!step.originals.empty()){
                    auto tick=Clock::now();auto tree=source.recover(step.originals);double value=d.audit(tree,penalty);double recovery_time=elapsed(tick);out.recovery+=recovery_time;out.total+=recovery_time;
                    if(value<incumbent_value){incumbent=std::move(tree);incumbent_value=value;checkpoint();}
                }
                bounds.emplace_back(out.lb,incumbent_value);
                if(step.status=="ERROR"){out.status="ERROR";break;}
                if(incumbent_value-out.lb<=1e-7){out.status="OPT";break;}
                if(step.status!="OPT"){out.status=step.status;break;}
                std::vector<uint8_t> seen(groups.size());std::vector<int> ids;
                for(int id:step.originals){int g=source.private_ids[id];if(g>=0&&!groups[g].exact&&!seen[g]){seen[g]=1;ids.push_back(g);}}
                if(ids.empty())throw std::runtime_error("Certified optimistic tree has open exact gap without unresolved state");
                evaluate(ids);
            }
        }catch(const Timed&){out.status="TIME";}catch(const Limited&){out.status="MEM";}catch(const std::bad_alloc&){out.status="MEM";}
        out.pick.value=incumbent_value;double total=elapsed(budget.start);std::ostringstream s;finish(s,out,d,incumbent,penalty);
        size_t exact=0;for(auto& g:groups)exact+=g.exact;
        s<<",\"native_seconds\":"<<total<<",\"backend\":\""<<((gpu||stump_gpu)?"cuda":"cpp_openmp")<<"\",\"private_depth\":"<<private_depth<<",\"threads\":"<<d.threads<<",\"remaining_clusters\":"<<(kind==4?source.M/2:source.M)
         <<",\"exact_private_states\":"<<exact<<",\"complete_cost_table\":"<<(prepared&&exact==groups.size()?"true":"false")
         <<",\"cost_mode\":\"on_demand_bound_refinement\",\"rmp_rebuilt_after_cost_updates\":true,\"endpoint_messages_exact\":"<<(endpoints_ready?"true":"false");
        s<<",\"bound_trace\":[";for(size_t i=0;i<bounds.size();++i){if(i)s<<",";s<<"{\"LB\":"<<bounds[i].first<<",\"UB\":"<<bounds[i].second<<"}";}s<<"]";
        for(auto& kv:stats)s<<",\""<<kv.first<<"\":"<<kv.second;close_stats(s);return s.str();
    }
};
}
API const char* revision_solve(void* p,double penalty,int kind,double seconds,uint64_t memory,int cap,int presolve,int lp_method,
    RevisionGpuProvider gpu,CostProvider stump_gpu,int batch,RevisionProgress progress,int private_depth,int lazy){return revision::result_call([&]{
    if(!p||private_depth<1||private_depth>3||(lazy!=0&&lazy!=1)||(lazy&&kind!=3&&kind!=4))throw std::invalid_argument("Invalid native solver request");
    int upper=static_cast<revision::Data*>(p)->D-private_depth;
    if(upper<1||upper>2)throw std::invalid_argument("This native control supports one or two explicit upper levels only");
    revision::OnDemand e(*static_cast<revision::Data*>(p),penalty,kind,seconds,memory,cap,presolve,lp_method,gpu,stump_gpu,batch,progress,private_depth);
    return lazy?e.run_lazy():e.run();});}
