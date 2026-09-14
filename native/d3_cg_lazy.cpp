// Included by d3_cg.cpp. Uncomputed representatives retain certified lower costs.
// A bounded cache stores statistics by exact routed bitset, independently of
// ancestor feature ids, node-specific permissions, costs, and the penalty.
namespace {
struct SparseWord {
    int index; uint64_t bits;
    bool operator==(const SparseWord& b)const{return index==b.index&&bits==b.bits;}
};
struct StumpStat {double loss=INF;int left=0,right=0;};
struct CostState {
    std::vector<SparseWord> words;
    std::vector<StumpStat> stumps;
    bool complete=false;
    uint64_t hash=1469598103934665603ULL;
    size_t bytes()const{return sizeof(*this)+words.size()*sizeof(SparseWord)+stumps.size()*sizeof(StumpStat);}
};
struct LazyCosts {
    Workspace& w;
    std::vector<uint8_t> done;
    std::vector<double> split_lower;
    bool metadata=false;
    std::unordered_map<uint64_t,std::vector<std::shared_ptr<CostState>>> cache;
    std::deque<std::shared_ptr<CostState>> fifo;
    size_t cache_bytes=0,peak_cache_bytes=0;
    uint64_t hits=0,misses=0,word_visits=0,stump_evaluations=0,batches=0;
    uint64_t gpu_batches=0;
    double gpu_seconds=0.;
    int adaptive_batch=128;
    explicit LazyCosts(Workspace& ws):w(ws),done(size_t(4)*w.F*w.F,0),split_lower(done.size(),INF){}
    int index(int q,int f,int g)const{return (q*w.F+f)*w.F+g;}
    int completed()const{return (int)std::count(done.begin(),done.end(),uint8_t(1));}
    void prepare_metadata(Clock::time_point deadline){
        if(metadata)return;
        w.pack(deadline);
        if(!w.branches){
            w.branch_size.assign(2*w.F,0);w.branch_loss.assign(size_t(2)*w.F*w.K,INF);
            for(int f=0;f<w.F;++f)for(int a=0;a<2;++a){
                if(Clock::now()>=deadline)throw Timeout{};
                std::vector<double> c(w.K,0.);int total=0;
                for(int j=0;j<w.W;++j){auto bits=w.branch(f,a,j);total+=pop(bits);
                    for(int k=0;k<w.K;++k)c[k]+=w.mass(bits&w.classes[size_t(k)*w.W+j],j);}
                w.branch_size[2*f+a]=total;
                double sum=std::accumulate(c.begin(),c.end(),0.);
                for(int k=0;k<w.K;++k)w.branch_loss[(size_t(2*f+a)*w.K)+k]=std::max(0.,sum-c[k]);
            }w.branches=true;
        }
        std::array<double,4> extra;extra.fill(INF);
        for(int q=0;q<4;++q)for(int h=0;h<w.F;++h)
            if(w.allowed[(3+q)*w.F+h])extra[q]=std::min(extra[q],w.extras[(3+q)*w.F+h]);
        std::atomic<bool> timeout(false);
        const int pairs=w.F*w.F,workers=std::max(1,w.threads);
        #pragma omp parallel for schedule(dynamic,16) num_threads(workers)
        for(int pair=0;pair<pairs;++pair){
            if(Clock::now()>=deadline){timeout=true;continue;}
            int f=pair/w.F,g=pair%w.F;
            for(int q=0;q<4;++q){
                int id=q*pairs+pair,a=q/2,b=q%2;
                if(w.ready[pair]){done[id]=1;continue;}
                if(!w.allowed[f]||!w.allowed[(1+a)*w.F+g]||(w.no_repeat&&f==g)){
                    done[id]=1;continue;
                }
                auto& tail=w.tails[id];std::vector<double> c(w.K,0.);int total=0;
                for(int j=0;j<w.W;++j){auto bits=w.branch(f,a,j)&w.branch(g,b,j);total+=pop(bits);
                    for(int k=0;k<w.K;++k)c[k]+=w.mass(bits&w.classes[size_t(k)*w.W+j],j);}
                auto stop=Workspace::leaf(c);tail.label=stop.first;
                if(total>=w.min_leaf)tail.stop=stop.second;
                // Two leaves can predict at most two labels. The mass outside
                // the two largest classes must be misclassified, for any h.
                double first=0.,second=0.,sum=0.;
                for(double x:c){sum+=x;if(x>first){second=first;first=x;}else second=std::max(second,x);}
                if(total>=2*w.min_leaf)split_lower[id]=std::max(0.,sum-first-second)+extra[q];
                if((w.early&&tail.stop==0.)||!std::isfinite(split_lower[id]))done[id]=1;
            }
        }
        if(timeout||Clock::now()>=deadline)throw Timeout{};
        metadata=true;refresh_ready();
    }
    void refresh_ready(){
        int pairs=w.F*w.F;
        for(int p=0;p<pairs;++p)w.ready[p]=done[p]&&done[pairs+p]&&done[2*pairs+p]&&done[3*pairs+p];
    }
    std::vector<Choice> table(double penalty,bool lower)const{
        auto tab=w.table(penalty);
        for(int q=0;q<4;++q)for(int f=0;f<w.F;++f)for(int g=0;g<w.F;++g){
            int id=index(q,f,g),s=f*w.A+g;
            if(done[id])continue;
            auto& c=tab[q*w.P+s];c.cost=INF;
            if(lower){
                double tail=split_lower[id]+penalty;
                if(w.early)tail=std::min(tail,w.tails[id].stop);
                c.cost=.25*(penalty+w.extras[f])+.5*(penalty+w.extras[(1+q/2)*w.F+g])+tail;
            }
        }return tab;
    }
    std::shared_ptr<CostState> state(int q,int f,int g){
        auto s=std::make_shared<CostState>();
        for(int j=0;j<w.W;++j){auto bits=w.branch(f,q/2,j)&w.branch(g,q%2,j);
            if(bits){s->words.push_back({j,bits});s->hash^=(uint64_t)j;s->hash*=1099511628211ULL;
                s->hash^=bits;s->hash*=1099511628211ULL;}}
        return s;
    }
    void calculate(CostState& s,Clock::time_point deadline){
        const int nz=(int)s.words.size(),K=w.K;
        std::vector<double> counts(K,0.),left(K,0.),right(K,0.);
        std::vector<uint64_t> masks(size_t(K)*nz);
        int total=0;
        for(int j=0;j<nz;++j){const auto& r=s.words[j];total+=pop(r.bits);
            for(int k=0;k<K;++k){auto bits=r.bits&w.classes[size_t(k)*w.W+r.index];
                masks[size_t(k)*nz+j]=bits;counts[k]+=w.mass(bits,r.index);}}
        s.stumps.resize(w.F);
        for(int h=0;h<w.F;++h){
            if(h%8==0&&Clock::now()>=deadline)throw Timeout{};
            const auto* feature=w.ones.data()+size_t(h)*w.W;int n0=0;
            std::fill(left.begin(),left.end(),0.);
            if(w.uniform){
                // Uniform multiclass: integer counts, only K-1 class popcounts.
                // Empty words are absent; the final class follows by subtraction.
                if(K==3){
                    int c0=0,c1=0;
                    for(int j=0;j<nz;++j){auto inv=~feature[s.words[j].index];
                        n0+=pop(s.words[j].bits&inv);c0+=pop(masks[j]&inv);c1+=pop(masks[nz+j]&inv);}
                    left[0]=c0*w.unit;left[1]=c1*w.unit;left[2]=(n0-c0-c1)*w.unit;
                }else{
                    std::vector<int> c(K,0);
                    for(int j=0;j<nz;++j){auto inv=~feature[s.words[j].index];n0+=pop(s.words[j].bits&inv);
                        for(int k=0;k<K-1;++k)c[k]+=pop(masks[size_t(k)*nz+j]&inv);}
                    int used=0;for(int k=0;k<K-1;++k){used+=c[k];left[k]=c[k]*w.unit;}
                    left[K-1]=(n0-used)*w.unit;
                }
            }else{
                for(int j=0;j<nz;++j){auto inv=~feature[s.words[j].index];n0+=pop(s.words[j].bits&inv);
                    for(int k=0;k<K;++k)left[k]+=w.mass(masks[size_t(k)*nz+j]&inv,s.words[j].index);}
            }
            if(n0<w.min_leaf||total-n0<w.min_leaf)continue;
            for(int k=0;k<K;++k)right[k]=std::max(0.,counts[k]-left[k]);
            auto l=Workspace::leaf(left),r=Workspace::leaf(right);
            s.stumps[h]={l.second+r.second,l.first,r.first};
        }s.complete=true;
    }
    void admit(const std::shared_ptr<CostState>& s){
        const size_t cap=size_t(w.cache_mb)*1024*1024;
        if(s->bytes()>cap)return;
        while(cache_bytes+s->bytes()>cap&&!fifo.empty()){
            auto old=fifo.front();fifo.pop_front();cache_bytes-=old->bytes();
            auto it=cache.find(old->hash);auto& group=it->second;
            group.erase(std::remove(group.begin(),group.end(),old),group.end());if(group.empty())cache.erase(it);
        }
        cache[s->hash].push_back(s);fifo.push_back(s);cache_bytes+=s->bytes();peak_cache_bytes=std::max(peak_cache_bytes,cache_bytes);
    }
    void compute_pairs(const std::vector<int>& pairs,Clock::time_point deadline){
        auto started=Clock::now();
        struct Task{int id,q,f,g;std::shared_ptr<CostState> state;};
        std::vector<Task> tasks;std::vector<std::shared_ptr<CostState>> fresh;
        std::unordered_map<uint64_t,std::vector<std::shared_ptr<CostState>>> pending;
        for(int pair:pairs)for(int q=0;q<4;++q){
            int f=pair/w.F,g=pair%w.F,id=index(q,f,g);if(done[id])continue;
            if(Clock::now()>=deadline)throw Timeout{};
            auto s=state(q,f,g),found=std::shared_ptr<CostState>();
            auto lookup=[&](const auto& map){auto it=map.find(s->hash);if(it!=map.end())for(const auto& t:it->second)
                if(t->words==s->words){found=t;break;}};
            lookup(cache);if(!found)lookup(pending);
            if(found){s=found;++hits;}else{pending[s->hash].push_back(s);fresh.push_back(s);++misses;}
            tasks.push_back({id,q,f,g,s});
        }
        std::atomic<bool> timeout(false);const int workers=std::max(1,w.threads),count=(int)fresh.size();
        if(w.cost_provider&&count){
            if(!w.uniform)throw std::runtime_error("GPU cost provider requires uniform weights");
            const int chunk=std::max(1,int((64ULL*1024*1024)/(size_t(w.W)*8+size_t(w.F)*24)));
            for(int offset=0;offset<count;offset+=chunk){
                if(Clock::now()>=deadline){timeout=true;break;}
                int nstate=std::min(chunk,count-offset);
                std::vector<uint64_t> rows(size_t(nstate)*w.W,0);
                std::vector<double> loss(size_t(nstate)*w.F);
                std::vector<int32_t> labels(size_t(nstate)*w.F*2);
                for(int i=0;i<nstate;++i)for(const auto& r:fresh[offset+i]->words)rows[size_t(i)*w.W+r.index]=r.bits;
                auto t=Clock::now();int code=w.cost_provider(nstate,w.ones.data(),w.classes.data(),rows.data(),loss.data(),labels.data(),
                    std::max(0.,std::chrono::duration<double>(deadline-Clock::now()).count()));gpu_seconds+=elapsed(t);
                if(code==2)break; // Explicitly recorded automatic CPU fallback.
                if(code==-1){timeout=true;break;}
                if(code!=1)throw std::runtime_error("GPU D3 cost callback failed");
                ++gpu_batches;
                for(int i=0;i<nstate;++i){auto& s=*fresh[offset+i];s.stumps.resize(w.F);
                    for(int h=0;h<w.F;++h){size_t j=size_t(i)*w.F+h;s.stumps[h]={loss[j],labels[2*j],labels[2*j+1]};}
                    s.complete=true;
                }
            }
        }
        #pragma omp parallel for schedule(dynamic,1) num_threads(workers)
        for(int i=0;i<count;++i){if(fresh[i]->complete)continue;
            try{calculate(*fresh[i],deadline);}catch(const Timeout&){timeout=true;}}
        for(const auto& s:fresh)if(s->complete){
            word_visits+=uint64_t(s->words.size())*w.F;stump_evaluations+=w.F;admit(s);
        }
        for(const auto& t:tasks)if(t.state->complete){
            auto& tail=w.tails[t.id];tail.split=INF;tail.h=-1;
            if(!(w.early&&tail.stop==0.))for(int h=0;h<w.F;++h){
                if(!w.allowed[(3+t.q)*w.F+h]||(w.no_repeat&&(h==t.f||h==t.g)))continue;
                const auto& c=t.state->stumps[h];double loss=c.loss+w.extras[(3+t.q)*w.F+h];
                if(loss<tail.split){tail.split=loss;tail.h=h;tail.left=c.left;tail.right=c.right;}
            }done[t.id]=1;
        }
        ++batches;refresh_ready();
        if(w.cost_batch==0&&!pairs.empty()){
            int desired=int(pairs.size()*.25/std::max(.001,elapsed(started)));
            int memory_limit=int((128ULL*1024*1024)/(4*(size_t(w.W)*sizeof(SparseWord)+size_t(w.F)*sizeof(StumpStat)+1)));
            int maximum=std::max(128,std::min(4096,memory_limit));
            adaptive_batch=std::max(128,std::min({maximum,desired,adaptive_batch*4}));
        }
        if(timeout||Clock::now()>=deadline)throw Timeout{};
    }
    std::vector<int> candidates(const std::vector<Choice>& lower,const std::vector<double>& rc)const{
        struct Rank{double joint,dual,stop;int pair;};
        std::vector<Rank> rank;std::vector<double> left(w.F,INF),right(w.F,INF);
        for(int f=0;f<w.F;++f)for(int g=0;g<w.F;++g){int s=f*w.A+g;
            left[f]=std::min(left[f],lower[s].cost+lower[w.P+s].cost);
            right[f]=std::min(right[f],lower[2*w.P+s].cost+lower[3*w.P+s].cost);}
        for(int f=0;f<w.F;++f)for(int g=0;g<w.F;++g){
            int s=f*w.A+g;double r=INF,hint=0.;bool unknown=false;
            for(int q=0;q<4;++q){int id=index(q,f,g);hint+=w.tails[id].stop;
                if(!done[id]){unknown=true;r=std::min(r,rc[q*w.P+s]);}}
            if(!unknown||r>=-TOL)continue;
            double joint=std::min(lower[s].cost+lower[w.P+s].cost+right[f],
                lower[2*w.P+s].cost+lower[3*w.P+s].cost+left[f]);
            rank.push_back({joint,r,hint,f*w.F+g});
        }
        auto cmp=[](const Rank& a,const Rank& b){if(a.joint!=b.joint)return a.joint<b.joint;
            if(a.dual!=b.dual)return a.dual<b.dual;if(a.stop!=b.stop)return a.stop<b.stop;return a.pair<b.pair;};
        int count=std::min(w.cost_batch?w.cost_batch:adaptive_batch,(int)rank.size());
        if(count<(int)rank.size())std::nth_element(rank.begin(),rank.begin()+count,rank.end(),cmp);
        rank.resize(count);std::sort(rank.begin(),rank.end(),cmp);
        std::vector<int> out;for(const auto& r:rank)out.push_back(r.pair);return out;
    }
};

void solve_lazy(Workspace& w,double penalty,double seconds,int batch,int cap,int method,Reporter report){
    auto start=Clock::now(),deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    if(!w.lazy)w.lazy=std::make_shared<LazyCosts>(w);auto& costs=*w.lazy;
    std::string status="RUNNING";Selection selected;std::vector<Choice> tab;std::vector<Event> trace;
    int iterations=0,columns=0,rows=0;double lb=0.,first=INF,proof=INF,initial=0.,compute=0.,pricing=0.,rmp=0.,updates=0.,environment=0.,first_rmp=INF,message_lb=0.;
    bool reused=costs.metadata;int before_done=costs.completed();
    auto before_hits=costs.hits,before_misses=costs.misses,before_stumps=costs.stump_evaluations,before_words=costs.word_visits,before_batches=costs.batches;
    auto before_gpu=costs.gpu_batches;double before_gpu_seconds=costs.gpu_seconds,callback_seconds=0.;
    auto json=[&](){
        std::ostringstream os;os<<std::setprecision(17);int major,minor,technical;GRBversion(&major,&minor,&technical);
        os<<"{\"status\":\""<<status<<"\",\"native_seconds\":"<<elapsed(start)<<",\"LB\":";number(os,lb);
        os<<",\"UB\":";number(os,selected.value);os<<",\"tree\":"<<w.tree(selected,tab)
          <<",\"iterations\":"<<iterations<<",\"columns\":"<<columns<<",\"rows\":"<<rows
          <<",\"pricing_setup_seconds\":"<<initial+compute<<",\"initial_setup_seconds\":"<<initial
          <<",\"cost_evaluation_seconds\":"<<compute<<",\"pricing_seconds\":"<<pricing<<",\"rmp_seconds\":"<<rmp
          <<",\"master_update_seconds\":"<<updates<<",\"environment_seconds\":"<<environment
          <<",\"workspace_reused\":"<<(reused?"true":"false")<<",\"cost_mode\":\"lazy\",\"cost_batch_size\":"<<w.cost_batch
          <<",\"cost_batch_final\":"<<(w.cost_batch?w.cost_batch:costs.adaptive_batch)
          <<",\"cost_backend\":\""<<(costs.gpu_batches>before_gpu?"cuda":"cpp_openmp")<<"\",\"gpu_cost_batches\":"<<costs.gpu_batches-before_gpu
          <<",\"gpu_cost_seconds\":"<<costs.gpu_seconds-before_gpu_seconds<<",\"callback_seconds\":"<<callback_seconds
          <<",\"cost_states_resolved\":"<<costs.completed()<<",\"cost_states_resolved_this_call\":"<<costs.completed()-before_done
          <<",\"cost_states_total\":"<<costs.done.size()<<",\"cost_table_complete\":"<<(costs.completed()==(int)costs.done.size()?"true":"false")
          <<",\"cost_cache_hits\":"<<costs.hits-before_hits<<",\"cost_cache_misses\":"<<costs.misses-before_misses
          <<",\"cost_cache_peak_bytes\":"<<costs.peak_cache_bytes<<",\"stump_evaluations\":"<<costs.stump_evaluations-before_stumps
          <<",\"cost_word_visits\":"<<costs.word_visits-before_words<<",\"cost_batches\":"<<costs.batches-before_batches
          <<",\"message_lower_bound\":";number(os,message_lb);
        os<<",\"first_rmp_seconds\":";number(os,first_rmp);os<<",\"first_final_ub_seconds\":";number(os,first);
        os<<",\"proof_seconds\":";number(os,proof);os<<",\"gurobi_version\":\""<<major<<"."<<minor<<"."<<technical<<"\",\"trace\":[";
        for(size_t j=0;j<trace.size();++j){const auto& e=trace[j];if(j)os<<",";
            os<<"{\"iteration\":"<<e.iteration<<",\"seconds\":"<<e.seconds<<",\"LB\":"<<e.lb<<",\"UB\":"<<e.ub
              <<",\"columns\":"<<e.columns<<",\"rows\":"<<e.rows<<",\"new_columns\":"<<e.new_columns
              <<",\"alpha_sum\":"<<e.alpha<<",\"pricing_lower_bounds\":[";
            for(int q=0;q<4;++q){if(q)os<<",";os<<e.minima[q];}
            os<<"],\"max_equality_residual\":"<<e.residual<<",\"minimum_active_reduced_cost\":"<<e.active_rc<<"}";
        }os<<"]}";w.output=os.str();return w.output.c_str();
    };
    auto check=[&](){if(Clock::now()>=deadline)throw Timeout{};};
    double last_publish=-INF,last_published_ub=INF;
    auto publish=[&](){
        if(report&&(status!="RUNNING"||selected.value<last_published_ub-1e-12||elapsed(start)-last_publish>=.5)){
            auto t=Clock::now();report(json());callback_seconds+=elapsed(t);
            last_publish=elapsed(start);last_published_ub=selected.value;
        }
    };
    auto compute_pairs=[&](const std::vector<int>& pairs){auto t=Clock::now();
        try{costs.compute_pairs(pairs,deadline);}catch(...){compute+=elapsed(t);throw;}compute+=elapsed(t);tab=costs.table(penalty,false);};
    try{
        check();w.count_global();
        if(w.early&&w.n>=w.min_leaf){int k=Workspace::leaf(w.counts).first,s=w.F*w.A+k;
            tab=w.table(penalty);selected={4*tab[s].cost,{s,s}};first=elapsed(start);publish();}
        auto t=Clock::now();try{costs.prepare_metadata(deadline);}catch(...){initial+=elapsed(t);throw;}initial+=elapsed(t);
        tab=costs.table(penalty,false);
        // With mandatory splits, discover a feasible compatible four-block seed.
        while(selected.prefix[0]<0){
            selected=w.glue(tab,nullptr,true);if(selected.prefix[0]>=0){first=elapsed(start);break;}
            auto lower=costs.table(penalty,true);auto candidate=w.glue(lower);
            if(candidate.prefix[0]<0){status="INFEASIBLE";lb=INF;json();return;}
            std::vector<int> pairs;for(int s:candidate.prefix)if(s<w.F*w.A&&s%w.A<w.F)pairs.push_back((s/w.A)*w.F+s%w.A);
            std::sort(pairs.begin(),pairs.end());pairs.erase(std::unique(pairs.begin(),pairs.end()),pairs.end());
            if(pairs.empty())throw std::runtime_error("Lazy D3 has no feasible seed or unresolved prefix");
            compute_pairs(pairs);check();
        }
        check();t=Clock::now();
        if(!w.env){auto env=std::make_unique<GRBEnv>(true);env->set(GRB_IntParam_OutputFlag,0);env->start();w.env=std::move(env);}
        environment+=elapsed(t);t=Clock::now();
        GRBModel model(*w.env);model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260905);
        model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        std::vector<GRBConstr> norm;for(int q=0;q<4;++q)norm.push_back(model.addConstr(GRBLinExpr()==1.));
        std::vector<GRBConstr> sep(w.edge_count());std::vector<uint8_t> has_row(w.edge_count(),0),active(size_t(4)*w.P,0);
        std::vector<GRBVar> vars(size_t(4)*w.P);std::vector<int> indices;rows=4;updates+=elapsed(t);
        auto add=[&](const std::vector<int>& ids){
            if(columns+(int)ids.size()>cap)throw std::length_error("Restricted column limit");auto begin=Clock::now();
            for(int id:ids){int q=id/w.P,s=id%w.P;if(!std::isfinite(tab[id].cost))throw std::runtime_error("Uncomputed column added");
                for(int slot=0;slot<w.degree(q);++slot){int e=w.edge(q,s,slot);if(!has_row[e]){sep[e]=model.addConstr(GRBLinExpr()==0.);has_row[e]=1;++rows;}}}
            model.update();for(int id:ids)if(!active[id]){int q=id/w.P,s=id%w.P;GRBColumn col;col.addTerm(1.,norm[q]);
                for(int slot=0;slot<w.degree(q);++slot)col.addTerm(w.sign(q,slot),sep[w.edge(q,s,slot)]);
                vars[id]=model.addVar(0.,GRB_INFINITY,tab[id].cost,GRB_CONTINUOUS,col);active[id]=1;indices.push_back(id);++columns;
            }model.update();updates+=elapsed(begin);
        };
        add({selected.prefix[0],w.P+selected.prefix[0],2*w.P+selected.prefix[1],3*w.P+selected.prefix[1]});
        for(int iteration=0;iteration<4*w.P+2;++iteration){
            check();model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,std::chrono::duration<double>(deadline-Clock::now()).count()));
            if(!std::isfinite(first_rmp))first_rmp=elapsed(start);
            t=Clock::now();model.optimize();rmp+=elapsed(t);check();int code=model.get(GRB_IntAttr_Status);
            if(code==GRB_TIME_LIMIT||code==GRB_INTERRUPTED)throw Timeout{};
            if(code!=GRB_OPTIMAL)throw std::runtime_error("Lazy D3 RMP failed, status "+std::to_string(code));
            ++iterations;std::vector<uint8_t> support(size_t(4)*w.P,0);
            for(int id:indices)if(vars[id].get(GRB_DoubleAttr_X)>1e-9)support[id]=1;
            auto candidate=w.glue(tab,&support);
            if(candidate.prefix[0]<0||std::abs(candidate.value-model.get(GRB_DoubleAttr_ObjVal))>1e-7)throw std::runtime_error("Lazy D3 gluing audit failed");
            if(candidate.value<selected.value-1e-12){selected=candidate;first=elapsed(start);}
            std::array<double,4> alpha;for(int q=0;q<4;++q)alpha[q]=norm[q].get(GRB_DoubleAttr_Pi);
            std::vector<double> pi(w.edge_count(),0.);for(int e=0;e<w.edge_count();++e)if(has_row[e])pi[e]=sep[e].get(GRB_DoubleAttr_Pi);
            double asum=std::accumulate(alpha.begin(),alpha.end(),0.),residual=0.,active_rc=INF;
            for(int q=0;q<4;++q)residual=std::max(residual,std::abs(norm[q].get(GRB_DoubleAttr_Slack)));
            for(int e=0;e<w.edge_count();++e)if(has_row[e])residual=std::max(residual,std::abs(sep[e].get(GRB_DoubleAttr_Slack)));
            for(int id:indices)active_rc=std::min(active_rc,vars[id].get(GRB_DoubleAttr_RC));
            if(residual>1e-7||active_rc<-1e-7)throw std::runtime_error("Lazy D3 RMP dual/residual audit failed");
            std::vector<int> seeds;
            while(seeds.empty()){
                check();t=Clock::now();auto lower=costs.table(penalty,true);std::vector<double> rc,rc_lower;
                auto minima=price(w,lower,alpha.data(),pi.data(),rc_lower);price(w,tab,alpha.data(),pi.data(),rc);
                double certificate=asum+std::accumulate(minima.begin(),minima.end(),0.);
                // Exact JT gluing of optimistic local costs is another valid LB.
                message_lb=w.glue(lower).value;certificate=std::max(certificate,message_lb);
                if(certificate>selected.value+1e-7)throw std::runtime_error("Lazy D3 lower bound exceeds a feasible tree");
                lb=std::max(lb,std::min(selected.value,certificate));
                for(int q=0;q<4;++q){std::vector<int> ids;
                    for(int s=0;s<w.P;++s){int id=q*w.P+s;if(!active[id]&&rc[id]<-TOL)ids.push_back(id);}
                    if(batch>0&&(int)ids.size()>batch){auto cmp=[&](int a,int b){return rc[a]!=rc[b]?rc[a]<rc[b]:a<b;};
                        std::nth_element(ids.begin(),ids.begin()+batch,ids.end(),cmp);ids.resize(batch);}
                    std::sort(ids.begin(),ids.end());seeds.insert(seeds.end(),ids.begin(),ids.end());
                }
                pricing+=elapsed(t);trace.push_back({elapsed(start),lb,selected.value,asum,residual,active_rc,minima,iteration,columns,rows,(int)seeds.size()});
                if(selected.value-lb<=TOL){check();status="OPT";proof=elapsed(start);publish();json();return;}
                publish();check();if(!seeds.empty())break;
                t=Clock::now();auto pairs=costs.candidates(lower,rc_lower);pricing+=elapsed(t);
                if(pairs.empty())throw std::runtime_error("Lazy pricing exhausted candidates with an open gap");
                compute_pairs(pairs);
            }
            check();add(seeds);
        }throw std::runtime_error("Finite lazy D3 domain did not converge");
    }catch(const Timeout&){status="TIME";}catch(const std::length_error&){status="RESOURCE";}
    json();publish();
}
}
