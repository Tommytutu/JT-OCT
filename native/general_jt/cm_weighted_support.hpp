#pragma once
// Anytime conditional weighted OCT. All partial maximizations retain an upper
// bound for EVERY unvisited branch. Optimization is C++; Python only audits.
namespace cm5 {
struct WeightedBound {double lo=0,hi=0;Tree tree;};
struct WeightedWorker {
    Context structure;
    std::map<std::tuple<std::string,int,int>,WeightedBound> memo;
    double a,b,gamma;U visits=0,hits=0,prunes=0;size_t limit=12000;std::array<double,32> global_caps;
    WeightedWorker(const Data&d,double aa,double bb,double g):structure(d),a(aa),b(bb),gamma(g){structure.incremental_cache=true;global_caps.fill(INF);}
    WeightedBound base(const Rows&r,int depth,int budget)const{
        WeightedBound v;v.tree.a[1]=a*r->p>b*r->n?-2:-1;
        v.lo=std::max(a*r->p,b*r->n);
        v.hi=depth&&budget?std::max(v.lo,a*r->p+b*r->n-gamma):v.lo;
        if(global_caps[budget]<v.lo-1e-6)throw std::runtime_error("shared support contradicts conditional leaf");
        v.hi=std::max(v.lo,std::min(v.hi,global_caps[budget]));return v;
    }
    WeightedBound peek(const Rows&r,int d,int k){k=std::min(k,(1<<d)-1);auto it=memo.find({r->key,d,k});
        if(it!=memo.end()){++hits;auto v=it->second;
            if(global_caps[k]<v.lo-1e-6)throw std::runtime_error("shared support contradicts conditional witness");
            v.hi=std::max(v.lo,std::min(v.hi,global_caps[k]));return v;}return base(r,d,k);}
    WeightedBound solve(const Rows&r,int d,int k,double end){
        k=std::min(k,(1<<d)-1);auto key=std::make_tuple(r->key,d,k);WeightedBound out=peek(r,d,k);
        if(out.hi<=out.lo+1e-10||now()>=end)return out;
        ++visits;auto order=structure.order(r);
        if(d==1||k==1){
            // Exact scalar stump terminal: class counts suffice; do not
            // allocate two masks and two memo entries for every candidate.
            for(auto c:*order){double v=std::max(a*c.p0,b*c.n0)+std::max(a*(r->p-c.p0),b*(r->n-c.n0))-gamma;
                if(v>out.lo){out.lo=v;out.tree=Tree();out.tree.a[1]=c.f;
                    out.tree.a[2]=a*c.p0>b*c.n0?-2:-1;out.tree.a[3]=a*(r->p-c.p0)>b*(r->n-c.n0)?-2:-1;}}
            out.hi=out.lo;if(memo.size()>=limit)memo.clear();memo[key]=out;return out;
        }
        struct Branch {int f,i,j;Rows l,r;double upper;};std::vector<Branch> branches;
        int cap=(1<<(d-1))-1;
        for(auto c:*order){auto children=structure.data.partition(r,c.f);
            for(int i=std::max(0,k-1-cap);i<=std::min(cap,k-1);++i){int j=k-1-i;
                auto left=peek(children[0],d-1,i),right=peek(children[1],d-1,j);
                double ub=left.hi+right.hi-gamma;
                if(ub<=out.lo+1e-10){++prunes;continue;}
                branches.push_back({c.f,i,j,children[0],children[1],ub});}
        }
        std::stable_sort(branches.begin(),branches.end(),[](const Branch&x,const Branch&y){return x.upper>y.upper;});
        double upper=out.lo;
        for(auto&branch:branches){
            if(branch.upper<=out.lo+1e-10){upper=std::max(upper,branch.upper);++prunes;continue;}
            if(now()>=end){upper=std::max(upper,branch.upper);continue;}
            auto left=solve(branch.l,d-1,branch.i,end),right=peek(branch.r,d-1,branch.j);
            if(left.hi+right.hi-gamma>out.lo+1e-10)right=solve(branch.r,d-1,branch.j,end);
            double feasible=left.lo+right.lo-gamma;
            if(feasible>out.lo){out.lo=feasible;out.tree=Tree();out.tree.a[1]=branch.f;transplant(out.tree,2,left.tree);transplant(out.tree,3,right.tree);}
            upper=std::max(upper,left.hi+right.hi-gamma);
        }
        out.hi=std::max(out.lo,std::min(out.hi,upper));
        // Eviction discards information, never invents a tighter certificate.
        if(memo.size()>=limit)memo.clear();memo[key]=out;return out;
    }
};
struct WeightedOracle {
    const Data&data;double a,b,gamma;int depth,budget,threads;
    struct RootBranch {int f,i,j;WeightedBound bound;};
    std::vector<RootBranch> branches;std::vector<std::unique_ptr<WeightedWorker>> workers;
    WeightedBound root;U rounds=0;double seconds=0;
    WeightedOracle(const Data&d,double aa,double bb,double g,int dep,int k,int nt):data(d),a(aa),b(bb),gamma(g),depth(dep),budget(std::min(k,(1<<dep)-1)),threads(nt){
        for(int j=0;j<threads;++j)workers.push_back(std::make_unique<WeightedWorker>(data,a,b,gamma));
        root=workers[0]->base(data.all,depth,budget);
        if(!depth||!budget)return;
        Context ctx(data);int cap=(1<<(depth-1))-1;
        for(auto c:*ctx.order(data.all))for(int i=std::max(0,budget-1-cap);i<=std::min(cap,budget-1);++i){int j=budget-1-i;
            auto l=workers[0]->base(data.root_rows[c.f][0],depth-1,i),r=workers[0]->base(data.root_rows[c.f][1],depth-1,j);
            WeightedBound z;z.lo=l.lo+r.lo-gamma;z.hi=l.hi+r.hi-gamma;z.tree.a[1]=c.f;transplant(z.tree,2,l.tree);transplant(z.tree,3,r.tree);
            branches.push_back({c.f,i,j,z});}
        refresh();
    }
    void seed(const Tree&t){auto c=data.counts(t,data.all,1,depth);if(c.k>budget)return;
        double v=a*c.tp+b*c.tn-gamma*c.k;if(v>root.lo){root.lo=v;root.tree=t;}root.hi=std::max(root.hi,root.lo);}
    void refresh(){double hi=workers[0]->base(data.all,depth,budget).lo;
        for(auto&j:branches){if(j.bound.lo>root.lo){root.lo=j.bound.lo;root.tree=j.bound.tree;}hi=std::max(hi,j.bound.hi);}
        root.hi=std::max(root.lo,std::min(root.hi,hi));}
    void shared_caps(const std::array<double,32>&caps){
        // A subset tree extends to the full data with unchanged K and gamma;
        // nonnegative class rewards only increase. Caps are for AT MOST k.
        if(caps[budget]<root.lo-1e-6)throw std::runtime_error("shared support contradicts root witness");
        for(auto&w:workers)w->global_caps=caps;
        root.hi=std::max(root.lo,std::min(root.hi,caps[budget]));
    }
    bool complete()const{return root.hi-root.lo<=1e-8;}
    void advance(double end){double start=now();
        while(now()<end&&!complete()){
            std::vector<size_t> jobs;for(size_t i=0;i<branches.size();++i)if(branches[i].bound.hi>root.lo+1e-9)jobs.push_back(i);
            std::stable_sort(jobs.begin(),jobs.end(),[&](size_t i,size_t j){return branches[i].bound.hi>branches[j].bound.hi;});
            // Process the globally optimistic root allocations first, with
            // bounded slices so a difficult allocation cannot starve the rest.
            std::vector<std::exception_ptr> errors(jobs.size());double incumbent=root.lo;
            #pragma omp parallel for schedule(dynamic,1) num_threads(threads)
            for(int q=0;q<int(jobs.size());++q)try{
                if(now()>=end)continue;auto&job=branches[jobs[q]];auto&w=*workers[omp_get_thread_num()];auto children=data.root_rows[job.f];
                double stop=std::min(end,now()+.025);
                auto left=w.solve(children[0],depth-1,job.i,stop),right=w.peek(children[1],depth-1,job.j);
                if(left.hi+right.hi-gamma>incumbent+1e-9)right=w.solve(children[1],depth-1,job.j,stop);
                double lo=left.lo+right.lo-gamma;
                if(lo>job.bound.lo){job.bound.lo=lo;job.bound.tree=Tree();job.bound.tree.a[1]=job.f;transplant(job.bound.tree,2,left.tree);transplant(job.bound.tree,3,right.tree);}
                job.bound.hi=std::max(job.bound.lo,std::min(job.bound.hi,left.hi+right.hi-gamma));
            }catch(...){errors[q]=std::current_exception();}
            for(auto&e:errors)if(e)std::rethrow_exception(e);refresh();++rounds;
        }
        seconds+=now()-start;
    }
    double cost_lower()const{return a*data.P+b*data.N-root.hi-1e-7*std::max(1.,a+b+gamma);}
};
}
