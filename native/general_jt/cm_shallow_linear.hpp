#pragma once
#include <iomanip>
#include <mutex>
namespace cm5 {
// Objective-independent D2 terminal. As in the D3 terminal, share class-count
// geometry across directions and return every at-most split budget at once.
// Only the upper triangle is stored: one intersection gives both child stumps.
struct ShallowLinearStore {
    struct Pair {int p=0,n=0;};
    struct Geometry {
        std::vector<Pair> marginal,joint;
        std::vector<unsigned char> done;
        bool complete=false;
    };
    const Data&data;size_t limit,bytes=0;std::mutex mutex;
    std::unordered_map<std::string,std::shared_ptr<Geometry>> cache;
    U hits=0,queries=0,resumes=0,limit_events=0;
    double geometry_seconds=0,query_seconds=0;
    ShallowLinearStore(const Data&d,size_t cap):data(d),limit(cap){}
    size_t index(int f,int g)const {if(f>g)std::swap(f,g);return size_t(f)*(2*size_t(data.F)-f-1)/2+g-f-1;}
    std::shared_ptr<Geometry> get(const Rows&r,int threads,double end){
        std::lock_guard<std::mutex> lock(mutex);
        auto found=cache.find(r->key);std::shared_ptr<Geometry> out;
        if(found!=cache.end()){out=found->second;++hits;if(out->complete)return out;}
        else {
            size_t pairs=size_t(data.F)*(data.F-1)/2;
            size_t used=sizeof(Geometry)+r->key.size()+128+sizeof(Pair)*(pairs+data.F)+data.F;
            if(used>limit-bytes){++limit_events;return {};}
            out=std::make_shared<Geometry>();out->marginal.resize(data.F);out->joint.resize(pairs);out->done.resize(data.F);
            cache.emplace(r->key,out);bytes+=used;
        }
        double start=now();++resumes;
        // A completed row is immutable. Partial construction can resume after
        // the caller's slice expires; no certificate uses partial geometry.
        #pragma omp parallel for schedule(dynamic,1) num_threads(threads) if(data.F>=32)
        for(int f=0;f<data.F;++f){
            if(out->done[f]||now()>=end)continue;
            Pair marginal;
            for(int w=0;w<data.W;++w){U bits=r->bits[w]&data.zero[size_t(f)*data.W+w];marginal.p+=int(__popcnt64(bits&data.pos[w]));marginal.n+=int(__popcnt64(bits&~data.pos[w]));}
            out->marginal[f]=marginal;bool complete=true;
            for(int g=f+1;g<data.F;++g){
                if((g&31)==0&&now()>=end){complete=false;break;}
                Pair joint;
                for(int w=0;w<data.W;++w){U bits=r->bits[w]&data.zero[size_t(f)*data.W+w]&data.zero[size_t(g)*data.W+w];joint.p+=int(__popcnt64(bits&data.pos[w]));joint.n+=int(__popcnt64(bits&~data.pos[w]));}
                out->joint[index(f,g)]=joint;
            }
            out->done[f]=complete;
        }
        out->complete=std::all_of(out->done.begin(),out->done.end(),[](unsigned char done){return done!=0;});
        geometry_seconds+=now()-start;return out;
    }
    // Exact K=2 attainable envelope for the generic coordinator, sharing the
    // same geometry as pricing. Install nothing when construction is partial.
    bool two_split_frontier(const Rows&r,int threads,double end,std::vector<int>&frontier){
        auto geometry=get(r,threads,end);if(!geometry||!geometry->complete||now()>=end)return false;
        std::vector<std::vector<int>> local(threads,std::vector<int>(r->p+1,-1));
        std::vector<unsigned char> done(data.F);int minimum=std::max(1,data.min_leaf);
        #pragma omp parallel for schedule(static) num_threads(threads) if(data.F>=32)
        for(int f=0;f<data.F;++f){
            if(now()>=end)continue;
            auto&out=local[omp_get_thread_num()];auto total=geometry->marginal[f];
            int p[2]={total.p,r->p-total.p},n[2]={total.n,r->n-total.n};
            if(std::min(p[0]+n[0],p[1]+n[1])<minimum){done[f]=1;continue;}
            for(int g=0;g<data.F;++g){if(g==f)continue;
                auto joint=geometry->joint[index(f,g)],marginal=geometry->marginal[g];
                int p0[2]={joint.p,marginal.p-joint.p},n0[2]={joint.n,marginal.n-joint.n};
                for(int side=0;side<2;++side){int p1=p[side]-p0[side],n1=n[side]-n0[side];
                    if(std::min(p0[side]+n0[side],p1+n1)<minimum)continue;
                    for(int bits=0;bits<8;++bits){int l=bits&1,rr=(bits>>1)&1,z=(bits>>2)&1;
                        int t=l*p0[side]+rr*p1+z*p[1-side],u=(1-l)*n0[side]+(1-rr)*n1+(1-z)*n[1-side];
                        out[t]=std::max(out[t],u);
                    }
                }
            }done[f]=1;
        }
        if(!std::all_of(done.begin(),done.end(),[](unsigned char done){return done!=0;}))return false;
        frontier.assign(r->p+1,-1);for(auto&out:local)for(int t=0;t<=r->p;++t)frontier[t]=std::max(frontier[t],out[t]);
        return true;
    }
    struct Answer {double reward=0;Tree tree;};
    bool solve(const Rows&r,double a,double b,double gamma,int threads,double end,std::array<Answer,4>&result){
        auto geometry=get(r,threads,end);if(!geometry||!geometry->complete||now()>=end)return false;
        double start=now();Answer leaf;leaf.reward=std::max(a*r->p,b*r->n);leaf.tree.a[1]=a*r->p>b*r->n?-2:-1;
        result.fill(leaf);std::vector<std::array<Answer,4>> roots(data.F);std::vector<unsigned char> done(data.F);
        auto label=[&](int p,int n){return a*p>b*n?-2:-1;};
        auto reward=[&](int p,int n){return std::max(a*p,b*n);};
        int minimum=std::max(1,data.min_leaf);
        #pragma omp parallel for schedule(static) num_threads(threads) if(data.F>=32)
        for(int f=0;f<data.F;++f){
            if(now()>=end)continue;
            auto&out=roots[f];out.fill(leaf);auto total=geometry->marginal[f];int p[2]={total.p,r->p-total.p},n[2]={total.n,r->n-total.n};
            if(std::min(p[0]+n[0],p[1]+n[1])<minimum){done[f]=1;continue;}
            Answer child[2][2];
            for(int side=0;side<2;++side){child[side][0].reward=reward(p[side],n[side]);child[side][0].tree.a[1]=label(p[side],n[side]);child[side][1]=child[side][0];}
            for(int g=0;g<data.F;++g){if(g==f)continue;
                auto joint=geometry->joint[index(f,g)],marginal=geometry->marginal[g];
                int p0[2]={joint.p,marginal.p-joint.p},n0[2]={joint.n,marginal.n-joint.n};
                for(int side=0;side<2;++side){int p1=p[side]-p0[side],n1=n[side]-n0[side];
                    if(std::min(p0[side]+n0[side],p1+n1)<minimum)continue;
                    double value=reward(p0[side],n0[side])+reward(p1,n1)-gamma;
                    if(value>child[side][1].reward){auto&best=child[side][1];best.reward=value;best.tree=Tree();best.tree.a[1]=g;best.tree.a[2]=label(p0[side],n0[side]);best.tree.a[3]=label(p1,n1);}
                }
            }
            for(int k=1;k<=3;++k)for(int left=0;left<=1;++left)for(int right=0;right<=1;++right){
                if(1+left+right>k)continue;
                double value=child[0][left].reward+child[1][right].reward-gamma;
                if(value>out[k].reward){out[k].reward=value;out[k].tree=Tree();out[k].tree.a[1]=f;transplant(out[k].tree,2,child[0][left].tree);transplant(out[k].tree,3,child[1][right].tree);}
            }
            done[f]=1;
        }
        bool complete=std::all_of(done.begin(),done.end(),[](unsigned char done){return done!=0;});
        if(complete)for(auto&out:roots)for(int k=1;k<=3;++k)if(out[k].reward>result[k].reward)result[k]=out[k];
        {std::lock_guard<std::mutex> lock(mutex);++queries;query_seconds+=now()-start;}
        return complete;
    }
    std::string diagnostics()const {std::ostringstream out;out<<std::setprecision(17)
        <<",\"shallow_queries\":"<<queries<<",\"shallow_cache_hits\":"<<hits<<",\"shallow_cache_bytes\":"<<bytes
        <<",\"shallow_geometry_resumes\":"<<resumes<<",\"shallow_cache_limit_events\":"<<limit_events
        <<",\"shallow_geometry_seconds\":"<<geometry_seconds<<",\"shallow_query_seconds\":"<<query_seconds;return out.str();}
};
}
