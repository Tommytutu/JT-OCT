#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <intrin.h>
#include <omp.h>

using U = std::uint64_t;
static U mix(U x) { x ^= x>>30; x*=0xbf58476d1ce4e5b9ULL; x^=x>>27;
    x*=0x94d049bb133111ebULL; return x^(x>>31); }
struct Workspace {
    std::vector<U> rows, keys;
    std::vector<int> nz, actions;
    std::vector<double> values;
    std::vector<unsigned char> valid;
    U stats[16] = {};
    Workspace(int w,int slots):rows(w),keys(std::size_t(w)*slots),nz(w),
        actions(slots),values(slots),valid(slots,0) {}
};
struct Engine {
    int W, nt, slots;
    std::vector<Workspace> work;
    Engine(int w,int threads,int s):W(w),nt(threads>0?threads:omp_get_max_threads()),slots(s) {
        for(int i=0;i<nt;++i)work.emplace_back(w,s);
    }
};
extern "C" __declspec(dllexport) void* jt_mp_plus_create(int w,int threads,int slots) {
    try { return new Engine(w,threads,slots); } catch(...) { return nullptr; }
}
extern "C" __declspec(dllexport) void jt_mp_plus_free(void* raw) { delete static_cast<Engine*>(raw); }
extern "C" __declspec(dllexport) void jt_mp_plus_stats(void* raw,U* out) {
    auto& e=*static_cast<Engine*>(raw); std::fill(out,out+16,0);
    for(auto& w:e.work)for(int k=0;k<16;++k)out[k]+=w.stats[k];
}
// ip: F,W,D,root,q,no_repeat,min_leaf,start,count,stride,bound,sparse,prune,nconf
// dp: uniform observation weight, split penalty, incumbent cutoff
extern "C" __declspec(dllexport) int jt_mp_plus_costs(void* raw,const U* masks,
    const U* positive,const int* conflict,const double* incoming,const int* ip,
    const double* dp,double* output,std::int32_t* tail) {
    auto& e=*static_cast<Engine*>(raw);
    const int F=ip[0],W=ip[1],D=ip[2],root=ip[3],q=ip[4],repeat=ip[5],ml=ip[6];
    const int start=ip[7],count=ip[8],stride=ip[9],bound=ip[10],sparse=ip[11],prune=ip[12],nc=ip[13];
    const double weight=dp[0],penalty=dp[1],cutoff=dp[2],INF=1e300;
    if(W!=e.W)return 1;
#pragma omp parallel num_threads(e.nt)
    {
        auto& w=e.work[omp_get_thread_num()];
#pragma omp for schedule(static)
        for(int local=0;local<count;++local) {
            output[local]=INF;tail[local]=-2;
            const int index=start+local;
            const double previous=incoming[index/stride];
            if(previous>=INF/2){++w.stats[14];continue;}
            int fs[5]={root,0,0,0,0},code=index,stop=D-1;
            for(int j=D-2;j>=1;--j){fs[j]=code%(F+1);code/=F+1;}
            bool valid=true;
            for(int j=1;j<D-1;++j) {
                if(fs[j]==F){if(stop==D-1)stop=j;}
                else {
                    if(stop!=D-1)valid=false;
                    if(repeat)for(int k=0;k<j;++k)if(fs[k]==fs[j])valid=false;
                }
            }
            if(!valid){++w.stats[15];continue;}
            ++w.stats[0];
            double allocated=0;
            for(int j=0;j<stop;++j)allocated+=penalty/(1<<(D-1-j));
            if(prune && previous+allocated>cutoff+1e-12){++w.stats[11];continue;}
            int total=0,pos=0,nz=0;
            U hash=0;
            for(int word=0;word<W;++word) {
                U set=~U(0);
                for(int j=0;j<stop;++j)set&=masks[(((q>>(D-2-j))&1)*F+fs[j])*W+word];
                w.rows[word]=set;
                total+=int(__popcnt64(set));pos+=int(__popcnt64(set&positive[word]));
                if(set){w.nz[nz++]=word;if(e.slots)hash^=mix(set^(0x9e3779b97f4a7c15ULL*(word+1)));}
            }
            w.stats[12]+=nz;w.stats[13]+=W;
            const int error=std::min(pos,total-pos);
            if(total<ml)continue;
            if(stop!=D-1) {
                const double val=previous+allocated+weight*error/(1<<(D-1-stop));
                if(prune && val>cutoff+1e-12){++w.stats[11];continue;}
                output[local]=val;tail[local]=-3;continue;
            }
            ++w.stats[1];
            int unavoidable=0;
            if(bound>=2)for(int g=0;g<nc;++g) {
                const int row=conflict[2*g];
                if(w.rows[row/64]&(U(1)<<(row%64)))unavoidable+=conflict[2*g+1];
            }
            const double stop_loss=weight*error;
            const double lower=std::min(stop_loss,penalty+weight*unavoidable);
            if(prune && previous+allocated+lower>cutoff+1e-12){++w.stats[11];continue;}
            int reason=-1;
            if(!error)reason=2;
            else if(bound>=1 && stop_loss<=penalty)reason=3;
            else if(bound>=2 && stop_loss<=penalty+weight*unavoidable)reason=4;
            if(reason>=0){++w.stats[reason];output[local]=previous+allocated+stop_loss;tail[local]=-1;continue;}
            std::size_t slot=0,offset=0;
            if(e.slots) {
                ++w.stats[5];slot=hash%e.slots;offset=slot*W;
                if(w.valid[slot] && !std::memcmp(w.keys.data()+offset,w.rows.data(),W*sizeof(U))) {
                    ++w.stats[6];output[local]=previous+allocated+w.values[slot];tail[local]=w.actions[slot];continue;
                }
            }
            double best=stop_loss;int action=-1;
            const int words=sparse?nz:W;
            for(int h=0;h<F;++h) {
                bool used=false;
                if(repeat)for(int j=0;j<D-1;++j)if(fs[j]==h)used=true;
                if(used)continue;
                ++w.stats[9];w.stats[10]+=words;
                int n0=0,y0=0;
                for(int k=0;k<words;++k){const int word=sparse?w.nz[k]:k;
                    const U left=w.rows[word]&masks[h*W+word];
                    n0+=int(__popcnt64(left));y0+=int(__popcnt64(left&positive[word]));}
                const int n1=total-n0,y1=pos-y0;
                if(n0<ml||n1<ml)continue;
                const double candidate=penalty+weight*(std::min(y0,n0-y0)+std::min(y1,n1-y1));
                if(candidate<best){best=candidate;action=h;}
            }
            if(e.slots){std::memcpy(w.keys.data()+offset,w.rows.data(),W*sizeof(U));
                w.values[slot]=best;w.actions[slot]=action;w.valid[slot]=1;++w.stats[8];}
            output[local]=previous+allocated+best;tail[local]=action;
        }
    }
    return 0;
}
