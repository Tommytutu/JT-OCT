// Accuracy D2/D3 terminal service. The existing JT-CG master stays unchanged.
#include "general_jt/cm5_common.hpp"
#include "general_jt/cm_weighted_support.hpp"
#include "general_jt/cm_mcc_gpu.hpp"
#include "general_jt/cm_shallow_linear.hpp"
#include <future>
#define EXPORT extern "C" __declspec(dllexport)
namespace {
thread_local std::string error;
struct Oracle {
    cm5::Data data;
    cm5::MccGpu gpu;
    cm5::ShallowLinearStore shallow;
    int threads;
    std::future<void> preparation;
    bool prefetch_started=false;
    Oracle(const unsigned char*x,const unsigned char*y,int n,int f,int ml,int nt,size_t cache_bytes,const wchar_t*rtc)
        :data(x,y,n,f,ml),gpu(data,nt,2,rtc),shallow(data,64*1024*1024),threads(nt) {
        gpu.adaptive_scalar_blocks=true;
        gpu.symmetric_scalar_pairs=true;
        gpu.scalar_root_detail=false;
        gpu.scalar_explicit_sync=false;
        gpu.geometry_limit=cache_bytes;
    }
    ~Oracle(){gpu.cancel_prefill=true;if(preparation.valid())preparation.wait();}
    void check_preparation(){if(preparation.valid()&&preparation.wait_for(std::chrono::seconds(0))==std::future_status::ready){preparation.get();gpu.async_geometry=false;}}
};
}
EXPORT void* accuracy_oracle_create(const unsigned char*x,const unsigned char*y,int n,int f,int ml,int nt,size_t cache_bytes,const wchar_t*rtc) {
    try {error.clear();if(!x||!y||!rtc||n<1||f<1||ml<0||nt<1||nt>64)throw std::invalid_argument("invalid oracle dimensions");return new Oracle(x,y,n,f,ml,nt,cache_bytes,rtc);}
    catch(const std::exception&e){error=e.what();return nullptr;}
}
EXPORT const char* accuracy_oracle_error(){return error.c_str();}
EXPORT void accuracy_oracle_free(void*p){delete static_cast<Oracle*>(p);}
EXPORT int accuracy_oracle_prefetch(void*ptr,const cm5::U*masks,int count,double seconds) {
    try {error.clear();if(!ptr||count<0||!std::isfinite(seconds)||seconds<0)throw std::invalid_argument("invalid prefetch request");
        auto&o=*static_cast<Oracle*>(ptr);if(o.preparation.valid())o.preparation.get();
        std::vector<cm5::Rows> domains;domains.reserve(count);
        for(int i=0;i<count;++i)domains.push_back(o.data.mask(std::vector<cm5::U>(masks+size_t(i)*o.data.W,masks+size_t(i+1)*o.data.W)));
        o.prefetch_started=true;o.gpu.cancel_prefill=false;o.gpu.async_geometry=true;o.gpu.prefill_done=false;double end=cm5::now()+seconds;
        o.preparation=std::async(std::launch::async,[&o,domains=std::move(domains),end](){o.gpu.prefill_geometry(domains,end);});return 1;
    }catch(const std::exception&e){error=e.what();return 0;}
}
// Per-request status: 1 exact, 0 deadline, -1 infeasible. Values in error-count units.
EXPORT int accuracy_oracle_batch(void*ptr,const cm5::U*masks,int count,int depth,double gamma,double seconds,
                                int*status,double*values,int*trees,double*timings) {
    try {
        error.clear();if(!ptr||count<0||(depth!=2&&depth!=3)||gamma<0||!std::isfinite(gamma)||!std::isfinite(seconds)||seconds<0)throw std::invalid_argument("invalid oracle request");
        auto&o=*static_cast<Oracle*>(ptr);auto&g=o.gpu;
        o.check_preparation();
        double begin=cm5::now(),end=begin+seconds;
        double geo=g.geometry_seconds,kernel=g.kernel_seconds,transfer=g.transfer_seconds,join=g.join_seconds;
        std::vector<cm5::Rows> domains;domains.reserve(count);
        for(int i=0;i<count;++i){std::vector<cm5::U> v(masks+size_t(i)*o.data.W,masks+size_t(i+1)*o.data.W);domains.push_back(o.data.mask(v));status[i]=0;}
        std::future<void> preparation;
        bool persistent=o.prefetch_started;
        if(depth==3&&count>1&&!persistent){g.async_geometry=true;g.prefill_done=false;
            preparation=std::async(std::launch::async,[&](){g.prefill_geometry(domains,end);});}
        try {
            for(int i=0;i<count;++i){auto&r=domains[i];cm5::Tree tree;tree.a[1]=r->p<=r->n?-1:-2;double value=std::min(r->p,r->n);
                if(r->p+r->n<o.data.min_leaf){status[i]=-1;value=cm5::INF;}
                else if(value<=gamma){status[i]=1;}
                else if(depth==3){status[i]=g.scalar(r,1.,1.,gamma,tree,value,end)?1:0;}
                else {std::array<cm5::ShallowLinearStore::Answer,4> a;
                    if(o.shallow.solve(r,1.,1.,gamma,o.threads,end,a)){tree=a[3].tree;value=r->p+r->n-a[3].reward;status[i]=1;}}
                if(status[i]>=0){auto c=o.data.counts(tree,r,1,depth);double actual=r->p-c.tp+r->n-c.tn+gamma*c.k;
                    // Interrupted kernels do not certify partially evaluated costs.
                    if(status[i]==1&&std::abs(actual-value)>1e-7)throw std::runtime_error("accuracy oracle witness mismatch");value=actual;}
                values[i]=value;std::copy(tree.a.begin(),tree.a.end(),trees+size_t(i)*cm5::SLOTS);
            }
            if(preparation.valid())preparation.get();if(!persistent)g.async_geometry=false;
            o.check_preparation();
        } catch(...) {if(preparation.valid())preparation.wait();if(!persistent)g.async_geometry=false;throw;}
        timings[0]=cm5::now()-begin;timings[1]=g.geometry_seconds-geo;timings[2]=g.kernel_seconds-kernel;
        timings[3]=g.transfer_seconds-transfer;timings[4]=g.join_seconds-join;
        return 1;
    } catch(const std::exception&e){error=e.what();return 0;}
}
