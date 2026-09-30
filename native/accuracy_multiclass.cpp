#include "general_jt/cm5_common.hpp"
#include "general_jt/cm_weighted_support.hpp"
#include "general_jt/cm_mcc_gpu.hpp"
#include "accuracy_multiclass.hpp"
#define EXPORT extern "C" __declspec(dllexport)
namespace {thread_local std::string error;}
EXPORT const char* accuracy_oracle_error(){return error.c_str();}
EXPORT void* accuracy_oracle_create(const unsigned char*x,const unsigned char*y,int n,int f,int ml,int nt,size_t cache,const wchar_t*rtc,int K){
    try{error.clear();return new accuracy::MultiOracle(x,y,n,f,K,ml,nt,rtc);}catch(const std::exception&e){error=e.what();return nullptr;}}
EXPORT void accuracy_oracle_free(void*p){delete static_cast<accuracy::MultiOracle*>(p);}
EXPORT void accuracy_oracle_counters(void*p,uint64_t*out){const auto&o=*static_cast<accuracy::MultiOracle*>(p);std::copy(o.diagnostics.begin(),o.diagnostics.end(),out);}
EXPORT int accuracy_oracle_prefetch(void*,const cm5::U*,int,double){return 1;}
EXPORT int accuracy_oracle_batch(void*ptr,const cm5::U*masks,int count,int depth,double gamma,double seconds,int*status,double*values,int*trees,double*timings){
    try{error.clear();if(!ptr||count<0||(depth!=2&&depth!=3)||gamma<0||!std::isfinite(gamma)||!std::isfinite(seconds)||seconds<0)throw std::invalid_argument("invalid accuracy request");
        auto&o=*static_cast<accuracy::MultiOracle*>(ptr);double start=cm5::now(),end=start+seconds,geo=o.prepare_seconds,gpu=o.gpu_seconds;
        for(int i=0;i<count;++i){auto r=masks+size_t(i)*o.data.W;cm5::Tree tree;std::vector<int>c(o.K);int N=0;
            for(int k=0;k<o.K;++k)for(int w=0;w<o.data.W;++w)c[k]+=int(__popcnt64(r[w]&o.labels[size_t(k)*o.data.W+w]));
            N=std::accumulate(c.begin(),c.end(),0);int label=int(std::max_element(c.begin(),c.end())-c.begin());tree.a[1]=-label-1;double value=N-c[label];
            if(N<o.data.min_leaf){status[i]=-1;value=cm5::INF;}else status[i]=o.solve(r,depth,gamma,end,tree,value)?1:0;
            values[i]=value;std::copy(tree.a.begin(),tree.a.end(),trees+size_t(i)*64);
        }
        timings[0]=cm5::now()-start;timings[1]=o.prepare_seconds-geo;timings[2]=o.gpu_seconds-gpu;timings[3]=timings[4]=0;return 1;
    }catch(const std::exception&e){error=e.what();return 0;}}
