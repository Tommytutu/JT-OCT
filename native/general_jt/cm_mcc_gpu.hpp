#pragma once
// Native CUDA driver/NVRTC owner. Python never launches or joins a kernel.
#include <cuda.h>
#include <nvrtc.h>
#include <mutex>
#include <condition_variable>
#include <memory_resource>
namespace cm5 {
// Large geometry bitmaps live until their owner finishes. Reserve address
// space once, commit in bounded chunks, and release pages once after joining
// all readers. Allocations beyond the arena limit use the ordinary allocator.
struct GeometryArena final:std::pmr::memory_resource {
    std::mutex mutex;unsigned char*base=nullptr;size_t limit=0,committed=0,used=0;
    void configure(size_t bytes){std::lock_guard<std::mutex> guard(mutex);if(!base)limit=(bytes+4095)&~size_t(4095);}
    void*do_allocate(size_t bytes,size_t alignment)override{
        std::lock_guard<std::mutex> guard(mutex);
        if(!base&&limit)base=static_cast<unsigned char*>(VirtualAlloc(nullptr,limit,MEM_RESERVE,PAGE_READWRITE));
        size_t offset=(used+alignment-1)&~(alignment-1);
        if(base&&offset<=limit&&bytes<=limit-offset){size_t stop=offset+bytes;
            if(stop>committed){size_t chunk=16*1024*1024,next=std::min(limit,((stop+chunk-1)/chunk)*chunk);
                if(!VirtualAlloc(base+committed,next-committed,MEM_COMMIT,PAGE_READWRITE))return std::pmr::new_delete_resource()->allocate(bytes,alignment);committed=next;}
            used=stop;return base+offset;}
        return std::pmr::new_delete_resource()->allocate(bytes,alignment);
    }
    void do_deallocate(void*p,size_t bytes,size_t alignment)override{
        std::lock_guard<std::mutex> guard(mutex);auto address=reinterpret_cast<uintptr_t>(p),first=reinterpret_cast<uintptr_t>(base);
        if(!base||address<first||address-first>=limit)std::pmr::new_delete_resource()->deallocate(p,bytes,alignment);
    }
    bool do_is_equal(const std::pmr::memory_resource&other)const noexcept override{return this==&other;}
    void reset(){if(base)VirtualFree(base,0,MEM_RELEASE);base=nullptr;used=committed=0;}
    ~GeometryArena(){reset();}
};
struct MccGpu {
    CUcontext context=nullptr; CUmodule module=nullptr; CUfunction kernel=nullptr,triple_kernel=nullptr,side_join_kernel=nullptr,root_join_kernel=nullptr,kernels[2][3]{};
    CUdeviceptr zero=0,pos=0,rows=0,costs=0,actions=0,histogram=0,feature_ids=0,scalar_zero=0,scalar_pos=0,scalar_dominated=0,extra_weights=0,join_p0=0,join_n0=0,join_sides=0,join_choices=0,join_costs=0,join_trees=0,join_roots=0;
    HMODULE rtc=nullptr,builtins=nullptr; const Data& data; int threads,shared_limit=49152; bool enabled=false;
    U calls=0; double seconds=0,setup_seconds=0; std::string failure;bool adaptive_scalar_blocks=false,symmetric_scalar_pairs=false;
    std::vector<double> values; std::vector<int> tails,hist; std::mutex lock,geometry_lock;
    std::atomic<bool> async_geometry{false},prefill_done{true},cancel_prefill{false};std::condition_variable geometry_ready;double prefill_seconds=0;U integer_scalar_calls=0;
    std::map<std::tuple<std::string,double,double>,std::array<WeightedBound,8>> cache;
    struct Pending {int next=0;std::vector<double> values;std::vector<int> tails,hist;explicit Pending(size_t n):values(n),tails(n),hist(2*n){}};
    std::map<std::tuple<std::string,double,double>,std::unique_ptr<Pending>> pending;
    U cache_hits=0,continuations=0;
    GeometryArena geometry_arena;
    struct ScalarGeometry {std::vector<int> ids,p0,n0,alias;std::pmr::vector<U> zero;std::vector<U> pos,mask;
        explicit ScalarGeometry(std::pmr::memory_resource*resource=std::pmr::new_delete_resource()):zero(resource){}};
    std::vector<double> scalar_root_costs;
    std::array<WeightedBound,8> packed_budgets;
    std::vector<std::array<double,2>> budget_directions;size_t budget_cache_bytes=0;U batched_profiles=0;
    bool device_reduction=true;U device_joins=0;
    // Accuracy pricing consumes only the minimum and its witness.
    bool scalar_root_detail=true, scalar_explicit_sync=true;
    double geometry_seconds=0,kernel_seconds=0,transfer_seconds=0,join_seconds=0;
    std::map<std::string,ScalarGeometry> scalar_geometry;
    U scalar_calls=0,scalar_completed=0,scalar_leaf_prunes=0,geometry_hits=0,scalar_pairs=0;size_t geometry_bytes=0,geometry_limit=32*1024*1024;
    static void check(CUresult r){if(r!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(r,&s);throw std::runtime_error(s?s:"CUDA driver error");}}
    template<class T>T proc(const char* name){auto p=GetProcAddress(rtc,name);if(!p)throw std::runtime_error("NVRTC symbol missing");return reinterpret_cast<T>(p);}
    MccGpu(const Data& d,int nt,int mode,const wchar_t* path):data(d),threads(nt){
        if(!mode)return; double start=now();
        try {
            check(cuInit(0));CUdevice device;check(cuDeviceGet(&device,0));
            check(cuDevicePrimaryCtxRetain(&context,device));check(cuCtxSetCurrent(context));
            rtc=LoadLibraryExW(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if(!rtc)throw std::runtime_error("Cannot load NVRTC; provide its installed DLL path");
            std::wstring directory(path);auto slash=directory.find_last_of(L"/\\");directory=directory.substr(0,slash+1);
            WIN32_FIND_DATAW found{};HANDLE search=FindFirstFileW((directory+L"nvrtc-builtins64_*.dll").c_str(),&found);
            if(search!=INVALID_HANDLE_VALUE){builtins=LoadLibraryExW((directory+found.cFileName).c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);FindClose(search);}
            if(!builtins)throw std::runtime_error("Cannot load the NVRTC builtins DLL next to NVRTC");
            auto create=proc<decltype(&nvrtcCreateProgram)>("nvrtcCreateProgram");
            auto compile=proc<decltype(&nvrtcCompileProgram)>("nvrtcCompileProgram");
            auto size=proc<decltype(&nvrtcGetPTXSize)>("nvrtcGetPTXSize");
            auto get=proc<decltype(&nvrtcGetPTX)>("nvrtcGetPTX");
            auto destroy=proc<decltype(&nvrtcDestroyProgram)>("nvrtcDestroyProgram");
            const char* source=R"CUDA(
typedef unsigned long long U;
template<int D,int I> __device__ __forceinline__ void mcc_work(const U* zero,const U* positive,const U* mask,
 int F,int W,int ml,double a,double b,int start,int count,double* out,int* tail,int* hist,const int* features,double gamma,const unsigned char* dominated,int symmetric,const double* extra,int directions_unused,int integer_unused,int compact){
 const int directions=D,integer_weights=I;
 int item=start+blockIdx.x, pair=item/4,side=item%4, fi=pair/F,gi=pair%F,tid=threadIdx.x;
 if(blockIdx.x>=count)return;
 if(compact){gi=(int)((1+sqrtf(1+8.0f*pair))*.5f);while(gi*(gi-1)/2>pair)--gi;while(gi*(gi+1)/2<=pair)++gi;fi=pair-gi*(gi-1)/2;item=4*(fi*F+gi)+side;}
 int f=fi,g=gi;
 if(symmetric && f>g)return;
 int mirror=(g*F+f)*4+2*(side%2)+side/2;
 int stride=4*F*F;
 if(!tid){for(int d=0;d<directions;++d){out[item+d*stride]=1e300;tail[item+d*stride]=-3;if(symmetric){out[mirror+d*stride]=1e300;tail[mirror+d*stride]=-3;}}hist[2*item]=hist[2*item+1]=0;if(symmetric){hist[2*mirror]=hist[2*mirror+1]=0;}}
 if(f==g || (dominated && dominated[2*f+side/2] && (!symmetric || dominated[2*g+side%2])))return;
 extern __shared__ U bits[];int* nzid=(int*)(bits+W);
 double* vs=(double*)((char*)bits+((W*12+7)&~7));int* ids=(int*)(vs+D*blockDim.x);int* iv=(int*)vs;
 __shared__ int ps[256],ns[256],nz;
 if(!tid)nz=0;__syncthreads();int p=0,n=0;
 for(int w=tid;w<W;w+=blockDim.x){U zf=zero[features?w*F+f:f*W+w],zg=zero[features?w*F+g:g*W+w];U x=mask[w] & ((side/2)?~zf:zf) & ((side%2)?~zg:zg);
 bits[w]=x;p+=__popcll(x&positive[w]);n+=__popcll(x&~positive[w]);if(!features && x)nzid[atomicAdd(&nz,1)]=w;}
 ps[tid]=p;ns[tid]=n;__syncthreads();
 for(int s=blockDim.x/2;s;s>>=1){if(tid<s){ps[tid]+=ps[tid+s];ns[tid]+=ns[tid+s];}__syncthreads();}
 p=ps[0];n=ns[0];if(!tid){hist[2*item]=p;hist[2*item+1]=n;if(symmetric){hist[2*mirror]=p;hist[2*mirror+1]=n;}}if(p+n<ml)return;
 double aa[3]={a,0,0},bb[3]={b,0,0},best[3];int winner[3],ai[3],bi[3],ibest[3];
 for(int d=0;d<directions;++d){if(d){aa[d]=extra[2*(d-1)];bb[d]=extra[2*(d-1)+1];}best[d]=fmin(aa[d]*p,bb[d]*n);winner[d]=(aa[d]*p<=bb[d]*n)?-1:-2;
 if(integer_weights){ai[d]=(int)(aa[d]*1024);bi[d]=(int)(bb[d]*1024);ibest[d]=min(ai[d]*p,bi[d]*n);}}
 // A stump pays gamma before correcting any errors. This proof applies
 // simultaneously to every batched direction, including pure routed leaves.
 bool stop=true;for(int d=0;d<directions;++d)stop=stop&&(best[d]<=gamma);
 if(stop){if(!tid)for(int d=0;d<directions;++d){out[item+d*stride]=best[d];tail[item+d*stride]=winner[d];if(symmetric){out[mirror+d*stride]=best[d];tail[mirror+d*stride]=winner[d];}}return;}
 for(int h=tid;h<F;h+=blockDim.x){if(h==f||h==g)continue;int p0=0,n0=0;
 // Packed scalar rows use sequential words; the original global-mask path
 // retains sparse word indices. Avoid atomics/indirection on compact data.
 for(int j=0;j<(features?W:nz);++j){int w=features?j:nzid[j];U x=bits[w]&zero[features?w*F+h:h*W+w];p0+=__popcll(x&positive[w]);n0+=__popcll(x&~positive[w]);}
 if(p0+n0<ml||p+n-p0-n0<ml)continue;
 for(int d=0;d<directions;++d){
 if(integer_weights){int a=ai[d],b=bi[d],c=min(a*p0,b*n0)+min(a*(p-p0),b*(n-n0));if(c<ibest[d]){ibest[d]=c;winner[d]=4*features[h]+(a*p0>b*n0)+2*(a*(p-p0)>b*(n-n0));}}
 else{double a=aa[d],b=bb[d];double c=gamma+fmin(a*p0,b*n0)+fmin(a*(p-p0),b*(n-n0));if(c<best[d]){best[d]=c;
 // Packed witnesses retain the two optimal leaf labels. Reconstructing them
 // later must not allocate global-length row masks for every budget layer.
 winner[d]=features?4*features[h]+(a*p0>b*n0)+2*(a*(p-p0)>b*(n-n0)):h;}}}}
 for(int d=0;d<directions;++d){int z=d*blockDim.x+tid;if(integer_weights)iv[z]=ibest[d];else vs[z]=best[d];ids[z]=winner[d];}__syncthreads();
 for(int s=blockDim.x/2;s;s>>=1){if(tid<s)for(int d=0;d<directions;++d){int z=d*blockDim.x+tid;
 if(integer_weights){if(iv[z+s]<iv[z] || (iv[z+s]==iv[z] && ids[z+s]<ids[z])){iv[z]=iv[z+s];ids[z]=ids[z+s];}}
 else if(vs[z+s]<vs[z] || (vs[z+s]==vs[z] && ids[z+s]<ids[z])){vs[z]=vs[z+s];ids[z]=ids[z+s];}}__syncthreads();}
 if(!tid)for(int d=0;d<directions;++d){int z=d*blockDim.x;double value=integer_weights?double(iv[z])/1024:vs[z];int action=ids[z];
 if(integer_weights&&action>=0){value+=gamma;double leaf=fmin(aa[d]*p,bb[d]*n);if(leaf<=value){value=leaf;action=aa[d]*p<=bb[d]*n?-1:-2;}}
 out[item+d*stride]=value;tail[item+d*stride]=action;if(symmetric){out[mirror+d*stride]=value;tail[mirror+d*stride]=action;}}
}
#define ENTRY(NAME,D,I) extern "C" __global__ void NAME(const U* zero,const U* positive,const U* mask,int F,int W,int ml,double a,double b,int start,int count,double* out,int* tail,int* hist,const int* features,double gamma,const unsigned char* dominated,int symmetric,const double* extra,int directions,int integer_weights,int compact){mcc_work<D,I>(zero,positive,mask,F,W,ml,a,b,start,count,out,tail,hist,features,gamma,dominated,symmetric,extra,directions,integer_weights,compact);}
ENTRY(mcc_d3,1,0)
ENTRY(mcc_d3_2,2,0)
ENTRY(mcc_d3_3,3,0)
ENTRY(mcc_d3_i,1,1)
ENTRY(mcc_d3_i2,2,1)
ENTRY(mcc_d3_i3,3,1)
// Reduce all second-level choices on the device; retain concrete actions.
// side_choice encodes 4*j+2*left_budget+right_budget, or -1 for a leaf.
extern "C" __global__ void cm_side_join(int F,int P,int N,const int* p0,const int* n0,
 const double* values,const int* hist,double a,double b,const double* extra,
 double gamma,int budgets,double* side_cost,int* side_choice){
 int d=blockIdx.y, fs=blockIdx.x,f=fs/2,side=fs%2,tid=threadIdx.x;
 if(d){a=extra[2*(d-1)];b=extra[2*(d-1)+1];}
 int p=side?P-p0[f]:p0[f],n=side?N-n0[f]:n0[f],stride=4*F*F;
 double best[4];int choice[4];for(int k=0;k<4;++k){best[k]=fmin(a*p,b*n);choice[k]=-1;}
 for(int j=tid;j<F;j+=blockDim.x){if(j==f)continue;int z=(f*F+j)*4+2*side;
  if(values[d*stride+z]>1e299||values[d*stride+z+1]>1e299)continue;
  if(!budgets){double v=gamma+values[d*stride+z]+values[d*stride+z+1];int c=4*j+3;
   if(v<best[3]||(v==best[3]&&c<choice[3])){best[3]=v;choice[3]=c;}continue;}
  for(int l=0;l<2;++l)for(int r=0;r<2;++r){
   double v=(l?values[d*stride+z]:fmin(a*hist[2*z],b*hist[2*z+1]))+
            (r?values[d*stride+z+1]:fmin(a*hist[2*z+2],b*hist[2*z+3]));
   int c=4*j+2*l+r;for(int k=1+l+r;k<4;++k)
    if(v<best[k]||(v==best[k]&&c<choice[k])){best[k]=v;choice[k]=c;}
  }
 }
 __shared__ double v[4*64];__shared__ int c[4*64];
 for(int k=0;k<4;++k){v[64*k+tid]=best[k];c[64*k+tid]=choice[k];}__syncthreads();
 for(int s=32;s;s>>=1){if(tid<s)for(int k=0;k<4;++k){int z=64*k+tid;
   if(v[z+s]<v[z]||(v[z+s]==v[z]&&c[z+s]<c[z])){v[z]=v[z+s];c[z]=c[z+s];}}
  __syncthreads();}
 if(!tid)for(int k=0;k<4;++k){int z=(d*2*F+fs)*4+k;side_cost[z]=v[64*k];side_choice[z]=c[64*k];}
}
extern "C" __global__ void cm_root_join(int F,int P,int N,const int* p0,const int* n0,
 const int* features,const int* hist,const int* tail,const double* side_cost,const int* side_choice,
 double a,double b,const double* extra,double gamma,int budgets,
 double* output,int* trees,double* root_cost){
 int d=blockIdx.y,k=blockIdx.x,tid=threadIdx.x,stride=4*F*F;
 if(d){a=extra[2*(d-1)];b=extra[2*(d-1)+1];}
 double best=fmin(a*P,b*N);int choice=-1;
 for(int f=tid;f<F;f+=blockDim.x){int z=(d*2*F+2*f)*4;
  if(!budgets){double v=gamma+side_cost[z+3]+side_cost[z+7];root_cost[f]=v;
   if(v<best||(v==best&&f*16+15<choice)){best=v;choice=f*16+15;}}
  else for(int l=0;l<4;++l)for(int r=0;r<4&&1+l+r<=k;++r){
   double v=side_cost[z+l]+side_cost[z+4+r];int c=f*16+l*4+r;
   if(v<best||(v==best&&c<choice)){best=v;choice=c;}}
 }
 __shared__ double v[256];__shared__ int c[256];v[tid]=best;c[tid]=choice;__syncthreads();
 for(int s=128;s;s>>=1){if(tid<s&&(v[tid+s]<v[tid]||(v[tid+s]==v[tid]&&c[tid+s]<c[tid]))){v[tid]=v[tid+s];c[tid]=c[tid+s];}__syncthreads();}
 if(tid)return;int out=d*8+k;output[out]=v[0];int* tree=trees+out*16;for(int i=0;i<16;++i)tree[i]=-99;
 tree[1]=a*P<=b*N?-1:-2;if(c[0]<0)return;
 int f=c[0]/16,lk=(c[0]%16)/4,rk=c[0]%4;tree[1]=features[f];
 for(int side=0;side<2;++side){int node=2+side,p=side?P-p0[f]:p0[f],n=side?N-n0[f]:n0[f];
  tree[node]=a*p<=b*n?-1:-2;int action=side_choice[(d*2*F+2*f+side)*4+(side?rk:lk)];if(action<0)continue;
  int j=action/4,l=(action%4)/2,r=action%2;tree[node]=features[j];
  for(int z=0;z<2;++z){int index=(f*F+j)*4+2*side+z,slot=2*node+z;
   int act=(z?r:l)?tail[d*stride+index]:(a*hist[2*index]<=b*hist[2*index+1]?-1:-2);
   if(act<0)tree[slot]=act;else{tree[slot]=act/4;tree[slot*2]=-1-(act&1);tree[slot*2+1]=-1-((act>>1)&1);}
  }
 }
}

// One warp per binary pattern; integer statistics are independent of metric.
extern "C" __global__ void cm_triples(const U* zero,const U* positive,const U* mask,
 int W,const int* triples,int start,int count,int* output){
 int index=start+blockIdx.x;if(index>=start+count)return;
 int pattern=threadIdx.x/32,lane=threadIdx.x%32,p=0,n=0;
 int f=triples[3*index],g=triples[3*index+1],h=triples[3*index+2];
 for(int w=lane;w<W;w+=32){U a=zero[f*W+w],b=zero[g*W+w],c=zero[h*W+w];
 U x=mask[w]&((pattern&1)?~a:a)&((pattern&2)?~b:b)&((pattern&4)?~c:c);
 p+=__popcll(x&positive[w]);n+=__popcll(x&~positive[w]);}
 for(int delta=16;delta;delta/=2){p+=__shfl_down_sync(0xffffffff,p,delta);n+=__shfl_down_sync(0xffffffff,n,delta);}
 if(!lane){output[index*16+2*pattern]=p;output[index*16+2*pattern+1]=n;}
})CUDA";
            nvrtcProgram program=nullptr;
            if(create(&program,source,"mcc_native.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("NVRTC create failed");
            int major=0,minor=0;check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
            std::string arch="--gpu-architecture=compute_"+std::to_string(major)+std::to_string(minor);
            const char* options[]={"--std=c++11",arch.c_str(),"--fmad=false"};
            if(compile(program,3,options)!=NVRTC_SUCCESS){
                size_t n=0;proc<decltype(&nvrtcGetProgramLogSize)>("nvrtcGetProgramLogSize")(program,&n);std::string log(n,' ');
                proc<decltype(&nvrtcGetProgramLog)>("nvrtcGetProgramLog")(program,log.data());destroy(&program);throw std::runtime_error(log);}
            size_t n=0;size(program,&n);std::vector<char> ptx(n);get(program,ptx.data());destroy(&program);
            check(cuModuleLoadData(&module,ptx.data()));const char*names[2][3]={{"mcc_d3","mcc_d3_2","mcc_d3_3"},{"mcc_d3_i","mcc_d3_i2","mcc_d3_i3"}};
            for(int i=0;i<2;++i)for(int j=0;j<3;++j)check(cuModuleGetFunction(&kernels[i][j],module,names[i][j]));kernel=kernels[0][0];check(cuModuleGetFunction(&triple_kernel,module,"cm_triples"));
            check(cuModuleGetFunction(&side_join_kernel,module,"cm_side_join"));check(cuModuleGetFunction(&root_join_kernel,module,"cm_root_join"));
            check(cuMemAlloc(&zero,d.zero.size()*sizeof(U)));check(cuMemAlloc(&pos,d.pos.size()*sizeof(U)));check(cuMemAlloc(&rows,d.W*sizeof(U)));
            check(cuMemcpyHtoD(zero,d.zero.data(),d.zero.size()*sizeof(U)));check(cuMemcpyHtoD(pos,d.pos.data(),d.pos.size()*sizeof(U)));
            check(cuDeviceGetAttribute(&shared_limit,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,device));
            if(size_t(d.W)*12+256*20+8>size_t(shared_limit))throw std::runtime_error("row mask exceeds GPU shared-memory capacity; use CPU mode");
            size_t items=size_t(d.F)*d.F*4;if(items>size_t(INT_MAX))throw std::runtime_error("too many GPU feature pairs");values.resize(items);tails.resize(items);hist.resize(2*items);
            check(cuMemAlloc(&costs,3*items*sizeof(double)));check(cuMemAlloc(&actions,3*items*sizeof(int)));check(cuMemAlloc(&histogram,2*items*sizeof(int)));check(cuMemAlloc(&feature_ids,d.F*sizeof(int)));check(cuMemAlloc(&extra_weights,4*sizeof(double)));
            check(cuMemAlloc(&scalar_zero,d.zero.size()*sizeof(U)));check(cuMemAlloc(&scalar_pos,d.pos.size()*sizeof(U)));check(cuMemAlloc(&scalar_dominated,2*d.F));
            check(cuMemAlloc(&join_p0,d.F*sizeof(int)));check(cuMemAlloc(&join_n0,d.F*sizeof(int)));
            check(cuMemAlloc(&join_sides,3*2*d.F*4*sizeof(double)));check(cuMemAlloc(&join_choices,3*2*d.F*4*sizeof(int)));
            check(cuMemAlloc(&join_costs,3*8*sizeof(double)));check(cuMemAlloc(&join_trees,3*8*16*sizeof(int)));check(cuMemAlloc(&join_roots,d.F*sizeof(double)));enabled=true;
        } catch(const std::exception&e){failure=e.what();release();if(mode==2)throw;}
        setup_seconds=now()-start;
    }
    void release(){if(context){cuCtxSetCurrent(context);for(auto p:{zero,pos,rows,costs,actions,histogram,feature_ids,scalar_zero,scalar_pos,scalar_dominated,extra_weights,join_p0,join_n0,join_sides,join_choices,join_costs,join_trees,join_roots})if(p)cuMemFree(p);if(module)cuModuleUnload(module);cuDevicePrimaryCtxRelease(0);}
        zero=pos=rows=costs=actions=histogram=feature_ids=scalar_zero=scalar_pos=scalar_dominated=extra_weights=join_p0=join_n0=join_sides=join_choices=join_costs=join_trees=join_roots=0;module=nullptr;context=nullptr;enabled=false;if(rtc){FreeLibrary(rtc);rtc=nullptr;}if(builtins){FreeLibrary(builtins);builtins=nullptr;}}
    void release_geometry(){
        scalar_geometry.clear();geometry_arena.reset();geometry_bytes=0;
    }
    ~MccGpu(){release();release_geometry();}
    bool triple_counts(const std::vector<std::array<int,3>>&triples,std::vector<std::array<int,16>>&out,double end,Rows domain={}){
        if(!enabled||now()>=end)return false;std::lock_guard<std::mutex> guard(lock);out.resize(triples.size());if(triples.empty())return true;
        CUdeviceptr input=0,output=0;check(cuCtxSetCurrent(context));
        try{check(cuMemAlloc(&input,triples.size()*sizeof(triples[0])));check(cuMemAlloc(&output,out.size()*sizeof(out[0])));
            check(cuMemcpyHtoD(input,triples.data(),triples.size()*sizeof(triples[0])));check(cuMemcpyHtoD(rows,(domain?domain:data.all)->bits.data(),data.W*sizeof(U)));
            for(int first=0;first<int(triples.size());first+=4096){if(now()>=end){cuMemFree(input);cuMemFree(output);return false;}
                int count=std::min(4096,int(triples.size())-first),W=data.W;void*args[]={&zero,&pos,&rows,&W,&input,&first,&count,&output};
                check(cuLaunchKernel(triple_kernel,count,1,1,256,1,1,0,nullptr,args,nullptr));check(cuCtxSynchronize());}
            check(cuMemcpyDtoH(out.data(),output,out.size()*sizeof(out[0])));cuMemFree(input);cuMemFree(output);return true;
        }catch(...){if(input)cuMemFree(input);if(output)cuMemFree(output);throw;}
    }
    Tree leaf(const Rows&r,double a,double b)const{Tree t;t.a[1]=a*r->p<=b*r->n?-1:-2;return t;}
    bool solve(const Rows&r,double a,double b,Tree& tree,double& value,double deadline,int budget=7){
        if(!enabled||now()>=deadline)return false;std::lock_guard<std::mutex> guard(lock);double start=now();
        auto key=std::make_tuple(r->key,a,b);auto cached=cache.find(key);if(cached!=cache.end()){++cache_hits;tree=cached->second[budget].tree;value=a*r->p+b*r->n-cached->second[budget].lo;return true;}
        if(now()>=deadline)return false;
        auto unfinished=pending.find(key);if(unfinished==pending.end()){
            // Evicting unfinished GPU work loses speed only. Its partial
            // maximization is never used as a certified upper bound.
            size_t capacity=std::max(size_t(1),size_t(64)*1024*1024/std::max(size_t(1),values.size()*20));
            if(pending.size()>=capacity)pending.erase(pending.begin());
            unfinished=pending.emplace(key,std::make_unique<Pending>(values.size())).first;
        }else ++continuations;auto&work=*unfinished->second;int begin=work.next;
        check(cuCtxSetCurrent(context));check(cuMemcpyHtoD(rows,r->bits.data(),data.W*sizeof(U)));
        int F=data.F,W=data.W,ml=std::max(1,data.min_leaf),total=F*F*4;
        for(int first=begin;first<total;first+=4096){if(now()>=deadline)break;int count=std::min(4096,total-first);
            CUdeviceptr no_features=0;double gamma=0;int symmetric=0,directions=1,integer_weights=0,compact=0;void* args[]={&zero,&pos,&rows,&F,&W,&ml,&a,&b,&first,&count,&costs,&actions,&histogram,&no_features,&gamma,&no_features,&symmetric,&extra_weights,&directions,&integer_weights,&compact};
            check(cuLaunchKernel(kernel,count,1,1,256,1,1,unsigned(((W*12+7)&~7)+256*12),nullptr,args,nullptr));check(cuCtxSynchronize());work.next=first+count;}
        size_t count=work.next-begin;if(count){check(cuMemcpyDtoH(work.values.data()+begin,costs+size_t(begin)*sizeof(double),count*sizeof(double)));check(cuMemcpyDtoH(work.tails.data()+begin,actions+size_t(begin)*sizeof(int),count*sizeof(int)));check(cuMemcpyDtoH(work.hist.data()+2*begin,histogram+size_t(begin)*2*sizeof(int),count*2*sizeof(int)));}
        if(work.next<total){seconds+=now()-start;return false;}
        values.swap(work.values);tails.swap(work.tails);hist.swap(work.hist);pending.erase(unfinished);
        std::vector<std::array<Tree,8>> trees(F);std::vector<std::array<double,8>> root_cost(F);for(auto&v:root_cost)v.fill(INF);std::vector<std::exception_ptr> errors(F);
        #pragma omp parallel for schedule(dynamic,1) num_threads(threads)
        for(int f=0;f<F;++f)try{
            auto children=data.partition(r,f);if(std::min(children[0]->p+children[0]->n,children[1]->p+children[1]->n)<ml)continue;
            std::array<std::array<Tree,4>,2> side_tree;double side_cost[2][4];
            for(int s=0;s<2;++s){auto sub=children[s];int chosen[4]={-1,-1,-1,-1},alloc[4][2]{};
                for(int k=0;k<4;++k){side_tree[s][k]=leaf(sub,a,b);side_cost[s][k]=std::min(a*sub->p,b*sub->n);}
                for(int g=0;g<F;++g){size_t idx=(size_t(f)*F+g)*4+2*s;if(values[idx]>1e299||values[idx+1]>1e299)continue;
                    for(int i=0;i<2;++i)for(int j=0;j<2;++j){double v=(i?values[idx]:std::min(a*hist[2*idx],b*hist[2*idx+1]))+(j?values[idx+1]:std::min(a*hist[2*idx+2],b*hist[2*idx+3]));
                        for(int k=1+i+j;k<4;++k)if(v<side_cost[s][k]){side_cost[s][k]=v;chosen[k]=g;alloc[k][0]=i;alloc[k][1]=j;}}}
                for(int k=1;k<4;++k)if(chosen[k]>=0){int g=chosen[k];Tree best;best.a[1]=g;auto halves=data.partition(sub,g);
                    for(int z=0;z<2;++z){int h=alloc[k][z]?tails[(size_t(f)*F+g)*4+2*s+z]:leaf(halves[z],a,b).a[1];Tree stump;
                        if(h<0)stump.a[1]=h;else{stump.a[1]=h;auto rr=data.partition(halves[z],h);stump.a[2]=leaf(rr[0],a,b).a[1];stump.a[3]=leaf(rr[1],a,b).a[1];}transplant(best,2+z,stump);}side_tree[s][k]=best;}}
            for(int k=1;k<8;++k)for(int i=0;i<4;++i)for(int j=0;j<4&&1+i+j<=k;++j){double v=side_cost[0][i]+side_cost[1][j];if(v<root_cost[f][k]){root_cost[f][k]=v;Tree t;t.a[1]=f;transplant(t,2,side_tree[0][i]);transplant(t,3,side_tree[1][j]);trees[f][k]=t;}}
        }catch(...){errors[f]=std::current_exception();}
        for(auto&e:errors)if(e)std::rethrow_exception(e);
        std::array<WeightedBound,8> profile;
        for(int k=0;k<8;++k){tree=leaf(r,a,b);value=std::min(a*r->p,b*r->n);for(int f=0;f<F;++f)if(root_cost[f][k]<value){value=root_cost[f][k];tree=trees[f][k];}
            profile[k].tree=tree;profile[k].lo=profile[k].hi=a*r->p+b*r->n-value;}
        if(cache.size()<4096)cache.emplace(std::move(key),profile);tree=profile[budget].tree;value=a*r->p+b*r->n-profile[budget].lo;
        auto c=data.counts(tree,r,1,3);double audit=a*(r->p-c.tp)+b*(r->n-c.tn);
        if(std::abs(value-audit)>1e-8*std::max(1.,a+b))throw std::runtime_error("GPU weighted witness disagrees with native audit");
        ++calls;seconds+=now()-start;return true;
    }
    // Exact additive D3: no K-layer enumeration. Geometry is weight independent;
    // scalar winners are recomputed for every (a,b,gamma). Only the winning tree
    // is recovered, after the parallel reduction over candidate roots.
    void join_packed_budgets(const Rows&r,const ScalarGeometry&g,double a,double b){
        int F=int(g.ids.size());struct Side{std::array<double,4> cost;std::array<std::array<int,3>,4> choice;};
        std::vector<std::array<Side,2>> sides(F);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int f=0;f<F;++f)for(int side=0;side<2;++side){auto&z=sides[f][side];int p=side?r->p-g.p0[f]:g.p0[f],n=side?r->n-g.n0[f]:g.n0[f];z.cost.fill(std::min(a*p,b*n));for(auto&c:z.choice)c={-1,0,0};
            if(F>1)for(int j=0;j<F;++j){if(j==f)continue;size_t idx=(size_t(f)*F+j)*4+2*side;if(values[idx]>1e299||values[idx+1]>1e299)continue;
                for(int l=0;l<2;++l)for(int rr=0;rr<2;++rr){double cost=(l?values[idx]:std::min(a*hist[2*idx],b*hist[2*idx+1]))+(rr?values[idx+1]:std::min(a*hist[2*idx+2],b*hist[2*idx+3]));
                    for(int k=1+l+rr;k<4;++k)if(cost<z.cost[k]){z.cost[k]=cost;z.choice[k]={j,l,rr};}}}}
        for(int k=0;k<8;++k){double best=std::min(a*r->p,b*r->n);int root=-1,lk=0,rk=0;
            for(int f=0;f<F;++f)for(int i=0;i<4;++i)for(int j=0;j<4&&1+i+j<=k;++j){double v=sides[f][0].cost[i]+sides[f][1].cost[j];if(v<best){best=v;root=f;lk=i;rk=j;}}
            Tree tree=leaf(r,a,b);if(root>=0){tree=Tree();tree.a[1]=g.ids[root];for(int side=0;side<2;++side){auto c=sides[root][side].choice[side?rk:lk];Tree child;
                    int p=side?r->p-g.p0[root]:g.p0[root],n=side?r->n-g.n0[root]:g.n0[root];child.a[1]=a*p<=b*n?-1:-2;
                    if(c[0]>=0){child.a[1]=g.ids[c[0]];for(int z=0;z<2;++z){size_t idx=(size_t(root)*F+c[0])*4+2*side+z;int action=c[1+z]?tails[idx]:(a*hist[2*idx]<=b*hist[2*idx+1]?-1:-2);Tree stump;
                            if(action<0)stump.a[1]=action;else{stump.a[1]=action/4;stump.a[2]=-1-(action&1);stump.a[3]=-1-((action>>1)&1);}transplant(child,2+z,stump);}}
                    transplant(tree,2+side,child);}}
            packed_budgets[k].tree=tree;packed_budgets[k].lo=packed_budgets[k].hi=a*r->p+b*r->n-best;
        }
    }
    bool reduce_device(const Rows&r,const ScalarGeometry&g,double a,double b,double gamma,bool budgets,const std::vector<std::array<double,2>>&extra){
        int F=int(g.ids.size()),P=r->p,N=r->n,D=1+int(extra.size()),use_budgets=budgets?1:0;
        double start=now();check(cuMemcpyHtoD(join_p0,g.p0.data(),F*sizeof(int)));check(cuMemcpyHtoD(join_n0,g.n0.data(),F*sizeof(int)));
        void* side_args[]={&F,&P,&N,&join_p0,&join_n0,&costs,&histogram,&a,&b,&extra_weights,&gamma,&use_budgets,&join_sides,&join_choices};
        check(cuLaunchKernel(side_join_kernel,2*F,D,1,64,1,1,0,nullptr,side_args,nullptr));
        void* root_args[]={&F,&P,&N,&join_p0,&join_n0,&feature_ids,&histogram,&actions,&join_sides,&join_choices,&a,&b,&extra_weights,&gamma,&use_budgets,&join_costs,&join_trees,&join_roots};
        check(cuLaunchKernel(root_join_kernel,budgets?8:1,D,1,256,1,1,0,nullptr,root_args,nullptr));
        std::array<double,24> values{};std::array<int,384> trees{};
        check(cuMemcpyDtoH(values.data(),join_costs,(budgets?D*8:1)*sizeof(double)));check(cuMemcpyDtoH(trees.data(),join_trees,(budgets?D*8:1)*16*sizeof(int)));
        if(!budgets&&scalar_root_detail){std::vector<double> roots(F);check(cuMemcpyDtoH(roots.data(),join_roots,F*sizeof(double)));
            for(int f=0;f<data.F;++f)scalar_root_costs[f]=g.alias[f]<0?INF:roots[g.alias[f]];}
        for(int d=0;d<D;++d){double aa=d?extra[d-1][0]:a,bb=d?extra[d-1][1]:b;std::array<WeightedBound,8> profile;
            for(int k=0;k<(budgets?8:1);++k){int index=d*8+k;std::copy_n(trees.begin()+index*16,16,profile[k].tree.a.begin());profile[k].lo=profile[k].hi=aa*P+bb*N-values[index];}
            if(!d)packed_budgets=profile;else{size_t bytes=r->key.size()+sizeof(profile)+128;
                if(budget_cache_bytes+bytes<=geometry_limit/2&&cache.emplace(std::make_tuple(r->key,aa,bb),profile).second){budget_cache_bytes+=bytes;++batched_profiles;}}}
        ++device_joins;join_seconds+=now()-start;return true;
    }
    ScalarGeometry make_geometry(const Rows&r,bool parallel){
        geometry_arena.configure(geometry_limit);ScalarGeometry temporary(&geometry_arena);std::map<std::string,int> seen;temporary.alias.assign(data.F,-1);for(int f=0;f<data.F;++f){auto rr=data.partition(r,f);
                if(std::min(rr[0]->p+rr[0]->n,rr[1]->p+rr[1]->n)<std::max(1,data.min_leaf))continue;
                auto inserted=seen.emplace(std::min(rr[0]->key,rr[1]->key),int(temporary.ids.size()));temporary.alias[f]=inserted.first->second;if(!inserted.second)continue;
                temporary.ids.push_back(f);temporary.p0.push_back(rr[0]->p);temporary.n0.push_back(rr[0]->n);}
            // Compact routed observations once; reuse the weight-independent
            // bitsets across all adaptive directions. Feature IDs in witnesses
            // remain global even though the GPU matrix uses local columns.
            std::vector<int> samples;for(int w=0;w<data.W;++w){U bits=r->bits[w];while(bits){unsigned long bit;_BitScanForward64(&bit,bits);samples.push_back(w*64+int(bit));bits&=bits-1;}}
            int W=(int(samples.size())+63)/64,F=int(temporary.ids.size());temporary.zero.resize(size_t(F)*W);temporary.pos.resize(W);temporary.mask.assign(W,~U(0));
            if(samples.size()%64)temporary.mask.back()=(U(1)<<(samples.size()%64))-1;
            for(size_t i=0;i<samples.size();++i)if(data.y[samples[i]])temporary.pos[i/64]|=U(1)<<(i%64);
            #pragma omp parallel for schedule(static) num_threads(threads) if(parallel)
            for(int j=0;j<F;++j){int f=temporary.ids[j];for(int w=0;w<W;++w){U bits=0;size_t stop=std::min(samples.size(),size_t(w+1)*64);for(size_t i=size_t(w)*64;i<stop;++i)if(!data.X[size_t(samples[i])*data.F+f])bits|=U(1)<<(i%64);temporary.zero[size_t(w)*F+j]=bits;}}
        return temporary;
    }
    size_t geometry_size(const Rows&r,const ScalarGeometry&temporary)const{return r->key.size()+(temporary.ids.size()*3+temporary.alias.size())*sizeof(int)+(temporary.zero.size()+temporary.pos.size()+temporary.mask.size())*sizeof(U)+384;}
    void prefill_geometry(const std::vector<Rows>&domains,double end){
        double start=now();std::vector<Rows> missing;{std::lock_guard<std::mutex> guard(geometry_lock);for(auto&r:domains)if(r->p&&r->n&&!scalar_geometry.count(r->key))missing.push_back(r);}
        std::atomic<bool> full{false};std::vector<std::exception_ptr> errors(missing.size());
        #pragma omp parallel for schedule(dynamic,1) num_threads(threads)
        for(int i=0;i<int(missing.size());++i){if(full||cancel_prefill||now()>=end)continue;
            try{auto g=make_geometry(missing[i],false);size_t used=geometry_size(missing[i],g);
                {std::lock_guard<std::mutex> guard(geometry_lock);if(!scalar_geometry.count(missing[i]->key)){
                    if(geometry_bytes+used<=geometry_limit){geometry_bytes+=used;scalar_geometry.emplace(missing[i]->key,std::move(g));}else full=true;}}geometry_ready.notify_all();
            }catch(...){errors[i]=std::current_exception();}}
        prefill_done=true;geometry_ready.notify_all();for(auto&e:errors)if(e)std::rethrow_exception(e);if(async_geometry)prefill_seconds+=now()-start;else geometry_seconds+=now()-start;
    }
    bool scalar(const Rows&r,double a,double b,double gamma,Tree&tree,double&value,double deadline,bool budgets=false){
        if(!enabled||now()>=deadline)return false;std::lock_guard<std::mutex> guard(lock);double start=now();
        if(budgets&&gamma!=0)throw std::invalid_argument("packed budget profile needs zero split cost");
        if(budgets){auto known=cache.find({r->key,a,b});if(known!=cache.end()){packed_budgets=known->second;tree=packed_budgets[7].tree;value=a*r->p+b*r->n-packed_budgets[7].lo;++cache_hits;return true;}}
        scalar_root_costs.assign(data.F,gamma);
        tree=leaf(r,a,b);value=std::min(a*r->p,b*r->n);if(value<=gamma){if(budgets)for(auto&z:packed_budgets){z.tree=tree;z.lo=z.hi=a*r->p+b*r->n-value;}++scalar_completed;++scalar_leaf_prunes;seconds+=now()-start;return true;}
        double geometry_start=now();ScalarGeometry temporary(&geometry_arena);const ScalarGeometry*geometry=nullptr;
        {std::unique_lock<std::mutex> guard(geometry_lock);auto known=scalar_geometry.find(r->key);
            if(known==scalar_geometry.end()&&async_geometry&&!prefill_done){geometry_ready.wait_for(guard,std::chrono::duration<double>(std::max(0.,std::min(.02,deadline-now()))),[&](){return prefill_done||scalar_geometry.count(r->key);});known=scalar_geometry.find(r->key);}
            if(known!=scalar_geometry.end()){geometry=&known->second;++geometry_hits;}}
        if(!geometry){temporary=make_geometry(r,!async_geometry);size_t used=geometry_size(r,temporary);
            std::lock_guard<std::mutex> guard(geometry_lock);auto known=scalar_geometry.find(r->key);
            if(known!=scalar_geometry.end())geometry=&known->second;
            else if(geometry_bytes+used<=geometry_limit){geometry_bytes+=used;geometry=&scalar_geometry.emplace(r->key,std::move(temporary)).first->second;}else geometry=&temporary;}
        geometry_seconds+=now()-geometry_start;const auto&g=*geometry;int F=int(g.ids.size()),W=int(g.pos.size()),ml=std::max(1,data.min_leaf),total=4*F*F;
        tree=leaf(r,a,b);value=std::min(a*r->p,b*r->n);
        std::vector<std::array<double,2>> extra;
        if(budgets)for(auto w:budget_directions)if((w[0]!=a||w[1]!=b)&&extra.size()<2)extra.push_back(w);
        int directions=1+int(extra.size());
        auto dyadic=[&](double x){return x>=0&&x*1024==std::floor(x*1024)&&x*1024*(r->p+r->n)<INT_MAX;};
        int integer_weights=dyadic(a)&&dyadic(b);for(auto w:extra)integer_weights=integer_weights&&dyadic(w[0])&&dyadic(w[1]);if(integer_weights)++integer_scalar_calls;
        if(F>1){double transfer_start=now();check(cuCtxSetCurrent(context));if(!extra.empty())check(cuMemcpyHtoD(extra_weights,extra.data(),extra.size()*sizeof(extra[0])));check(cuMemcpyHtoD(rows,g.mask.data(),W*sizeof(U)));check(cuMemcpyHtoD(feature_ids,g.ids.data(),F*sizeof(int)));
            check(cuMemcpyHtoD(scalar_zero,g.zero.data(),g.zero.size()*sizeof(U)));check(cuMemcpyHtoD(scalar_pos,g.pos.data(),W*sizeof(U)));
            // If stopping at a child is no worse than one split, no deeper
            // feature pair in that child can improve its scalar optimum.
            std::vector<unsigned char> dominated(2*F);for(int f=0;f<F;++f){dominated[2*f]=std::min(a*g.p0[f],b*g.n0[f])<=gamma;dominated[2*f+1]=std::min(a*(r->p-g.p0[f]),b*(r->n-g.n0[f]))<=gamma;}
            check(cuMemcpyHtoD(scalar_dominated,dominated.data(),dominated.size()));
            transfer_seconds+=now()-transfer_start;double kernel_start=now();
            // Several tiles per synchronization; the global deadline is still
            // checked between bounded batches. No partial maximum is certified.
            int compact=symmetric_scalar_pairs?1:0,tasks=compact?2*F*(F-1):total;
            for(int first=0;first<tasks;first+=65536){if(now()>=deadline){kernel_seconds+=now()-kernel_start;seconds+=now()-start;return false;}int count=std::min(65536,tasks-first);
                int symmetric=symmetric_scalar_pairs?1:0;void*args[]={&scalar_zero,&scalar_pos,&rows,&F,&W,&ml,&a,&b,&first,&count,&costs,&actions,&histogram,&feature_ids,&gamma,&scalar_dominated,&symmetric,&extra_weights,&directions,&integer_weights,&compact};
                int block=256;if(adaptive_scalar_blocks){block=32;while(block<F&&block<64)block*=2;}
                while(block>32&&((W*12+7)&~7)+block*directions*12+2052>shared_limit)block/=2;
                unsigned shared_bytes=unsigned(((W*12+7)&~7)+block*directions*12);
                check(cuLaunchKernel(kernels[integer_weights][directions-1],count,1,1,block,1,1,shared_bytes,nullptr,args,nullptr));
                // The final blocking witness copy synchronizes the same stream.
                // Intermediate tiles still synchronize before deadline checks.
                if(scalar_explicit_sync||first+count<tasks||!device_reduction)check(cuCtxSynchronize());}
            kernel_seconds+=now()-kernel_start;
            if(device_reduction){reduce_device(r,g,a,b,gamma,budgets,extra);int k=budgets?7:0;tree=packed_budgets[k].tree;value=a*r->p+b*r->n-packed_budgets[k].lo;
                ++scalar_completed;++scalar_calls;scalar_pairs+=total;seconds+=now()-start;return true;}
            transfer_start=now();check(cuMemcpyDtoH(values.data(),costs,total*sizeof(double)));check(cuMemcpyDtoH(tails.data(),actions,total*sizeof(int)));if(budgets)check(cuMemcpyDtoH(hist.data(),histogram,2*total*sizeof(int)));transfer_seconds+=now()-transfer_start;scalar_pairs+=total;}
        double join_start=now();if(budgets){join_packed_budgets(r,g,a,b);auto primary=packed_budgets;
            for(size_t d=0;d<extra.size();++d){auto w=extra[d];if(F>1){check(cuMemcpyDtoH(values.data(),costs+(d+1)*total*sizeof(double),total*sizeof(double)));check(cuMemcpyDtoH(tails.data(),actions+(d+1)*total*sizeof(int),total*sizeof(int)));}
                join_packed_budgets(r,g,w[0],w[1]);size_t bytes=r->key.size()+sizeof(packed_budgets)+128;
                if(budget_cache_bytes+bytes<=geometry_limit/2&&cache.emplace(std::make_tuple(r->key,w[0],w[1]),packed_budgets).second){budget_cache_bytes+=bytes;++batched_profiles;}}
            packed_budgets=primary;tree=packed_budgets[7].tree;value=a*r->p+b*r->n-packed_budgets[7].lo;++scalar_completed;if(F>1)++scalar_calls;join_seconds+=now()-join_start;seconds+=now()-start;return true;}
        std::vector<double> root_cost(F,INF);std::vector<std::array<int,2>> choices(F);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int f=0;f<F;++f){double cost=gamma;for(int side=0;side<2;++side){int p=side?r->p-g.p0[f]:g.p0[f],n=side?r->n-g.n0[f]:g.n0[f];
                double best=std::min(a*p,b*n);int selected=-1;
                for(int j=0;j<F&&F>1;++j){if(j==f)continue;size_t idx=(size_t(f)*F+j)*4+2*side;double v=gamma+values[idx]+values[idx+1];if(v<best){best=v;selected=j;}}
                choices[f][side]=selected;cost+=best;}root_cost[f]=cost;}
        for(int f=0;f<data.F;++f)scalar_root_costs[f]=g.alias[f]<0?INF:root_cost[g.alias[f]];
        int root=-1;for(int f=0;f<F;++f)if(root_cost[f]<value){root=f;value=root_cost[f];}
        if(root>=0){tree=Tree();tree.a[1]=g.ids[root];
            for(int side=0;side<2;++side){int j=choices[root][side];Tree child;int p=side?r->p-g.p0[root]:g.p0[root],n=side?r->n-g.n0[root]:g.n0[root];child.a[1]=a*p<=b*n?-1:-2;
                if(j>=0){child.a[1]=g.ids[j];
                    for(int z=0;z<2;++z){int h=tails[(size_t(root)*F+j)*4+2*side+z];Tree stump;
                        if(h<0)stump.a[1]=h;else{stump.a[1]=h/4;stump.a[2]=-1-(h&1);stump.a[3]=-1-((h>>1)&1);}transplant(child,2+z,stump);}}
                transplant(tree,2+side,child);}}
        auto c=data.counts(tree,r,1,3);double audit=a*(r->p-c.tp)+b*(r->n-c.tn)+gamma*c.k;
        if(std::abs(audit-value)>1e-8*std::max({1.,a+b+gamma,std::abs(value)}))throw std::runtime_error("additive D3 witness mismatch");
        ++scalar_completed;if(F>1)++scalar_calls;join_seconds+=now()-join_start;seconds+=now()-start;return true;
    }
};
}
