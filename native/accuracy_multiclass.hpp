#pragma once
#include "accuracy_histogram_kernels.hpp"
// JT-OCT accuracy-only pricing. Label-segmented words remove per-class
// bitset intersections; PEXT packs routed observations without scanning rows.
namespace accuracy {
struct MultiOracle {
    std::vector<unsigned char> binary;
    cm5::Data data;
    cm5::MccGpu owner;
    const unsigned char* y;
    int K,threads;
    std::vector<cm5::U> labels;
    CUmodule module=nullptr;
    CUfunction cost_kernel,warp_kernel,side_kernel,root_kernel;
    CUdeviceptr zero=0,mask=0,ends=0,ids=0,counts=0,left=0,costs=0,actions=0,side_cost=0,side_choice=0,value_out=0,tree_out=0;
    double prepare_seconds=0,gpu_seconds=0;
    struct Reference {std::vector<cm5::U> rows;std::vector<double> roots;cm5::Tree tree;double value,gamma;int depth,size;double largest_bound;};
    std::vector<Reference> references;
    CUdeviceptr eligible=0,root_values=0;
    CUdeviceptr pair_histogram=0,triple_histogram=0;
    CUfunction pair_histogram_kernel,triple_histogram_kernel,histogram_cost_kernel;
    std::array<uint64_t,3> diagnostics{}; // considered roots, excluded roots, transfer-only solves
    static std::vector<unsigned char> bin(const unsigned char*y,int n){std::vector<unsigned char>b(n);for(int i=0;i<n;++i)b[i]=y[i]!=0;return b;}
    MultiOracle(const unsigned char*x,const unsigned char*yy,int n,int f,int k,int ml,int nt,const wchar_t*rtc)
        :binary(bin(yy,n)),data(x,binary.data(),n,f,ml),owner(data,nt,2,rtc,true),y(yy),K(k),threads(nt),labels(size_t(k)*data.W){
        if(k<2||k>32)throw std::invalid_argument("native accuracy supports 2..32 labels");
        for(int i=0;i<n;++i){if(y[i]>=K)throw std::invalid_argument("invalid class index");labels[size_t(y[i])*data.W+i/64]|=cm5::U(1)<<(i%64);}
        const char* body=R"CUDA(
typedef unsigned long long U;
extern "C" __global__ void accuracy_cost(const U*z,const U*mask,const int*end,const int*ids,int F,int W,int ml,int depth,double gamma,int first,int count,double*out,int*act,int streamed,const int*root_counts,const int*root_left,const int*eligible){
 int task=first+blockIdx.x;if(blockIdx.x>=count)return;int pair=task/4,q=task%4;
 int g=(int)((1+sqrtf(1+8.0f*pair))*.5f);while(g*(g-1)/2>pair)--g;while(g*(g+1)/2<=pair)++g;
 int f=pair-g*(g-1)/2,t=threadIdx.x,index=(f*F+g)*4+q,mirror=(g*F+f)*4+2*(q%2)+q/2;
 if(!eligible[f]&&!eligible[g])return;
 extern __shared__ U bits[];__shared__ int hist[K*64];__shared__ double v[64];__shared__ int choice[64];
 int total[K],N=0,wstart=0;
 for(int c=0;c<K;++c){int local=0;for(int w=wstart+t;w<end[c];w+=64){U a=z[w*F+f],b=z[w*F+g];U r=mask[w]&((q/2)?~a:a)&((q%2)?~b:b);if(!streamed)bits[w]=r;local+=__popcll(r);}hist[c*64+t]=local;wstart=end[c];}
 __syncthreads();for(int step=32;step;step/=2){if(t<step)for(int c=0;c<K;++c)hist[c*64+t]+=hist[c*64+t+step];__syncthreads();}
 int label=0;for(int c=0;c<K;++c){total[c]=hist[c*64];N+=total[c];if(total[c]>total[label])label=c;}
 double best=N<ml?1e300:double(N-total[label]);int winner=-label-1;
 if(depth==3&&best>gamma&&N>=ml)for(int h=t;h<F;h+=64){if(h==f||h==g)continue;
   // Every child retaining the current majority label has the STOP error
   // plus a nonnegative split cost. Bound all competitors by their marginal
   // class support, and stop scanning as soon as both labels are certified.
   int upper0=0,upper1=0;for(int c=0;c<K;++c)if(c!=label){upper0=max(upper0,min(total[c],root_left[c*F+h]));upper1=max(upper1,min(total[c],root_counts[c]-root_left[c*F+h]));}
   int majority0=0,majority1=0;bool dominated=false;
   for(int w=label?end[label-1]:0;w<end[label];++w){U r;if(streamed){U a=z[w*F+f],b=z[w*F+g];r=mask[w]&((q/2)?~a:a)&((q%2)?~b:b);}else r=bits[w];U x=r&z[w*F+h];majority0+=__popcll(x);majority1+=__popcll(r^x);if(majority0>=upper0&&majority1>=upper1){dominated=true;break;}}
   if(dominated)continue;
   int ltotal=0,rbest=0,lbest=0,ll=0,rr=0,start=0;
   for(int c=0;c<K;++c){int n0=c==label?majority0:0;if(c!=label)for(int w=start;w<end[c];++w){U r;if(streamed){U a=z[w*F+f],b=z[w*F+g];r=mask[w]&((q/2)?~a:a)&((q%2)?~b:b);}else r=bits[w];n0+=__popcll(r&z[w*F+h]);}
     start=end[c];ltotal+=n0;if(n0>lbest){lbest=n0;ll=c;}if(total[c]-n0>rbest){rbest=total[c]-n0;rr=c;}}
   if(ltotal<ml||N-ltotal<ml)continue;double cost=gamma+N-lbest-rbest;
   int a=ids[h]*K*K+ll*K+rr;if(cost<best||(cost==best&&a<winner)){best=cost;winner=a;}}
 v[t]=best;choice[t]=winner;__syncthreads();for(int s=32;s;s/=2){if(t<s&&(v[t+s]<v[t]||(v[t+s]==v[t]&&choice[t+s]<choice[t]))){v[t]=v[t+s];choice[t]=choice[t+s];}__syncthreads();}
 if(!t){out[index]=out[mirror]=v[0];act[index]=act[mirror]=choice[0];}
}
extern "C" __global__ void accuracy_side(int F,int ml,const int*counts,const int*left,const double*cost,double gamma,double*out,int*action,const int*eligible){
 int f=blockIdx.x/2,s=blockIdx.x%2,t=threadIdx.x,N=0,m=0,label=0;
 if(!eligible[f])return;
 for(int c=0;c<K;++c){int n=s?counts[c]-left[c*F+f]:left[c*F+f];N+=n;if(n>m){m=n;label=c;}}
 double best=N<ml?1e300:double(N-m);int winner=-label-1;
 for(int g=t;g<F;g+=64){if(g==f)continue;int i=(f*F+g)*4+2*s;double v=gamma+cost[i]+cost[i+1];if(v<best||(v==best&&g<winner)){best=v;winner=g;}}
 __shared__ double v[64];__shared__ int a[64];v[t]=best;a[t]=winner;__syncthreads();for(int k=32;k;k/=2){if(t<k&&(v[t+k]<v[t]||(v[t+k]==v[t]&&a[t+k]<a[t]))){v[t]=v[t+k];a[t]=a[t+k];}__syncthreads();}if(!t){out[2*f+s]=v[0];action[2*f+s]=a[0];}
}
extern "C" __global__ void accuracy_cost_warp(const U*z,const U*mask,const int*end,const int*ids,int F,int W,int ml,int depth,double gamma,int first,int count,double*out,int*act,int streamed,const int*root_counts,const int*root_left,const int*eligible){
 int task=first+blockIdx.x;if(blockIdx.x>=count)return;int pair=task/4,q=task%4;
 int g=(int)((1+sqrtf(1+8.0f*pair))*.5f);while(g*(g-1)/2>pair)--g;while(g*(g+1)/2<=pair)++g;
 int f=pair-g*(g-1)/2,t=threadIdx.x,lane=t%32,warp=t/32,index=(f*F+g)*4+q,mirror=(g*F+f)*4+2*(q%2)+q/2;
 if(!eligible[f]&&!eligible[g])return;
 extern __shared__ U bits[];__shared__ int hist[K*128];__shared__ double values[4];__shared__ int choices[4];
 int total[K],N=0,start=0;
 for(int c=0;c<K;++c){int n=0;for(int w=start+t;w<end[c];w+=128){U a=z[f*W+w],b=z[g*W+w];U r=mask[w]&((q/2)?~a:a)&((q%2)?~b:b);if(!streamed)bits[w]=r;n+=__popcll(r);}hist[c*128+t]=n;start=end[c];}
 __syncthreads();for(int step=64;step;step/=2){if(t<step)for(int c=0;c<K;++c)hist[c*128+t]+=hist[c*128+t+step];__syncthreads();}
 int label=0;for(int c=0;c<K;++c){total[c]=hist[c*128];N+=total[c];if(total[c]>total[label])label=c;}
 double best=N<ml?1e300:double(N-total[label]);int winner=-label-1;
 if(depth==3&&N>=ml&&best>gamma)for(int h=warp;h<F;h+=4){if(h==f||h==g)continue;
   int upper0=0,upper1=0;for(int c=0;c<K;++c)if(c!=label){upper0=max(upper0,min(total[c],root_left[c*F+h]));upper1=max(upper1,min(total[c],root_counts[c]-root_left[c*F+h]));}
   int majority0=0,majority1=0;bool dominated=false;
   for(int base=label?end[label-1]:0;base<end[label];base+=128){int a0=0,a1=0;for(int w=base+lane;w<min(base+128,end[label]);w+=32){U r;if(streamed){U a=z[f*W+w],b=z[g*W+w];r=mask[w]&((q/2)?~a:a)&((q%2)?~b:b);}else r=bits[w];U v=r&z[h*W+w];a0+=__popcll(v);a1+=__popcll(r^v);}
     for(int s=16;s;s/=2){a0+=__shfl_xor_sync(0xffffffff,a0,s);a1+=__shfl_xor_sync(0xffffffff,a1,s);}majority0+=a0;majority1+=a1;
     if(majority0>=upper0&&majority1>=upper1){dominated=true;break;}}
   if(dominated)continue;int ltotal=0,lbest=0,rbest=0,ll=0,rr=0,start=0;
   for(int c=0;c<K;++c){int n0=0;if(c==label)n0=majority0;else{for(int w=start+lane;w<end[c];w+=32){U r;if(streamed){U a=z[f*W+w],b=z[g*W+w];r=mask[w]&((q/2)?~a:a)&((q%2)?~b:b);}else r=bits[w];n0+=__popcll(r&z[h*W+w]);}for(int s=16;s;s/=2)n0+=__shfl_xor_sync(0xffffffff,n0,s);}
     start=end[c];ltotal+=n0;if(n0>lbest){lbest=n0;ll=c;}if(total[c]-n0>rbest){rbest=total[c]-n0;rr=c;}}
   if(ltotal<ml||N-ltotal<ml)continue;double v=gamma+N-lbest-rbest;int a=ids[h]*K*K+ll*K+rr;if(v<best||(v==best&&a<winner)){best=v;winner=a;}}
 if(!lane){values[warp]=best;choices[warp]=winner;}__syncthreads();if(!t){for(int i=1;i<4;++i)if(values[i]<values[0]||(values[i]==values[0]&&choices[i]<choices[0])){values[0]=values[i];choices[0]=choices[i];}out[index]=out[mirror]=values[0];act[index]=act[mirror]=choices[0];}
}
extern "C" __global__ void accuracy_root(int F,int ml,const int*counts,const int*ids,const int*tail,const double*cost,const int*action,double gamma,double*out,int*tree,const int*eligible,double*roots){
 int t=threadIdx.x,N=0,m=0,label=0;for(int c=0;c<K;++c){N+=counts[c];if(counts[c]>m){m=counts[c];label=c;}}
 double best=N<ml?1e300:double(N-m);int winner=-1;for(int f=t;f<F;f+=64){if(!eligible[f])continue;double v=gamma+cost[2*f]+cost[2*f+1];roots[f]=v;if(v<best||(v==best&&f<winner)){best=v;winner=f;}}
 __shared__ double v[64];__shared__ int a[64];v[t]=best;a[t]=winner;__syncthreads();for(int k=32;k;k/=2){if(t<k&&(v[t+k]<v[t]||(v[t+k]==v[t]&&a[t+k]<a[t]))){v[t]=v[t+k];a[t]=a[t+k];}__syncthreads();}
 if(t)return;*out=v[0];for(int j=0;j<16;++j)tree[j]=-99;tree[1]=-label-1;int f=a[0];if(f<0)return;tree[1]=ids[f];
 for(int s=0;s<2;++s){int g=action[2*f+s],node=2+s;if(g<0){tree[node]=g;continue;}tree[node]=ids[g];for(int q=0;q<2;++q){int h=tail[(f*F+g)*4+2*s+q],slot=2*node+q;if(h<0)tree[slot]=h;else{tree[slot]=h/(K*K);tree[2*slot]=-1-(h/K)%K;tree[2*slot+1]=-1-h%K;}}}
}
)CUDA";
        std::string source="#define K "+std::to_string(K)+"\n"+body+histogram_source;
        auto create=owner.proc<decltype(&nvrtcCreateProgram)>("nvrtcCreateProgram");auto compile=owner.proc<decltype(&nvrtcCompileProgram)>("nvrtcCompileProgram");
        auto destroy=owner.proc<decltype(&nvrtcDestroyProgram)>("nvrtcDestroyProgram");nvrtcProgram program=nullptr;
        create(&program,source.c_str(),"accuracy.cu",0,nullptr,nullptr);const char* options[]={"--std=c++11","--gpu-architecture=compute_75","--fmad=false"};
        if(compile(program,3,options)!=NVRTC_SUCCESS){size_t n=0;owner.proc<decltype(&nvrtcGetProgramLogSize)>("nvrtcGetProgramLogSize")(program,&n);std::string log(n,' ');owner.proc<decltype(&nvrtcGetProgramLog)>("nvrtcGetProgramLog")(program,log.data());destroy(&program);throw std::runtime_error(log);}
        size_t bytes=0;owner.proc<decltype(&nvrtcGetPTXSize)>("nvrtcGetPTXSize")(program,&bytes);std::vector<char>ptx(bytes);owner.proc<decltype(&nvrtcGetPTX)>("nvrtcGetPTX")(program,ptx.data());destroy(&program);
        cm5::MccGpu::check(cuModuleLoadData(&module,ptx.data()));cm5::MccGpu::check(cuModuleGetFunction(&cost_kernel,module,"accuracy_cost"));cm5::MccGpu::check(cuModuleGetFunction(&side_kernel,module,"accuracy_side"));cm5::MccGpu::check(cuModuleGetFunction(&root_kernel,module,"accuracy_root"));
        cm5::MccGpu::check(cuModuleGetFunction(&warp_kernel,module,"accuracy_cost_warp"));
        cm5::MccGpu::check(cuModuleGetFunction(&pair_histogram_kernel,module,"accuracy_pair_hist"));
        cm5::MccGpu::check(cuModuleGetFunction(&triple_histogram_kernel,module,"accuracy_triple_hist"));
        cm5::MccGpu::check(cuModuleGetFunction(&histogram_cost_kernel,module,"accuracy_hist_cost"));
        auto alloc=[](CUdeviceptr&p,size_t n){cm5::MccGpu::check(cuMemAlloc(&p,n));};int W=data.W+K;
        alloc(zero,size_t(W)*f*8);alloc(mask,size_t(W)*8);alloc(ends,K*4);alloc(ids,f*4);alloc(counts,K*4);alloc(left,size_t(K)*f*4);
        alloc(costs,size_t(f)*f*4*8);alloc(actions,size_t(f)*f*4*4);alloc(side_cost,f*2*8);alloc(side_choice,f*2*4);alloc(value_out,8);alloc(tree_out,16*4);
        alloc(eligible,f*4);alloc(root_values,f*8);
        if(K==3&&f>=64&&f<=384){alloc(pair_histogram,size_t(f)*(f-1)/2*K*4);alloc(triple_histogram,size_t(f)*(f-1)*(f-2)/6*K*4);}
    }
    ~MultiOracle(){cuCtxSetCurrent(owner.context);for(auto p:{zero,mask,ends,ids,counts,left,costs,actions,side_cost,side_choice,value_out,tree_out,eligible,root_values,pair_histogram,triple_histogram})if(p)cuMemFree(p);if(module)cuModuleUnload(module);}
    struct Geometry {std::vector<int>ids,counts,left,end;std::vector<cm5::U>zero,mask;};
    Geometry geometry(const cm5::U*r){
        Geometry g;g.counts.assign(K,0);std::vector<std::vector<std::pair<int,cm5::U>>> words(K);
        for(int c=0;c<K;++c)for(int w=0;w<data.W;++w){auto bits=r[w]&labels[size_t(c)*data.W+w];if(bits){words[c].push_back({w,bits});g.counts[c]+=int(__popcnt64(bits));}}
        int W=0;for(int c=0;c<K;++c){W+=(g.counts[c]+63)/64;g.end.push_back(W);for(int w=0;w<(g.counts[c]+63)/64;++w)g.mask.push_back(w==(g.counts[c]/64)&&g.counts[c]%64?(cm5::U(1)<<(g.counts[c]%64))-1:~cm5::U(0));}
        std::vector<cm5::U> all(size_t(data.F)*W);std::vector<int>lc(size_t(data.F)*K);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int f=0;f<data.F;++f){int start=0;for(int c=0;c<K;++c){int position=0,count=0;for(auto word:words[c]){cm5::U bits=_pext_u64(data.zero[size_t(f)*data.W+word.first],word.second);int size=int(__popcnt64(word.second));count+=int(__popcnt64(bits));int at=start+position/64,shift=position%64;all[size_t(f)*W+at]|=bits<<shift;if(shift&&shift+size>64)all[size_t(f)*W+at+1]|=bits>>(64-shift);position+=size;}lc[size_t(c)*data.F+f]=count;start=g.end[c];}}
        std::unordered_set<std::string>seen;std::vector<cm5::U> complement(W);int N=std::accumulate(g.counts.begin(),g.counts.end(),0);
        for(int f=0;f<data.F;++f){int n=0;for(int c=0;c<K;++c)n+=lc[size_t(c)*data.F+f];if(std::min(n,N-n)<std::max(1,data.min_leaf))continue;auto p=all.data()+size_t(f)*W;
            for(int w=0;w<W;++w)complement[w]=p[w]^g.mask[w];std::string a(reinterpret_cast<char*>(p),W*8),b(reinterpret_cast<char*>(complement.data()),W*8);if(seen.insert(std::min(a,b)).second)g.ids.push_back(f);}
        int F=int(g.ids.size());g.zero.resize(size_t(F)*W);g.left.resize(size_t(F)*K);
        for(int j=0;j<F;++j){for(int w=0;w<W;++w)g.zero[W>64?size_t(j)*W+w:size_t(w)*F+j]=all[size_t(g.ids[j])*W+w];for(int c=0;c<K;++c)g.left[size_t(c)*F+j]=lc[size_t(c)*data.F+g.ids[j]];}
        return g;
    }
    double restrict_tree(const cm5::Tree&source,const std::vector<cm5::U>&r,cm5::Tree&out,int src,int dst,double gamma){
        int f=source.a[src];
        if(f<0){int N=0,best=0,label=0;for(int c=0;c<K;++c){int n=0;for(int w=0;w<data.W;++w)n+=int(__popcnt64(r[w]&labels[size_t(c)*data.W+w]));N+=n;if(n>best){best=n;label=c;}}out.a[dst]=-label-1;return N-best;}
        std::vector<cm5::U>a(data.W),b(data.W);bool hasa=false,hasb=false;for(int w=0;w<data.W;++w){a[w]=r[w]&data.zero[size_t(f)*data.W+w];b[w]=r[w]^a[w];hasa|=a[w]!=0;hasb|=b[w]!=0;}
        if(!hasa)return restrict_tree(source,b,out,2*src+1,dst,gamma);if(!hasb)return restrict_tree(source,a,out,2*src,dst,gamma);
        out.a[dst]=f;return gamma+restrict_tree(source,a,out,2*src,2*dst,gamma)+restrict_tree(source,b,out,2*src+1,2*dst+1,gamma);
    }
    void remember(const cm5::U*r,int depth,double gamma,const cm5::Tree&tree,double value,std::vector<double> bounds){
        if(depth!=3||data.min_leaf)return;for(auto&v:bounds)v=std::max(v,value);
        if(references.size()>=16)references.erase(references.begin()+1);
        int count=0;for(int w=0;w<data.W;++w)count+=int(__popcnt64(r[w]));double largest=*std::max_element(bounds.begin(),bounds.end());
        references.push_back({std::vector<cm5::U>(r,r+data.W),std::move(bounds),tree,value,gamma,depth,count,largest});
    }
    bool solve(const cm5::U*r,int depth,double gamma,double deadline,cm5::Tree&tree,double&value){
        if(cm5::now()>=deadline)return false;double start=cm5::now();auto g=geometry(r);prepare_seconds+=cm5::now()-start;int F=int(g.ids.size()),W=int(g.mask.size()),ml=std::max(1,data.min_leaf),N=std::accumulate(g.counts.begin(),g.counts.end(),0);
        int label=int(std::max_element(g.counts.begin(),g.counts.end())-g.counts.begin());value=N-g.counts[label];tree.a[1]=-1-label;if(!F||value<=gamma)return true;if(cm5::now()>=deadline)return false;
        std::vector<double> bounds(data.F,gamma);cm5::Tree incumbent=tree;double upper=value;std::vector<int> viable(F,1);
        if(depth==3&&!data.min_leaf){int best_ref=-1,min_removed=INT_MAX;
            for(int i=0;i<int(references.size());++i){const auto&ref=references[i];if(ref.gamma!=gamma||ref.depth!=depth||ref.largest_bound-std::max(0,ref.size-N)<=gamma)continue;int removed=0;for(int w=0;w<data.W;++w)removed+=int(__popcnt64(ref.rows[w]&~r[w]));
                for(int f=0;f<data.F;++f)bounds[f]=std::max(bounds[f],ref.roots[f]-removed-1e-7);
                if(removed<min_removed){min_removed=removed;best_ref=i;}}
            if(best_ref>=0){cm5::Tree candidate;double cost=restrict_tree(references[best_ref].tree,std::vector<cm5::U>(r,r+data.W),candidate,1,1,gamma);if(cost<upper){upper=cost;incumbent=candidate;}}
            for(int f=0;f<F;++f)viable[f]=bounds[g.ids[f]]<upper-1e-8;
            diagnostics[0]+=F;diagnostics[1]+=std::count(viable.begin(),viable.end(),0);
            if(std::none_of(viable.begin(),viable.end(),[](int v){return v!=0;})){++diagnostics[2];tree=incumbent;value=upper;remember(r,depth,gamma,tree,value,bounds);return true;}
        }
        // Any tree outside depth two contains at least three split nodes.
        // When a feasible incumbent costs at most 3*gamma, a D2 solve and
        // this split-count bound certify the full D3 problem.
        bool shallow_certificate=depth==3&&gamma>0&&upper<=3*gamma;
        int kernel_depth=shallow_certificate?2:depth;
        auto copy=[](CUdeviceptr p,const auto&v){if(!v.empty())cm5::MccGpu::check(cuMemcpyHtoD(p,v.data(),v.size()*sizeof(v[0])));};
        cuCtxSetCurrent(owner.context);start=cm5::now();copy(zero,g.zero);copy(mask,g.mask);copy(ends,g.end);copy(ids,g.ids);copy(counts,g.counts);copy(left,g.left);copy(eligible,viable);
        int block=W>64?128:64;CUfunction kernel=W>64?warp_kernel:cost_kernel;
        int streamed=size_t(W)*8+K*block*4+64*12>size_t(owner.shared_limit);int tasks=2*F*(F-1);
        bool histogram=triple_histogram&&F>=64&&W>=128&&(N-g.counts[label])>.15*N;
        if(histogram){
            int pairs=F*(F-1)/2,triples=F*(F-1)*(F-2)/6;
            void*pa[]={&zero,&ends,&F,&W,&pairs,&pair_histogram};cm5::MccGpu::check(cuLaunchKernel(pair_histogram_kernel,(pairs+7)/8,1,1,256,1,1,0,nullptr,pa,nullptr));
            if(kernel_depth==3)for(int first=0;first<triples;first+=262144){if(cm5::now()>=deadline){gpu_seconds+=cm5::now()-start;return false;}int count=std::min(262144,triples-first);void*ta[]={&zero,&ends,&F,&W,&first,&count,&triple_histogram,&eligible,&counts,&left,&pair_histogram};cm5::MccGpu::check(cuLaunchKernel(triple_histogram_kernel,(count+7)/8,1,1,256,1,1,0,nullptr,ta,nullptr));if(first+count<triples)cm5::MccGpu::check(cuCtxSynchronize());}
            void*ca[]={&F,&ml,&kernel_depth,&counts,&left,&ids,&pair_histogram,&triple_histogram,&eligible,&gamma,&costs,&actions};cm5::MccGpu::check(cuLaunchKernel(histogram_cost_kernel,pairs,1,1,64,1,1,0,nullptr,ca,nullptr));
        }else{
            for(int first=0;first<tasks;first+=16384){if(cm5::now()>=deadline){gpu_seconds+=cm5::now()-start;return false;}int count=std::min(16384,tasks-first);void*args[]={&zero,&mask,&ends,&ids,&F,&W,&ml,&kernel_depth,&gamma,&first,&count,&costs,&actions,&streamed,&counts,&left,&eligible};cm5::MccGpu::check(cuLaunchKernel(kernel,count,1,1,block,1,1,streamed?0:W*8,nullptr,args,nullptr));if(first+count<tasks)cm5::MccGpu::check(cuCtxSynchronize());}
        }
        void*sa[]={&F,&ml,&counts,&left,&costs,&gamma,&side_cost,&side_choice,&eligible};cm5::MccGpu::check(cuLaunchKernel(side_kernel,2*F,1,1,64,1,1,0,nullptr,sa,nullptr));
        void*ra[]={&F,&ml,&counts,&ids,&actions,&side_cost,&side_choice,&gamma,&value_out,&tree_out,&eligible,&root_values};cm5::MccGpu::check(cuLaunchKernel(root_kernel,1,1,1,64,1,1,0,nullptr,ra,nullptr));cm5::MccGpu::check(cuMemcpyDtoH(&value,value_out,8));cm5::MccGpu::check(cuMemcpyDtoH(tree.a.data(),tree_out,16*4));
        std::vector<double> conditional(F);cm5::MccGpu::check(cuMemcpyDtoH(conditional.data(),root_values,F*8));for(int f=0;f<F;++f)if(viable[f])bounds[g.ids[f]]=std::max(bounds[g.ids[f]],shallow_certificate?std::min(conditional[f],3*gamma):conditional[f]);
        if(upper<value){value=upper;tree=incumbent;}remember(r,depth,gamma,tree,value,bounds);gpu_seconds+=cm5::now()-start;return true;
    }
};
}
