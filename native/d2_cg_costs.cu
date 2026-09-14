// Exact D2 sufficient statistics. Shared storage is O(block size), not O(n).
// The triangular pair kernel scans each unordered feature pair only once.
extern "C" {
__device__ int d2_sum(int v) {
    __shared__ int scratch[8];
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    for(int offset=16;offset;offset>>=1) v+=__shfl_down_sync(0xffffffff,v,offset);
    if(lane==0) scratch[warp]=v;
    __syncthreads();int answer=0;
    if(threadIdx.x==0) for(int j=0;j<8;++j) answer+=scratch[j];
    __syncthreads();return answer;
}
__device__ double d2_inf(){return __longlong_as_double(0x7ff0000000000000LL);}

__global__ void d2_marginals(const unsigned long long* features,
    const unsigned long long* classes,int W,int F,int K,int* counts) {
    int f=blockIdx.x,k=blockIdx.y,v=0;
    for(int w=threadIdx.x;w<W;w+=256) {
        auto bits=classes[(long long)k*W+w];
        if(f<F) bits&=features[(long long)f*W+w];
        v+=__popcll(bits);
    }
    v=d2_sum(v);if(threadIdx.x==0) counts[f*K+k]=v;
}

__global__ void d2_pairs(const unsigned long long* features,
    const unsigned long long* classes,const int* counts,
    int W,int F,int K,int min_leaf,double unit,int root_start,int root_count,
    double* loss,int* labels) {
    int g=blockIdx.x,f=root_start+blockIdx.y;
    if(g<f) return;
    int totals[4]={0,0,0,0}, maxima[4]={-1,-1,-1,-1}, majority[4]={0,0,0,0};
    for(int k=0;k<K;++k) {
        int both=0;
        for(int w=threadIdx.x;w<W;w+=256)
            both+=__popcll(features[(long long)f*W+w]&features[(long long)g*W+w]&classes[(long long)k*W+w]);
        both=d2_sum(both);
        if(threadIdx.x==0) {
            int a=counts[f*K+k],b=counts[g*K+k],t=counts[F*K+k];
            int cell[4]={t-a-b+both,b-both,a-both,both};
            for(int j=0;j<4;++j) {
                totals[j]+=cell[j];
                if(cell[j]>maxima[j]) {maxima[j]=cell[j];majority[j]=k;}
            }
        }
    }
    if(threadIdx.x==0) for(int transpose=0;transpose<(f==g?1:2);++transpose) {
        int root=transpose?g:f,child=transpose?f:g;
        for(int side=0;side<2;++side) {
            int a=transpose?side:2*side,b=transpose?side+2:2*side+1;
            long long index=((long long)root*F+child)*2+side;
            loss[index]=(totals[a]<min_leaf||totals[b]<min_leaf)?d2_inf():
                (totals[a]+totals[b]-maxima[a]-maxima[b])*unit;
            labels[2*index]=majority[a];labels[2*index+1]=majority[b];
        }
    }
}

// Diagnostic reference: two routed branches are scanned independently, as in
// the old CPU local-cost schedule. Same constraints, loss and CG master.
__global__ void d2_generic(const unsigned long long* features,
    const unsigned long long* classes,const int* counts,
    int W,int F,int K,int min_leaf,double unit,int state_start,
    double* loss,int* labels) {
    int g=blockIdx.x,state=state_start+blockIdx.y,f=state/2,side=state%2;
    int nl=0,nr=0,ml=-1,mr=-1,kl=0,kr=0;
    for(int k=0;k<K;++k) {
        int left=0;
        for(int w=threadIdx.x;w<W;w+=256) {
            auto root=features[(long long)f*W+w];if(!side) root=~root;
            left+=__popcll(root&~features[(long long)g*W+w]&classes[(long long)k*W+w]);
        }
        left=d2_sum(left);
        if(threadIdx.x==0) {
            int total=side?counts[f*K+k]:counts[F*K+k]-counts[f*K+k],right=total-left;
            nl+=left;nr+=right;
            if(left>ml){ml=left;kl=k;}if(right>mr){mr=right;kr=k;}
        }
    }
    if(threadIdx.x==0) {
        long long index=((long long)f*F+g)*2+side;
        loss[index]=(nl<min_leaf||nr<min_leaf)?d2_inf():(nl+nr-ml-mr)*unit;
        labels[2*index]=kl;labels[2*index+1]=kr;
    }
}

__global__ void d2_reduce(const double* loss,const int* labels,const int* counts,
    const unsigned char* allowed,const double* extras,int F,int K,int min_leaf,
    int no_repeat,double unit,double* best,int* actions) {
    int f=blockIdx.x,side=blockIdx.y,tid=threadIdx.x;
    __shared__ double values[256];__shared__ int args[256];
    double val=d2_inf();int arg=-1;
    if(allowed[f]) for(int g=tid;g<F;g+=256) {
        if(!allowed[(side+1)*F+g]||(no_repeat&&f==g)) continue;
        double candidate=loss[((long long)f*F+g)*2+side]+extras[(side+1)*F+g];
        if(candidate<val||(candidate==val&&candidate<d2_inf()&&(arg<0||g<arg))){val=candidate;arg=g;}
    }
    values[tid]=val;args[tid]=arg;__syncthreads();
    for(int stride=128;stride;stride>>=1){
        if(tid<stride){
            double v=values[tid+stride];int a=args[tid+stride];
            if(v<values[tid]||(v==values[tid]&&a>=0&&(args[tid]<0||a<args[tid])))
                {values[tid]=v;args[tid]=a;}
        }__syncthreads();
    }
    if(tid==0){
        int state=2*f+side,total=0,maximum=-1,majority=0;
        for(int k=0;k<K;++k){
            int v=side?counts[f*K+k]:counts[F*K+k]-counts[f*K+k];total+=v;
            if(v>maximum){maximum=v;majority=k;}
        }
        best[2*state]=(allowed[f]&&total>=min_leaf)?(total-maximum)*unit:d2_inf();
        best[2*state+1]=values[0];
        actions[4*state]=majority;actions[4*state+1]=args[0];
        long long index=((long long)f*F+(args[0]<0?0:args[0]))*2+side;
        actions[4*state+2]=args[0]<0?0:labels[2*index];
        actions[4*state+3]=args[0]<0?0:labels[2*index+1];
    }
}
}
