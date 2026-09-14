// Complete GPU tables: build routed masks and reduce h on-device.
extern "C" {
__global__ void cg_direct_rows(const int* ids,const unsigned long long* features,
    int n,int W,int F,int states,unsigned long long* rows) {
    long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=(long long)states*W)return;
    int s=j/W,w=j%W,id=ids[s],q=id/(F*F),pair=id%(F*F),f=pair/F,g=pair%F;
    unsigned long long valid=(w==W-1&&n%64)?((1ULL<<(n%64))-1):~0ULL;
    auto a=features[(long long)f*W+w],b=features[(long long)g*W+w];
    rows[j]=((q/2)?a:~a)&((q%2)?b:~b)&valid;
}
__device__ void cg_better(double& value,int& id,double other,int other_id) {
    if(other<value||(other==value&&other_id<id)){value=other;id=other_id;}
}
__global__ void cg_direct_min(const int* ids,const double* losses,const int* labels,
    const unsigned char* allowed,const double* extras,int F,int no_repeat,
    double* best_cost,int* best_action) {
    int s=blockIdx.x,id=ids[s],q=id/(F*F),pair=id%(F*F),f=pair/F,g=pair%F;
    int best=2147483647;double value=__longlong_as_double(0x7ff0000000000000LL);
    for(int h=threadIdx.x;h<F;h+=256){
        if(!allowed[(3+q)*F+h]||(no_repeat&&(h==f||h==g)))continue;
        double v=losses[(long long)s*F+h]+extras[(3+q)*F+h];
        if(isfinite(v))cg_better(value,best,v,h);
    }
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int d=16;d;d>>=1){double v=__shfl_down_sync(0xffffffff,value,d);
        int h=__shfl_down_sync(0xffffffff,best,d);cg_better(value,best,v,h);}
    __shared__ double values[8];__shared__ int actions[8];
    if(lane==0){values[warp]=value;actions[warp]=best;}__syncthreads();
    if(warp==0){
        value=lane<8?values[lane]:__longlong_as_double(0x7ff0000000000000LL);
        best=lane<8?actions[lane]:2147483647;
        for(int d=16;d;d>>=1){double v=__shfl_down_sync(0xffffffff,value,d);
            int h=__shfl_down_sync(0xffffffff,best,d);cg_better(value,best,v,h);}
        if(lane==0){best_cost[s]=value;int* out=best_action+3*s;
            out[0]=best==2147483647?-1:best;
            out[1]=best==2147483647?0:labels[2*((long long)s*F+best)];
            out[2]=best==2147483647?0:labels[2*((long long)s*F+best)+1];}
    }
}
}
