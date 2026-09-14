// Batched routed-state / feature statistics for native JT-CG pricing.
// Shared memory depends only on the thread block, never on n, F or K.
extern "C" {
__device__ int cg_sum(int value) {
    __shared__ int scratch[8];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int offset=16;offset;offset>>=1)value+=__shfl_down_sync(0xffffffff,value,offset);
    if(lane==0)scratch[warp]=value;
    __syncthreads();
    int answer=0;if(threadIdx.x==0)for(int j=0;j<8;++j)answer+=scratch[j];
    __syncthreads();return answer;
}
__global__ void cg_parent_counts(const unsigned long long* rows,
    const unsigned long long* classes,int W,int K,int* counts) {
    int s=blockIdx.x,k=blockIdx.y,value=0;
    for(int w=threadIdx.x;w<W;w+=256)
        value+=__popcll(rows[(long long)s*W+w]&classes[(long long)k*W+w]);
    value=cg_sum(value);if(threadIdx.x==0)counts[s*K+k]=value;
}
__global__ void cg_stump_costs(const unsigned long long* rows,
    const unsigned long long* features,const unsigned long long* classes,
    const int* counts,int W,int F,int K,int min_leaf,double unit,
    double* loss,int* labels) {
    int h=blockIdx.x,s=blockIdx.y;
    long long out=(long long)s*F+h;
    int left_total=0,right_total=0,best_left=-1,best_right=-1,label_left=0,label_right=0;
    for(int k=0;k<K;++k){
        int value=0;
        for(int w=threadIdx.x;w<W;w+=256){
            auto bits=rows[(long long)s*W+w]&classes[(long long)k*W+w];
            value+=__popcll(bits&~features[(long long)h*W+w]);
        }
        value=cg_sum(value);
        if(threadIdx.x==0){
            int right=counts[s*K+k]-value;left_total+=value;right_total+=right;
            if(value>best_left){best_left=value;label_left=k;}
            if(right>best_right){best_right=right;label_right=k;}
        }
    }
    if(threadIdx.x==0){
        loss[out]=(left_total<min_leaf||right_total<min_leaf)?__longlong_as_double(0x7ff0000000000000LL):
            (left_total+right_total-best_left-best_right)*unit;
        labels[2*out]=label_left;labels[2*out+1]=label_right;
    }
}
}
