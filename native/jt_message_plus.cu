typedef unsigned long long U;
__device__ U mix_mp(U x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;
    x*=0x94d049bb133111ebULL;return x^(x>>31);}
#define ADD(k,v) do {if(tid==0)atomicAdd(stats+(k),(U)(v));} while(0)
extern "C" __global__ void jt_mp_plus_gpu(
    const U* masks,const U* positive,const int* conflict,const double* incoming,
    int F,int W,int D,int root,int q,int repeat,int ml,int start,int count,int stride,
    int bound,int sparse,int prune,int nc,double weight,double penalty,double cutoff,
    double* output,int* tail,U* stats,volatile U* keys,int* locks,volatile int* valid,volatile double* cache_values,
    volatile int* cache_actions,int slots) {
    const int local=blockIdx.x,tid=threadIdx.x,index=start+local;
    if(local>=count)return;
    const double INF=1e300,previous=incoming[index/stride];
    if(!tid){output[local]=INF;tail[local]=-2;}
    if(previous>=INF/2){ADD(14,1);return;}
    int fs[5]={root,0,0,0,0},code=index,stop=D-1;
    for(int j=D-2;j>=1;--j){fs[j]=code%(F+1);code/=F+1;}
    bool feasible=true;
    for(int j=1;j<D-1;++j){
        if(fs[j]==F){if(stop==D-1)stop=j;}
        else {if(stop!=D-1)feasible=false;
            if(repeat)for(int k=0;k<j;++k)if(fs[k]==fs[j])feasible=false;}}
    if(!feasible){ADD(15,1);return;}
    ADD(0,1);
    double allocated=0;
    for(int j=0;j<stop;++j)allocated+=penalty/(1<<(D-1-j));
    if(prune && previous+allocated>cutoff+1e-12){ADD(11,1);return;}
    extern __shared__ U routed[];
    int* nz_indices=(int*)(routed+W);
    __shared__ int totals[256],positives[256],actions[256],nz,held,slot,hit,mismatch;
    __shared__ double minima[256];
    __shared__ U hashes[256];
    if(!tid)nz=0;
    __syncthreads();
    int total=0,pos=0;
    for(int word=tid;word<W;word+=256){
        U set=~U(0);
        for(int j=0;j<stop;++j)set&=masks[(((q>>(D-2-j))&1)*F+fs[j])*W+word];
        routed[word]=set;
        total+=__popcll(set);pos+=__popcll(set&positive[word]);
        if(set){const int k=atomicAdd(&nz,1);nz_indices[k]=word;}}
    totals[tid]=total;positives[tid]=pos;
    __syncthreads();
    for(int s=128;s;s>>=1){if(tid<s){totals[tid]+=totals[tid+s];positives[tid]+=positives[tid+s];}__syncthreads();}
    total=totals[0];pos=positives[0];
    ADD(12,nz);ADD(13,W);
    if(total<ml)return;
    const int error=pos<total-pos?pos:total-pos;
    if(stop!=D-1){
        const double val=previous+allocated+weight*error/(1<<(D-1-stop));
        if(prune && val>cutoff+1e-12){ADD(11,1);return;}
        if(!tid){output[local]=val;tail[local]=-3;}return;}
    ADD(1,1);
    int unavoidable=0;
    if(bound>=2)for(int g=0;g<nc;++g){const int row=conflict[2*g];
        if(routed[row/64]&(U(1)<<(row%64)))unavoidable+=conflict[2*g+1];}
    const double stop_loss=weight*error;
    const double candidate_lb=penalty+weight*unavoidable;
    const double lower=stop_loss<candidate_lb?stop_loss:candidate_lb;
    if(prune && previous+allocated+lower>cutoff+1e-12){ADD(11,1);return;}
    int reason=-1;
    if(!error)reason=2;
    else if(bound>=1 && stop_loss<=penalty)reason=3;
    else if(bound>=2 && stop_loss<=candidate_lb)reason=4;
    if(reason>=0){ADD(reason,1);if(!tid){output[local]=previous+allocated+stop_loss;tail[local]=-1;}return;}
    if(!tid){held=0;hit=0;slot=0;mismatch=0;}
    __syncthreads();
    if(slots){
        U hash=0;
        for(int word=tid;word<W;word+=256)if(routed[word])hash^=mix_mp(routed[word]^(0x9e3779b97f4a7c15ULL*(word+1)));
        hashes[tid]=hash;__syncthreads();
        for(int s=128;s;s>>=1){if(tid<s)hashes[tid]^=hashes[tid+s];__syncthreads();}
        if(!tid){slot=hashes[0]%slots;held=atomicCAS(locks+slot,0,1)==0;__threadfence();}
        __syncthreads();
        ADD(5,1);
        if(!held){ADD(7,1);}
        if(held && valid[slot]){
            for(int word=tid;word<W;word+=256)if(keys[(U)slot*W+word]!=routed[word])atomicOr(&mismatch,1);
            __syncthreads();if(!tid)hit=!mismatch;__syncthreads();
        }
        if(hit){ADD(6,1);if(!tid){output[local]=previous+allocated+cache_values[slot];tail[local]=cache_actions[slot];
                atomicExch(locks+slot,0);}return;}
    }
    const int words=sparse?nz:W;
    ADD(9,F-(repeat?D-1:0));ADD(10,(U)(F-(repeat?D-1:0))*words);
    double best=INF;int action=-2;
    for(int h=tid;h<F;h+=256){
        bool used=false;if(repeat)for(int j=0;j<D-1;++j)if(fs[j]==h)used=true;
        if(used)continue;
        int n0=0,y0=0;
        for(int k=0;k<words;++k){const int word=sparse?nz_indices[k]:k;
            const U left=routed[word]&masks[h*W+word];n0+=__popcll(left);y0+=__popcll(left&positive[word]);}
        const int n1=total-n0,y1=pos-y0;
        if(n0<ml||n1<ml)continue;
        const int e0=y0<n0-y0?y0:n0-y0,e1=y1<n1-y1?y1:n1-y1;
        const double value=penalty+weight*(e0+e1);
        if(value<best || (value==best && (action<0 || h<action))){best=value;action=h;}}
    minima[tid]=best;actions[tid]=action;__syncthreads();
    for(int s=128;s;s>>=1){if(tid<s){const double other=minima[tid+s];const int a=actions[tid+s];
        if(other<minima[tid] || (other==minima[tid] && a>=0 && (actions[tid]<0 || a<actions[tid]))){
            minima[tid]=other;actions[tid]=a;}}__syncthreads();}
    if(!tid){if(stop_loss<=minima[0]){minima[0]=stop_loss;actions[0]=-1;}
        output[local]=previous+allocated+minima[0];tail[local]=actions[0];}
    __syncthreads();
    if(slots && held){
        for(int word=tid;word<W;word+=256)keys[(U)slot*W+word]=routed[word];
        __threadfence();__syncthreads();
        if(!tid){cache_values[slot]=minima[0];cache_actions[slot]=actions[0];valid[slot]=1;
            __threadfence();atomicExch(locks+slot,0);atomicAdd(stats+8,U(1));}}
}
