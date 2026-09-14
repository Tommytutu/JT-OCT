// Appended to the independent baseline source. K is the true class count.
// Route once per word, fuse K-1 class counts, derive the final count exactly.
#define FUSED_K @CLASSES@
extern "C" __global__ void resident_d3_fused(
    const U* z,const U* labels,const U* masks,const int* indices,const int* nw,
    const int* feature,const double* costs,const unsigned char* allowed,
    const double* floors,const unsigned char* dominated,
    int F,int W,int repeat,int early,int ml,int bounds,int start,int count,
    double weight,double* output,int* tail,unsigned char* reasons,int cache_pair) {
    int s=blockIdx.y,pair=start+blockIdx.x/4,q=blockIdx.x%4;
    int f=pair/F,g=pair%F,a=q/2,b=q%2,t=threadIdx.x,lane=t%32,warp=t/32;
    long long out=((long long)s*4*F+q*F+f)*F+g;
    const double inf=1e300;
    if(!t){output[out]=inf;tail[out]=-2;reasons[out]=0;}
    costs+=(long long)s*7*F;allowed+=(long long)s*7*F;
    floors+=s*7;dominated+=(long long)s*2*F;feature+=s*F;
    if(!allowed[f]||!allowed[(1+a)*F+g]||(repeat&&f==g))return;
    if(dominated[a*F+f]){if(!t)reasons[out]=2;return;}
    const U* mask=masks+(long long)s*W;
    const int* idx=indices+(long long)s*W;
    int words=nw[s]<0?W:nw[s],ff=feature[f],gg=feature[g];
    extern __shared__ U pair_rows[];
    __shared__ int sums[8],hist[FUSED_K],parts[FUSED_K][8],acts[8];
    __shared__ double mins[8];
    int n=0,cs[FUSED_K]={0};
    for(int k=t;k<words;k+=256){
        int w=nw[s]<0?k:idx[k];U r=routed(z,mask,W,w,ff,gg,a,b);
        if(cache_pair)pair_rows[k]=r;
        n+=__popcll(r);
        #pragma unroll
        for(int c=0;c<FUSED_K-1;++c)cs[c]+=__popcll(r&labels[(long long)c*W+w]);
    }
    n=warp_sum(n);if(!lane)sums[warp]=n;
    #pragma unroll
    for(int c=0;c<FUSED_K-1;++c){cs[c]=warp_sum(cs[c]);if(!lane)parts[c][warp]=cs[c];}
    __syncthreads();
    if(!t){
        for(int j=1;j<8;++j)sums[0]+=sums[j];
        int taken=0;
        for(int c=0;c<FUSED_K-1;++c){hist[c]=0;for(int j=0;j<8;++j)hist[c]+=parts[c][j];taken+=hist[c];}
        hist[FUSED_K-1]=sums[0]-taken;
    }
    __syncthreads();
    int total=sums[0];if(total<ml)return;
    int majority=0;for(int c=0;c<FUSED_K;++c)majority=max(majority,hist[c]);
    double stop=early?weight*(total-majority):inf;
    double base=.25*costs[f]+.5*costs[(1+a)*F+g];
    if(bounds&&early&&stop<=floors[3+q]){
        if(!t){output[out]=base+stop;tail[out]=-1;reasons[out]=3;}return;
    }
    double best=inf;int action=-2;
    for(int h=warp;h<F;h+=8){
        if(!allowed[(3+q)*F+h]||(repeat&&(h==f||h==g)))continue;
        int n0=0,left_counts[FUSED_K]={0},hh=feature[h];
        for(int k=lane;k<words;k+=32){
            int w=nw[s]<0?k:idx[k];
            U r=cache_pair?pair_rows[k]:routed(z,mask,W,w,ff,gg,a,b);
            if(!r)continue;
            U left=r&z[(long long)hh*W+w];n0+=__popcll(left);
            #pragma unroll
            for(int c=0;c<FUSED_K-1;++c)left_counts[c]+=__popcll(left&labels[(long long)c*W+w]);
        }
        n0=warp_sum(n0);int taken=0,max0=0,max1=0;
        #pragma unroll
        for(int c=0;c<FUSED_K-1;++c){
            int v=warp_sum(left_counts[c]);
            if(!lane){taken+=v;max0=max(max0,v);max1=max(max1,hist[c]-v);}
        }
        if(!lane&&n0>=ml&&total-n0>=ml){
            int last=n0-taken;max0=max(max0,last);max1=max(max1,hist[FUSED_K-1]-last);
            double v=costs[(3+q)*F+h]+weight*(total-max0-max1);
            if(v<best||(v==best&&(action<0||h<action))){best=v;action=h;}
        }
    }
    if(!lane){mins[warp]=best;acts[warp]=action;}__syncthreads();
    if(!t){
        for(int j=1;j<8;++j)if(mins[j]<mins[0]||(mins[j]==mins[0]&&acts[j]>=0&&(acts[0]<0||acts[j]<acts[0]))){mins[0]=mins[j];acts[0]=acts[j];}
        if(early&&stop<=mins[0]){mins[0]=stop;acts[0]=-1;}
        output[out]=base+mins[0];tail[out]=acts[0];reasons[out]=1;
    }
}
