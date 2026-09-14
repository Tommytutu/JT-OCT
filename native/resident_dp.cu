// Streaming min-sum messages over an immutable global feature-bitset dictionary.
// grid.y batches independent routed states. A warp cooperates over sample words;
// h is strided over warps, so there is no 256-feature limit or F^3 allocation.
typedef unsigned long long U;
__device__ int warp_sum(int v) {
    for(int s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
    return v;
}
__device__ U routed(const U* z,const U* mask,int W,int w,int f,int g,int a,int b) {
    U r=mask[w],zf=z[(long long)f*W+w],zg=z[(long long)g*W+w];
    return r&(a?~zf:zf)&(b?~zg:zg);
}
extern "C" __global__ void resident_d3(
    const U* z,const U* positive,const U* masks,const int* indices,const int* nw,
    const int* feature,const double* costs,const unsigned char* allowed,
    const double* floors,const unsigned char* dominated,
    int F,int W,int repeat,int early,int ml,int bounds,int start,int count,
    double weight,double* output,int* tail,unsigned char* reasons) {
    int s=blockIdx.y,pair=start+blockIdx.x/4,q=blockIdx.x%4;
    int f=pair/F,g=pair%F,a=q/2,b=q%2,t=threadIdx.x,lane=t%32,warp=t/32;
    long long out=((long long)s*4*F+q*F+f)*F+g;
    const double INF=1e300;
    if(!t){output[out]=INF;tail[out]=-2;reasons[out]=0;}
    costs+=(long long)s*7*F;allowed+=(long long)s*7*F;
    floors+=s*7;dominated+=(long long)s*2*F;feature+=s*F;
    if(!allowed[f]||!allowed[(1+a)*F+g]||(repeat&&f==g))return;
    if(dominated[a*F+f]){if(!t)reasons[out]=2;return;}
    const U* mask=masks+(long long)s*W;
    const int* idx=indices+(long long)s*W;
    int words=nw[s]<0?W:nw[s],ff=feature[f],gg=feature[g];
    __shared__ int sums[8],ys[8],acts[8];
    __shared__ double mins[8];
    int n=0,y=0;
    for(int k=t;k<words;k+=256){int w=nw[s]<0?k:idx[k];
        U r=routed(z,mask,W,w,ff,gg,a,b);n+=__popcll(r);y+=__popcll(r&positive[w]);}
    n=warp_sum(n);y=warp_sum(y);
    if(!lane){sums[warp]=n;ys[warp]=y;}__syncthreads();
    if(!t){for(int j=1;j<8;++j){sums[0]+=sums[j];ys[0]+=ys[j];}}__syncthreads();
    int total=sums[0],pos=ys[0];if(total<ml)return;
    double stop=early?weight*min(pos,total-pos):INF;
    double base=.25*costs[f]+.5*costs[(1+a)*F+g];
    if(bounds&&early&&stop<=floors[3+q]){
        if(!t){output[out]=base+stop;tail[out]=-1;reasons[out]=3;}return;
    }
    double best=INF;int action=-2;
    for(int h=warp;h<F;h+=8){
        if(!allowed[(3+q)*F+h]||(repeat&&(h==f||h==g)))continue;
        int n0=0,y0=0,hh=feature[h];
        for(int k=lane;k<words;k+=32){int w=nw[s]<0?k:idx[k];
            U left=routed(z,mask,W,w,ff,gg,a,b)&z[(long long)hh*W+w];
            n0+=__popcll(left);y0+=__popcll(left&positive[w]);}
        n0=warp_sum(n0);y0=warp_sum(y0);
        if(!lane&&n0>=ml&&total-n0>=ml){
            double v=costs[(3+q)*F+h]+weight*(min(y0,n0-y0)+min(pos-y0,total-n0-pos+y0));
            if(v<best||(v==best&&(action<0||h<action))){best=v;action=h;}
        }
    }
    if(!lane){mins[warp]=best;acts[warp]=action;}__syncthreads();
    if(!t){for(int j=1;j<8;++j){if(mins[j]<mins[0]||(mins[j]==mins[0]&&acts[j]>=0&&(acts[0]<0||acts[j]<acts[0]))){mins[0]=mins[j];acts[0]=acts[j];}}
        if(early&&stop<=mins[0]){mins[0]=stop;acts[0]=-1;}
        output[out]=base+mins[0];tail[out]=acts[0];reasons[out]=1;
    }
}

// Dedicated D2 messages: each block computes a complete (root f, child g, side)
// stump using a word-parallel reduction. No third feature or D3 state is visited.
extern "C" __global__ void resident_d2(
    const U* z,const U* positive,const U* masks,const int* indices,const int* nw,
    const int* feature,const double* costs,const unsigned char* allowed,
    int F,int W,int repeat,int ml,int start,int count,double weight,double* output) {
    int s=blockIdx.y,pair=start+blockIdx.x/2,a=blockIdx.x%2;
    int f=pair/F,g=pair%F,t=threadIdx.x,lane=t%32,warp=t/32;
    long long out=((long long)s*2*F+a*F+f)*F+g;
    if(!t)output[out]=1e300;
    costs+=(long long)s*7*F;allowed+=(long long)s*7*F;feature+=s*F;
    if(!allowed[f]||!allowed[(1+a)*F+g]||(repeat&&f==g))return;
    const U* mask=masks+(long long)s*W;const int* idx=indices+(long long)s*W;
    int words=nw[s]<0?W:nw[s],ff=feature[f],gg=feature[g];
    int n=0,y=0,n0=0,y0=0;
    for(int k=t;k<words;k+=256){int w=nw[s]<0?k:idx[k];
        U r=mask[w]&(a?~z[(long long)ff*W+w]:z[(long long)ff*W+w]);
        U left=r&z[(long long)gg*W+w];
        n+=__popcll(r);y+=__popcll(r&positive[w]);n0+=__popcll(left);y0+=__popcll(left&positive[w]);}
    n=warp_sum(n);y=warp_sum(y);n0=warp_sum(n0);y0=warp_sum(y0);
    __shared__ int sums[4][8];
    if(!lane){sums[0][warp]=n;sums[1][warp]=y;sums[2][warp]=n0;sums[3][warp]=y0;}__syncthreads();
    if(!t){for(int k=0;k<4;++k)for(int j=1;j<8;++j)sums[k][0]+=sums[k][j];
        n=sums[0][0];y=sums[1][0];n0=sums[2][0];y0=sums[3][0];
        if(n0>=ml&&n-n0>=ml)output[out]=.5*costs[f]+costs[(1+a)*F+g]+weight*(min(y0,n0-y0)+min(y-y0,n-n0-y+y0));
    }
}
