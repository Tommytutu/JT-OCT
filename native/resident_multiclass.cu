// Class count is specialized by NVRTC, not truncated to a binary target.
// Features and routed sample words remain streaming; no F^3 tensor is stored.
#define CLASSES @CLASSES@
typedef unsigned long long U;
__device__ int warp_sum(int v) {
    for (int s=16;s;s>>=1) v+=__shfl_down_sync(0xffffffff,v,s);
    return v;
}
__device__ U routed(const U* z,const U* mask,int W,int w,int f,int g,int a,int b) {
    U r=mask[w],zf=z[(long long)f*W+w],zg=z[(long long)g*W+w];
    return r&(a?~zf:zf)&(b?~zg:zg);
}
extern "C" __global__ void resident_d3(
    const U* z,const U* labels,const U* masks,const int* indices,const int* nw,
    const int* feature,const double* costs,const unsigned char* allowed,
    const double* floors,const unsigned char* dominated,
    int F,int W,int repeat,int early,int ml,int bounds,int start,int count,
    double weight,double* output,int* tail,unsigned char* reasons) {
    int s=blockIdx.y,pair=start+blockIdx.x/4,q=blockIdx.x%4;
    int f=pair/F,g=pair%F,a=q/2,b=q%2,t=threadIdx.x,lane=t%32,warp=t/32;
    long long out=((long long)s*4*F+q*F+f)*F+g;
    const double INF=1e300;
    if (!t) { output[out]=INF;tail[out]=-2;reasons[out]=0; }
    costs+=(long long)s*7*F;allowed+=(long long)s*7*F;
    floors+=s*7;dominated+=(long long)s*2*F;feature+=s*F;
    if (!allowed[f]||!allowed[(1+a)*F+g]||(repeat&&f==g)) return;
    if (dominated[a*F+f]) { if (!t) reasons[out]=2;return; }
    const U* mask=masks+(long long)s*W;
    const int* idx=indices+(long long)s*W;
    int words=nw[s]<0?W:nw[s],ff=feature[f],gg=feature[g];
    __shared__ int hist[CLASSES],total,majority,acts[8];
    __shared__ double mins[8];
    for (int c=warp;c<CLASSES;c+=8) {
        int v=0;
        for (int k=lane;k<words;k+=32) {
            int w=nw[s]<0?k:idx[k];
            v+=__popcll(routed(z,mask,W,w,ff,gg,a,b)&labels[(long long)c*W+w]);
        }
        v=warp_sum(v);if (!lane) hist[c]=v;
    }
    __syncthreads();
    if (!t) { total=0;majority=0;
        for (int c=0;c<CLASSES;++c) { total+=hist[c];majority=max(majority,hist[c]); }
    }
    __syncthreads();
    if (total<ml) return;
    double stop=early?weight*(total-majority):INF;
    double base=.25*costs[f]+.5*costs[(1+a)*F+g];
    if (bounds&&early&&stop<=floors[3+q]) {
        if (!t) { output[out]=base+stop;tail[out]=-1;reasons[out]=3; } return;
    }
    double best=INF;int action=-2;
    for (int h=warp;h<F;h+=8) {
        if (!allowed[(3+q)*F+h]||(repeat&&(h==f||h==g))) continue;
        int n0=0,max0=0,max1=0,hh=feature[h];
        for (int c=0;c<CLASSES;++c) {
            int v=0;
            for (int k=lane;k<words;k+=32) {
                int w=nw[s]<0?k:idx[k];
                U left=routed(z,mask,W,w,ff,gg,a,b)&z[(long long)hh*W+w];
                v+=__popcll(left&labels[(long long)c*W+w]);
            }
            v=warp_sum(v);
            if (!lane) { n0+=v;max0=max(max0,v);max1=max(max1,hist[c]-v); }
        }
        if (!lane&&n0>=ml&&total-n0>=ml) {
            double v=costs[(3+q)*F+h]+weight*(total-max0-max1);
            if (v<best||(v==best&&(action<0||h<action))) { best=v;action=h; }
        }
    }
    if (!lane) { mins[warp]=best;acts[warp]=action; } __syncthreads();
    if (!t) {
        for (int j=1;j<8;++j)
            if (mins[j]<mins[0]||(mins[j]==mins[0]&&acts[j]>=0&&(acts[0]<0||acts[j]<acts[0]))) {
                mins[0]=mins[j];acts[0]=acts[j];
            }
        if (early&&stop<=mins[0]) { mins[0]=stop;acts[0]=-1; }
        output[out]=base+mins[0];tail[out]=acts[0];reasons[out]=1;
    }
}

extern "C" __global__ void resident_d2(
    const U* z,const U* labels,const U* masks,const int* indices,const int* nw,
    const int* feature,const double* costs,const unsigned char* allowed,
    int F,int W,int repeat,int ml,int start,int count,double weight,double* output) {
    int s=blockIdx.y,pair=start+blockIdx.x/2,a=blockIdx.x%2;
    int f=pair/F,g=pair%F,t=threadIdx.x,lane=t%32,warp=t/32;
    long long out=((long long)s*2*F+a*F+f)*F+g;
    if (!t) output[out]=1e300;
    costs+=(long long)s*7*F;allowed+=(long long)s*7*F;feature+=s*F;
    if (!allowed[f]||!allowed[(1+a)*F+g]||(repeat&&f==g)) return;
    const U* mask=masks+(long long)s*W;const int* idx=indices+(long long)s*W;
    int words=nw[s]<0?W:nw[s],ff=feature[f],gg=feature[g];
    __shared__ int left_counts[CLASSES],right_counts[CLASSES];
    for (int c=warp;c<CLASSES;c+=8) {
        int n0=0,n1=0;
        for (int k=lane;k<words;k+=32) {
            int w=nw[s]<0?k:idx[k];
            U r=mask[w]&(a?~z[(long long)ff*W+w]:z[(long long)ff*W+w]);
            r&=labels[(long long)c*W+w];
            U left=r&z[(long long)gg*W+w];
            n0+=__popcll(left);n1+=__popcll(r^left);
        }
        n0=warp_sum(n0);n1=warp_sum(n1);
        if (!lane) { left_counts[c]=n0;right_counts[c]=n1; }
    }
    __syncthreads();
    if (!t) {
        int n0=0,n1=0,max0=0,max1=0;
        for (int c=0;c<CLASSES;++c) {
            n0+=left_counts[c];n1+=right_counts[c];
            max0=max(max0,left_counts[c]);max1=max(max1,right_counts[c]);
        }
        if (n0>=ml&&n1>=ml) output[out]=.5*costs[f]+costs[(1+a)*F+g]+weight*(n0+n1-max0-max1);
    }
}
