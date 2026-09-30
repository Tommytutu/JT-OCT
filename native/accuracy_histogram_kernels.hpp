#pragma once
namespace accuracy {
// Joint binary counts are symmetric in feature order. A three-feature table
// is determined by its one-, two-, and three-way zero intersections. Count
// each unordered triple once, then reuse it for all three choices of parent
// pair and all four routed sides. No metric approximation is involved.
inline const char* histogram_source=R"CUDA(
__device__ __forceinline__ int pair_index(int f,int g){int a=min(f,g),b=max(f,g);return b*(b-1)/2+a;}
__device__ __forceinline__ int triple_index(int f,int g,int h){int a=min(f,min(g,h)),c=max(f,max(g,h)),b=f+g+h-a-c;return c*(c-1)*(c-2)/6+b*(b-1)/2+a;}
extern "C" __global__ void accuracy_pair_hist(const U*z,const int*end,int F,int W,int pairs,int*out){
 int i=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(i>=pairs)return;
 int f=0,g=0;if(!lane){g=(int)((1+sqrtf(1+8.f*i))*.5f);while(g*(g-1)/2>i)--g;while(g*(g+1)/2<=i)++g;f=i-g*(g-1)/2;}
 f=__shfl_sync(0xffffffff,f,0);g=__shfl_sync(0xffffffff,g,0);int start=0;
 for(int c=0;c<K;++c){int n=0;for(int w=start+lane;w<end[c];w+=32)n+=__popcll(z[f*W+w]&z[g*W+w]);start=end[c];for(int s=16;s;s/=2)n+=__shfl_down_sync(0xffffffff,n,s);if(!lane)out[i*K+c]=n;}
}
extern "C" __global__ void accuracy_triple_hist(const U*z,const int*end,int F,int W,int first,int count,int*out,const int*eligible,const int*counts,const int*left,const int*pairs){
 int local=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(local>=count)return;int i=first+local,f=0,g=0,h=0;
 if(!lane){h=(int)cbrtf(6.f*i)+1;while(h*(h-1)*(h-2)/6>i)--h;while((h+1)*h*(h-1)/6<=i)++h;int p=i-h*(h-1)*(h-2)/6;
   g=(int)((1+sqrtf(1+8.f*p))*.5f);while(g*(g-1)/2>p)--g;while(g*(g+1)/2<=p)++g;f=p-g*(g-1)/2;}
 f=__shfl_sync(0xffffffff,f,0);g=__shfl_sync(0xffffffff,g,0);h=__shfl_sync(0xffffffff,h,0);if(!eligible[f]&&!eligible[g]&&!eligible[h])return;
 int start=0;for(int c=0;c<K;++c){int a=left[c*F+f],b=left[c*F+g],d=left[c*F+h],ab=pairs[pair_index(f,g)*K+c],ad=pairs[pair_index(f,h)*K+c],bd=pairs[pair_index(g,h)*K+c];
   int lo=max(0,max(ab+ad-a,max(ab+bd-b,ad+bd-d))),hi=min(min(ab,ad),min(bd,counts[c]-a-b-d+ab+ad+bd));
   int n=lo;if(lo!=hi){n=0;for(int w=start+lane;w<end[c];w+=32)n+=__popcll(z[f*W+w]&z[g*W+w]&z[h*W+w]);for(int s=16;s;s/=2)n+=__shfl_down_sync(0xffffffff,n,s);}start=end[c];if(!lane)out[i*K+c]=n;}
}
extern "C" __global__ void accuracy_hist_cost(int F,int ml,int depth,const int*counts,const int*left,const int*ids,const int*pairs,const int*triples,const int*eligible,double gamma,double*out,int*actions){
 int p=blockIdx.x,t=threadIdx.x,g=(int)((1+sqrtf(1+8.f*p))*.5f);while(g*(g-1)/2>p)--g;while(g*(g+1)/2<=p)++g;int f=p-g*(g-1)/2;
 if(!eligible[f]&&!eligible[g])return;int total[4][K],N[4]={},majority[4]={},label[4]={};double best[4];int winner[4];
 for(int c=0;c<K;++c){int both=pairs[p*K+c],a=left[c*F+f],b=left[c*F+g];total[0][c]=both;total[1][c]=a-both;total[2][c]=b-both;total[3][c]=counts[c]-a-b+both;
   for(int q=0;q<4;++q){N[q]+=total[q][c];if(total[q][c]>majority[q]){majority[q]=total[q][c];label[q]=c;}}}
 for(int q=0;q<4;++q){best[q]=N[q]<ml?1e300:double(N[q]-majority[q]);winner[q]=-label[q]-1;}
 if(depth==3)for(int h=t;h<F;h+=64){if(h==f||h==g)continue;int ix=triple_index(f,g,h),fh=pair_index(f,h),gh=pair_index(g,h);
   int n0[4]={},m0[4]={},m1[4]={},l0[4]={},l1[4]={};
   for(int c=0;c<K;++c){int abc=triples[ix*K+c],ac=pairs[fh*K+c],bc=pairs[gh*K+c];int a[4]={abc,ac-abc,bc-abc,left[c*F+h]-ac-bc+abc};
     for(int q=0;q<4;++q){n0[q]+=a[q];if(a[q]>m0[q]){m0[q]=a[q];l0[q]=c;}if(total[q][c]-a[q]>m1[q]){m1[q]=total[q][c]-a[q];l1[q]=c;}}}
   for(int q=0;q<4;++q){if(n0[q]<ml||N[q]-n0[q]<ml)continue;double v=gamma+N[q]-m0[q]-m1[q];int a=ids[h]*K*K+l0[q]*K+l1[q];if(v<best[q]||(v==best[q]&&a<winner[q])){best[q]=v;winner[q]=a;}}}
 __shared__ double v[4*64];__shared__ int a[4*64];for(int q=0;q<4;++q){v[q*64+t]=best[q];a[q*64+t]=winner[q];}__syncthreads();
 for(int s=32;s;s/=2){if(t<s)for(int q=0;q<4;++q){int j=q*64+t;if(v[j+s]<v[j]||(v[j+s]==v[j]&&a[j+s]<a[j])){v[j]=v[j+s];a[j]=a[j+s];}}__syncthreads();}
 if(!t)for(int q=0;q<4;++q){int i=(f*F+g)*4+q,j=(g*F+f)*4+2*(q%2)+q/2;out[i]=out[j]=v[q*64];actions[i]=actions[j]=a[q*64];}
}
)CUDA";
}
