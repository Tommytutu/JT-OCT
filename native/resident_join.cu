// One block per state. Warp-coalesced g reductions, then a stable f reduction.
// Match the original STOP <= split rule and smallest rule-index tie breaks.
extern "C" __global__ void resident_join(const double* values,const int* tails,
 const double* stops,int F,int depth,double* answer){
 int s=blockIdx.x,t=threadIdx.x,lane=t%32,warp=t/32;
 int blocks=depth==3?4:2;
 const double* v=values+(long long)s*blocks*F*F;
 const double* stop=stops+(long long)s*2*F;
 __shared__ double bests[8];__shared__ int fs[8],gs0[8],gs1[8];
 double best=__longlong_as_double(0x7ff0000000000000ULL);int bf=2147483647,bg0=-1,bg1=-1;
 for(int f=warp;f<F;f+=8){
  double side[2];int chosen[2];
  for(int a=0;a<2;a++){
   double value=__longlong_as_double(0x7ff0000000000000ULL);int gi=2147483647;
   for(int g=lane;g<F;g+=32){
    double z=depth==3?v[((long long)(2*a)*F+f)*F+g]+v[((long long)(2*a+1)*F+f)*F+g]:v[((long long)a*F+f)*F+g];
    if(z<value||(z==value&&g<gi)){value=z;gi=g;}
   }
   for(int offset=16;offset;offset/=2){
    double z=__shfl_down_sync(0xffffffff,value,offset);int g=__shfl_down_sync(0xffffffff,gi,offset);
    if(z<value||(z==value&&g<gi)){value=z;gi=g;}
   }
   side[a]=stop[a*F+f]<=value?stop[a*F+f]:value;
   chosen[a]=stop[a*F+f]<=value?-1:gi;
  }
  if(!lane){double z=side[0]+side[1];if(z<best||(z==best&&f<bf)){best=z;bf=f;bg0=chosen[0];bg1=chosen[1];}}
 }
 if(!lane){bests[warp]=best;fs[warp]=bf;gs0[warp]=bg0;gs1[warp]=bg1;}
 __syncthreads();
 if(!t){
  int win=0;for(int j=1;j<8;j++)if(bests[j]<bests[win]||(bests[j]==bests[win]&&fs[j]<fs[win]))win=j;
  double* out=answer+(long long)s*8;out[0]=bests[win];out[1]=fs[win];out[2]=gs0[win];out[3]=gs1[win];
  for(int q=0;q<4;q++){out[4+q]=-1;if(depth==3){int g=q<2?gs0[win]:gs1[win];if(g<0)g=0;
   out[4+q]=tails[((long long)s*4*F+q*F+fs[win])*F+g];}}
 }
}
