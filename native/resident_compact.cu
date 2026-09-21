// Gather every routed sample from the resident dictionary into dense local words.
extern "C" __global__ void compact_rows(const unsigned long long* zero,
 const unsigned long long* labels,const int* ids,const int* features,
 int globalW,int W,int F,int K,unsigned long long* outzero,unsigned long long* outlabels){
 int w=blockIdx.x,plane=blockIdx.y,s=blockIdx.z,lane=threadIdx.x;
 const unsigned long long* source=plane<F?zero+(long long)features[s*F+plane]*globalW:labels+(long long)(plane-F)*globalW;
 int a=ids[(long long)s*W*64+w*64+lane],b=ids[(long long)s*W*64+w*64+lane+32];
 bool va=a>=0&&((source[a/64]>>(a%64))&1ULL);
 bool vb=b>=0&&((source[b/64]>>(b%64))&1ULL);
 unsigned low=__ballot_sync(0xffffffff,va),high=__ballot_sync(0xffffffff,vb);
 if(lane==0){unsigned long long* target=plane<F?outzero+((long long)s*F+plane)*W:outlabels+((long long)s*K+plane-F)*W;
  target[w]=(unsigned long long)low|((unsigned long long)high<<32);}
}
