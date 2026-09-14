typedef unsigned long long U;
// One warp packs 64 routed samples. Data stay feature-major on the device.
extern "C" __global__ void d3_opt_pack(const unsigned char* X, const unsigned char* y,
    const int* ids, const int* features, int N, int n, int F, int W, U* zero, U* positive) {
    const int lane=threadIdx.x%32, item=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(item>=F*W)return;
    const int f=item/W,w=item%W,i0=w*64+lane,i1=i0+32;
    const int r0=i0<n?ids[i0]:0,r1=i1<n?ids[i1]:0;
    const unsigned lo=__ballot_sync(0xffffffff,i0<n && X[features[f]*N+r0]==0);
    const unsigned hi=__ballot_sync(0xffffffff,i1<n && X[features[f]*N+r1]==0);
    if(!lane)zero[item]=U(lo)|(U(hi)<<32);
    if(f==0){
        const unsigned yl=__ballot_sync(0xffffffff,i0<n && y[r0]);
        const unsigned yh=__ballot_sync(0xffffffff,i1<n && y[r1]);
        if(!lane)positive[w]=U(yl)|(U(yh)<<32);
    }
}

extern "C" __global__ void d3_opt_costs(const U* zero,const U* positive,
    const double* costs,const unsigned char* allowed,const double* floors,
    const unsigned char* dominated,int F,int W,int n,int repeat,int early,int ml,
    int bounds,int sparse,int start,int count,double weight,
    double* output,int* tail,unsigned char* reasons) {
    const int pair=start+blockIdx.x/4,q=blockIdx.x%4,tid=threadIdx.x;
    if(pair>=start+count)return;
    const int f=pair/F,g=pair%F,a=q/2,b=q%2,out=(q*F+f)*F+g;
    const double INF=1e300;
    if(!tid){output[out]=INF;tail[out]=-2;reasons[out]=0;}
    if(!allowed[f] || !allowed[(1+a)*F+g] || (repeat && f==g))return;
    if(dominated[a*F+f]){if(!tid)reasons[out]=2;return;}
    extern __shared__ U rows[];
    int* nz_indices=(int*)(rows+W);
    __shared__ int totals[256],positives[256],actions[256],nz;
    __shared__ double minima[256];
    if(!tid)nz=0;__syncthreads();
    const U last=n%64 ? (U(1)<<(n%64))-1:~U(0);
    int total=0,pos=0;
    for(int w=tid;w<W;w+=256){
        const U valid=w==W-1?last:~U(0);
        const U rf=a?valid^zero[f*W+w]:zero[f*W+w];
        const U rg=b?valid^zero[g*W+w]:zero[g*W+w];
        const U r=rf&rg;rows[w]=r;total+=__popcll(r);pos+=__popcll(r&positive[w]);
        if(sparse && r){int k=atomicAdd(&nz,1);nz_indices[k]=w;}
    }
    totals[tid]=total;positives[tid]=pos;__syncthreads();
    for(int s=128;s;s>>=1){if(tid<s){totals[tid]+=totals[tid+s];positives[tid]+=positives[tid+s];}__syncthreads();}
    total=totals[0];pos=positives[0];
    if(total<ml)return;
    const double base=0.25*costs[f]+0.5*costs[(1+a)*F+g];
    const double stop=early ? weight*(pos<total-pos?pos:total-pos):INF;
    if(bounds && early && stop<=floors[3+q]){
        if(!tid){output[out]=base+stop;tail[out]=-1;reasons[out]=3;}return;
    }
    if(!tid)reasons[out]=1;
    const int words=sparse?nz:W;
    double best=INF;int action=-2;
    for(int h=tid;h<F;h+=256){
        if(!allowed[(3+q)*F+h] || (repeat && (h==f || h==g)))continue;
        int n0=0,y0=0;
        for(int k=0;k<words;++k){const int w=sparse?nz_indices[k]:k;
            const U left=rows[w]&zero[h*W+w];n0+=__popcll(left);y0+=__popcll(left&positive[w]);}
        const int n1=total-n0,y1=pos-y0;
        if(n0<ml || n1<ml)continue;
        const double value=costs[(3+q)*F+h]+weight*((y0<n0-y0?y0:n0-y0)+(y1<n1-y1?y1:n1-y1));
        if(value<best || (value==best && (action<0 || h<action))){best=value;action=h;}
    }
    minima[tid]=best;actions[tid]=action;__syncthreads();
    for(int s=128;s;s>>=1){if(tid<s){const double v=minima[tid+s];const int h=actions[tid+s];
        if(v<minima[tid] || (v==minima[tid] && h>=0 && (actions[tid]<0 || h<actions[tid]))){minima[tid]=v;actions[tid]=h;}}
        __syncthreads();}
    if(!tid){if(early && stop<=minima[0]){minima[0]=stop;actions[0]=-1;}
        output[out]=base+minima[0];tail[out]=actions[0];}
}

// Exact large-row variant.  The original kernel keeps the complete routed
// (f,g,q) bitset and its nonzero-word indices in one block's shared memory.
// Here the sample words are streamed through a fixed-size shared tile, so the
// shared-memory requirement is independent of the total number of rows.
// One thread owns one h; this path therefore supports F <= blockDim.x.
extern "C" __global__ void d3_opt_costs_word_tiled(const U* zero,const U* positive,
    const double* costs,const unsigned char* allowed,const double* floors,
    const unsigned char* dominated,int F,int W,int n,int repeat,int early,int ml,
    int bounds,int sparse,int start,int count,int tile_words,double weight,
    double* output,int* tail,unsigned char* reasons) {
    const int pair=start+blockIdx.x/4,q=blockIdx.x%4,tid=threadIdx.x;
    if(pair>=start+count)return;
    const int f=pair/F,g=pair%F,a=q/2,b=q%2,out=(q*F+f)*F+g;
    const double INF=1e300;
    if(!tid){output[out]=INF;tail[out]=-2;reasons[out]=0;}
    if(!allowed[f] || !allowed[(1+a)*F+g] || (repeat && f==g))return;
    if(dominated[a*F+f]){if(!tid)reasons[out]=2;return;}

    extern __shared__ U rows_tile[];
    __shared__ int totals[256],positives[256],actions[256];
    __shared__ double minima[256];
    const U last=n%64 ? (U(1)<<(n%64))-1:~U(0);

    int local_total=0,local_pos=0;
    for(int w=tid;w<W;w+=blockDim.x){
        const U valid=w==W-1?last:~U(0);
        const U rf=a?valid^zero[f*W+w]:zero[f*W+w];
        const U rg=b?valid^zero[g*W+w]:zero[g*W+w];
        const U r=rf&rg;
        local_total+=__popcll(r);local_pos+=__popcll(r&positive[w]);
    }
    totals[tid]=local_total;positives[tid]=local_pos;__syncthreads();
    for(int s=128;s;s>>=1){
        if(tid<s){totals[tid]+=totals[tid+s];positives[tid]+=positives[tid+s];}
        __syncthreads();
    }
    const int total=totals[0],pos=positives[0];
    if(total<ml)return;
    const double base=0.25*costs[f]+0.5*costs[(1+a)*F+g];
    const double stop=early ? weight*(pos<total-pos?pos:total-pos):INF;
    if(bounds && early && stop<=floors[3+q]){
        if(!tid){output[out]=base+stop;tail[out]=-1;reasons[out]=3;}return;
    }
    if(!tid)reasons[out]=1;

    const int h=tid;
    const bool valid_h=h<F && allowed[(3+q)*F+h] && !(repeat && (h==f || h==g));
    int n0=0,y0=0;
    for(int base_word=0;base_word<W;base_word+=tile_words){
        const int words=(base_word+tile_words<W)?tile_words:W-base_word;
        for(int k=tid;k<words;k+=blockDim.x){
            const int w=base_word+k;
            const U valid=w==W-1?last:~U(0);
            const U rf=a?valid^zero[f*W+w]:zero[f*W+w];
            const U rg=b?valid^zero[g*W+w]:zero[g*W+w];
            rows_tile[k]=rf&rg;
        }
        __syncthreads();
        if(valid_h){
            for(int k=0;k<words;++k){
                const int w=base_word+k;
                const U left=rows_tile[k]&zero[h*W+w];
                n0+=__popcll(left);y0+=__popcll(left&positive[w]);
            }
        }
        __syncthreads();
    }

    double best=INF;int action=-2;
    if(valid_h){
        const int n1=total-n0,y1=pos-y0;
        if(n0>=ml && n1>=ml){
            best=costs[(3+q)*F+h]+weight*((y0<n0-y0?y0:n0-y0)+(y1<n1-y1?y1:n1-y1));
            action=h;
        }
    }
    minima[tid]=best;actions[tid]=action;__syncthreads();
    for(int s=128;s;s>>=1){
        if(tid<s){const double v=minima[tid+s];const int candidate=actions[tid+s];
            if(v<minima[tid] || (v==minima[tid] && candidate>=0 &&
                    (actions[tid]<0 || candidate<actions[tid]))){
                minima[tid]=v;actions[tid]=candidate;
            }}
        __syncthreads();
    }
    if(!tid){
        if(early && stop<=minima[0]){minima[0]=stop;actions[0]=-1;}
        output[out]=base+minima[0];tail[out]=actions[0];
    }
}
