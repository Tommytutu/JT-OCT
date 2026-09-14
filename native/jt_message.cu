// GPU direct path-cluster min-sum update, one block per ancestor context.
// Raw costs are never materialized as an F^D tensor.
extern "C" __global__ void jt_path_costs_gpu(
    const unsigned long long* masks, const unsigned long long* positive,
    const double* incoming, int F, int W, int D, int root, int q,
    int no_repeat, int min_leaf, double weight, double penalty,
    int start, int count, int incoming_stride,
    double* output, int* tail_action) {
    const int local = blockIdx.x, tid = threadIdx.x;
    if (local >= count) return;
    const int index = start + local, B = F + 1;
    const double INF = 1.0e300;
    const double previous = incoming[index / incoming_stride];
    if (previous >= INF/2) {
        if (!tid) { output[local] = INF; tail_action[local] = -2; }
        return;
    }
    int features[5] = {root,0,0,0,0};
    int code = index;
    for (int level = D-2; level >= 1; --level) {
        features[level] = code % B; code /= B;
    }
    int stop = D-1;
    bool valid = true;
    for (int level=1; level<D-1; ++level) {
        if (features[level] == F) { if (stop == D-1) stop = level; }
        else {
            if (stop != D-1) valid = false;
            if (no_repeat) for (int earlier=0; earlier<level; ++earlier)
                if (features[earlier] == features[level]) valid = false;
        }
    }
    if (!valid) {
        if (!tid) { output[local]=INF; tail_action[local]=-2; }
        return;
    }
    extern __shared__ unsigned long long routed[];
    __shared__ int totals[256], positives[256];
    __shared__ double minima[256];
    __shared__ int actions[256];
    int total=0, pos=0;
    for (int w=tid; w<W; w+=blockDim.x) {
        unsigned long long set = ~0ULL;
        for (int level=0; level<stop; ++level) {
            const int bit = (q >> (D-2-level)) & 1;
            set &= masks[(bit*F+features[level])*W+w];
        }
        routed[w]=set;
        total += __popcll(set); pos += __popcll(set & positive[w]);
    }
    totals[tid]=total; positives[tid]=pos;
    __syncthreads();
    for(int stride=128; stride; stride>>=1) {
        if(tid<stride) {
            totals[tid]+=totals[tid+stride];
            positives[tid]+=positives[tid+stride];
        }
        __syncthreads();
    }
    total=totals[0]; pos=positives[0];
    double allocated=0;
    for(int level=0;level<stop;++level)
        allocated += penalty/(1 << (D-1-level));
    const int err = pos < total-pos ? pos : total-pos;
    if(stop!=D-1) {
        if(!tid) {
            output[local]=total>=min_leaf ? previous+allocated+weight*err/(1 << (D-1-stop)) : INF;
            tail_action[local]=total>=min_leaf ? -3 : -2;
        }
        return;
    }
    const double stop_loss = total>=min_leaf ? weight*err : INF;
    if(stop_loss==0.0) {
        if(!tid) { output[local]=previous+allocated; tail_action[local]=-1; }
        return;
    }
    double best=INF; int action=-2;
    for(int h=tid;h<F;h+=blockDim.x) {
        bool used=false;
        if(no_repeat) for(int level=0;level<D-1;++level)
            if(features[level]==h) used=true;
        if(used) continue;
        int n0=0,y0=0;
        for(int w=0;w<W;++w) {
            const unsigned long long left=routed[w]&masks[h*W+w];
            n0+=__popcll(left); y0+=__popcll(left & positive[w]);
        }
        const int n1=total-n0,y1=pos-y0;
        if(n0<min_leaf||n1<min_leaf) continue;
        const int e0 = y0<n0-y0 ? y0:n0-y0;
        const int e1 = y1<n1-y1 ? y1:n1-y1;
        const double candidate = penalty+weight*(e0+e1);
        if(candidate<best || (candidate==best && (action<0 || h<action))) {
            best=candidate; action=h;
        }
    }
    minima[tid]=best; actions[tid]=action;
    __syncthreads();
    for(int stride=128;stride;stride>>=1) {
        if(tid<stride) {
            const double other=minima[tid+stride];
            const int a=actions[tid+stride];
            if(other<minima[tid] || (other==minima[tid] && a>=0 &&
                                    (actions[tid]<0 || a<actions[tid]))) {
                minima[tid]=other; actions[tid]=a;
            }
        }
        __syncthreads();
    }
    if(!tid) {
        best=minima[0]; action=actions[0];
        if(stop_loss<=best) { best=stop_loss; action=stop_loss<INF/2 ? -1:-2; }
        output[local]=best<INF/2 ? previous+allocated+best : INF;
        tail_action[local]=action;
    }
}
