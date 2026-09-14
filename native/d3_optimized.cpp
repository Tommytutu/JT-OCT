#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>
#include <intrin.h>
#define EXPORT extern "C" __declspec(dllexport)
using U = std::uint64_t;
static int pop(U x) { return int(__popcnt64(x)); }
struct Scratch { std::vector<U> rows; std::vector<int> nz; };
struct Workspace {
    int threads;
    std::vector<Scratch> scratch;
    Workspace(int w, int nt): threads(nt > 0 ? nt : omp_get_max_threads()), scratch(threads) {
        for (auto& s:scratch) { s.rows.resize(w); s.nz.resize(w); }
    }
};
EXPORT void* d3_opt_create(int w, int nt) {
    try { return new Workspace(w, nt); } catch (...) { return nullptr; }
}
EXPORT void d3_opt_free(void* context) { delete static_cast<Workspace*>(context); }
// Costs/allowed use root, two second-level nodes, four private third-level nodes.
// Each (f,g,q) routes its bitset once and reuses it across all private h choices.
EXPORT int d3_opt_costs(void* context, const U* zero, const U* positive,
    const double* costs, const unsigned char* allowed, const double* floors,
    const unsigned char* dominated, const int* ip, double weight,
    double* output, int* tail, unsigned char* reasons) {
    auto& ws=*static_cast<Workspace*>(context);
    const int F=ip[0],W=ip[1],n=ip[2],repeat=ip[3],early=ip[4],ml=ip[5],
        bounds=ip[6],sparse=ip[7],start=ip[8],count=ip[9];
    const double INF=1e300;
    const U last=n%64 ? (U(1)<<(n%64))-1 : ~U(0);
    #pragma omp parallel for schedule(static) num_threads(ws.threads)
    for (int pair=start;pair<start+count;++pair) {
        auto& s=ws.scratch[omp_get_thread_num()];
        const int f=pair/F,g=pair%F;
        for (int q=0;q<4;++q) {
            const int a=q/2,b=q%2,out=(q*F+f)*F+g;
            output[out]=INF;tail[out]=-2;reasons[out]=0;
            if (!allowed[f] || !allowed[(1+a)*F+g] || (repeat && f==g)) continue;
            if (dominated[a*F+f]) { reasons[out]=2;continue; }
            int total=0,pos=0,nz=0;
            for (int w=0;w<W;++w) {
                const U valid=w==W-1?last:~U(0);
                const U rf=a ? valid^zero[f*W+w]:zero[f*W+w];
                const U rg=b ? valid^zero[g*W+w]:zero[g*W+w];
                const U rows=rf&rg;s.rows[w]=rows;
                total+=pop(rows);pos+=pop(rows&positive[w]);
                if (rows) s.nz[nz++]=w;
            }
            if (total<ml) continue;
            double best=early ? weight*std::min(pos,total-pos):INF;
            int action=early ? -1:-2;
            const double floor=floors[3+q];
            if (bounds && early && best<=floor) reasons[out]=3;
            else {
                reasons[out]=1;
                const int words=sparse ? nz:W;
                for (int h=0;h<F;++h) {
                    if (!allowed[(3+q)*F+h] || (repeat && (h==f || h==g))) continue;
                    int n0=0,y0=0;
                    for (int k=0;k<words;++k) {
                        const int w=sparse?s.nz[k]:k;
                        const U left=s.rows[w]&zero[h*W+w];
                        n0+=pop(left);y0+=pop(left&positive[w]);
                    }
                    const int n1=total-n0,y1=pos-y0;
                    if (n0<ml || n1<ml) continue;
                    const double value=costs[(3+q)*F+h]+weight*(std::min(y0,n0-y0)+std::min(y1,n1-y1));
                    if (value<best) { best=value;action=h; }
                    // Nonnegative misclassification gives an exact private lower bound.
                    if (bounds && best<=floor) break;
                }
            }
            output[out]=best+0.25*costs[f]+0.5*costs[(1+a)*F+g];tail[out]=action;
        }
    }
    return 0;
}

// Class-major bitsets use the actual K labels, including nonconsecutive codes.
// The caller maps class positions back to labels when recovering a leaf.
EXPORT void d3_opt_counts_multiclass(const U* zero, const U* labels, const U* rows,
    int F, int W, int K, int* output) {
    for (int f=0; f<F; ++f) {
        int n=0;
        for (int w=0; w<W; ++w) n+=pop(zero[(size_t)f*W+w]&rows[w]);
        output[f]=n;
        for (int c=0; c<K; ++c) {
            int count=0;
            for (int w=0; w<W; ++w)
                count+=pop(zero[(size_t)f*W+w]&rows[w]&labels[(size_t)c*W+w]);
            output[(size_t)(c+1)*F+f]=count;
        }
    }
}

EXPORT int d3_opt_costs_multiclass(void* context, const U* zero, const U* labels,
    const double* costs, const unsigned char* allowed, const double* floors,
    const unsigned char* dominated, const int* ip, double weight,
    double* output, int* tail, unsigned char* reasons) {
    auto& ws=*static_cast<Workspace*>(context);
    const int F=ip[0],W=ip[1],n=ip[2],repeat=ip[3],early=ip[4],ml=ip[5],
        bounds=ip[6],sparse=ip[7],start=ip[8],count=ip[9],K=ip[10];
    const double INF=1e300;
    const U last=n%64 ? (U(1)<<(n%64))-1 : ~U(0);
    #pragma omp parallel num_threads(ws.threads)
    {
        auto& s=ws.scratch[omp_get_thread_num()];
        std::vector<int> hist(K);
        #pragma omp for schedule(static)
        for (int pair=start;pair<start+count;++pair) {
            const int f=pair/F,g=pair%F;
            for (int q=0;q<4;++q) {
                const int a=q/2,b=q%2;
                const size_t out=((size_t)q*F+f)*F+g;
                output[out]=INF;tail[out]=-2;reasons[out]=0;
                if (!allowed[f] || !allowed[(1+a)*F+g] || (repeat&&f==g)) continue;
                if (dominated[a*F+f]) { reasons[out]=2;continue; }
                int total=0,nz=0;
                for (int w=0;w<W;++w) {
                    const U valid=w==W-1?last:~U(0);
                    const U rf=a?valid^zero[(size_t)f*W+w]:zero[(size_t)f*W+w];
                    const U rg=b?valid^zero[(size_t)g*W+w]:zero[(size_t)g*W+w];
                    s.rows[w]=rf&rg;total+=pop(s.rows[w]);
                    if (s.rows[w]) s.nz[nz++]=w;
                }
                if (total<ml) continue;
                const int words=sparse?nz:W;
                int majority=0;
                for (int c=0;c<K;++c) {
                    int v=0;
                    for (int k=0;k<words;++k) {
                        const int w=sparse?s.nz[k]:k;
                        v+=pop(s.rows[w]&labels[(size_t)c*W+w]);
                    }
                    hist[c]=v;majority=std::max(majority,v);
                }
                double best=early?weight*(total-majority):INF;
                int action=early?-1:-2;
                const double floor=floors[3+q];
                if (bounds&&early&&best<=floor) reasons[out]=3;
                else {
                    reasons[out]=1;
                    for (int h=0;h<F;++h) {
                        if (!allowed[(3+q)*F+h] || (repeat&&(h==f||h==g))) continue;
                        int n0=0,max0=0,max1=0;
                        for (int c=0;c<K;++c) {
                            int v=0;
                            for (int k=0;k<words;++k) {
                                const int w=sparse?s.nz[k]:k;
                                v+=pop(s.rows[w]&zero[(size_t)h*W+w]&labels[(size_t)c*W+w]);
                            }
                            n0+=v;max0=std::max(max0,v);max1=std::max(max1,hist[c]-v);
                        }
                        if (n0<ml || total-n0<ml) continue;
                        const double value=costs[(3+q)*F+h]+weight*(total-max0-max1);
                        if (value<best) { best=value;action=h; }
                        if (bounds&&best<=floor) break;
                    }
                }
                output[out]=best+.25*costs[f]+.5*costs[(1+a)*F+g];tail[out]=action;
            }
        }
    }
    return 0;
}

// Dedicated D2: no h enumeration. Each side is a complete two-leaf stump.
EXPORT int d2_opt_costs_multiclass(void* context, const U* zero, const U* labels,
    const double* costs, const unsigned char* allowed, const double* floors,
    const unsigned char* dominated, const int* ip, double weight,
    double* output, int* tail, unsigned char* reasons) {
    auto& ws=*static_cast<Workspace*>(context);
    const int F=ip[0],W=ip[1],n=ip[2],repeat=ip[3],ml=ip[5],start=ip[8],count=ip[9],K=ip[10];
    const U last=n%64 ? (U(1)<<(n%64))-1 : ~U(0);
    #pragma omp parallel for schedule(static) num_threads(ws.threads)
    for (int pair=start;pair<start+count;++pair) {
        const int f=pair/F,g=pair%F;
        for (int a=0;a<2;++a) {
            const size_t out=((size_t)a*F+f)*F+g;
            output[out]=1e300;tail[out]=-2;reasons[out]=0;
            if (!allowed[f] || !allowed[(1+a)*F+g] || (repeat&&f==g)) continue;
            int n0=0,n1=0,max0=0,max1=0;
            for (int c=0;c<K;++c) {
                int c0=0,c1=0;
                for (int w=0;w<W;++w) {
                    const U valid=w==W-1?last:~U(0);
                    const U r=(a?valid^zero[(size_t)f*W+w]:zero[(size_t)f*W+w])&labels[(size_t)c*W+w];
                    const U left=r&zero[(size_t)g*W+w];
                    c0+=pop(left);c1+=pop(r^left);
                }
                n0+=c0;n1+=c1;max0=std::max(max0,c0);max1=std::max(max1,c1);
            }
            if (n0<ml || n1<ml) continue;
            output[out]=.5*costs[f]+costs[(1+a)*F+g]+weight*(n0+n1-max0-max1);
            tail[out]=-1;reasons[out]=1;
        }
    }
    return 0;
}
