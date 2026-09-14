#include <cstdint>
#include <intrin.h>
using U=std::uint64_t;
extern "C" __declspec(dllexport) void d3_prepare_counts(
    const U* zero,const U* positive,const U* rows,int F,int W,int* output) {
    // Sequential native popcount: avoid one OpenMP launch for these small statistics.
    for(int f=0;f<F;++f){
        int n=0,p=0;
        for(int w=0;w<W;++w){const U r=zero[f*W+w]&rows[w];
            n+=int(__popcnt64(r));p+=int(__popcnt64(r&positive[w]));}
        output[f]=n;output[F+f]=p;
    }
}
