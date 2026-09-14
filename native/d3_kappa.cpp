#include <algorithm>
#include <cstdint>
#include <limits>
#ifdef _MSC_VER
#include <intrin.h>
#define JT_EXPORT extern "C" __declspec(dllexport)
static inline int popcount64(std::uint64_t x) { return static_cast<int>(__popcnt64(x)); }
#else
#define JT_EXPORT extern "C"
static inline int popcount64(std::uint64_t x) { return __builtin_popcountll(x); }
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// Fused sufficient-statistic generation and min_h reduction.  Layouts match
// jt_oct.d3_batched and only the O(4F^2) output message is materialized.
JT_EXPORT int d3_kappa_cpu(
    const std::uint64_t* masks, const std::uint64_t* positive,
    const double* root_cost, const double* second_cost, const double* third_cost,
    const std::uint8_t* allowed_root, const std::uint8_t* allowed_second,
    const std::uint8_t* allowed_third, int F, int W, int no_repeat,
    int early_stop, int min_leaf, double weight, int requested_threads,
    double* output, std::int32_t* argmin_h) {
#ifdef _OPENMP
    if (requested_threads > 0) omp_set_num_threads(requested_threads);
#else
    (void)requested_threads;
#endif
    const double INF = 1.0e300;
#pragma omp parallel for schedule(static)
    for (int pair = 0; pair < F * F; ++pair) {
        const int f = pair / F;
        const int g = pair - f * F;
        int total[4] = {0, 0, 0, 0};
        int pos[4] = {0, 0, 0, 0};
        double best[4] = {INF, INF, INF, INF};
        int action[4] = {-2, -2, -2, -2};
        if (allowed_root[f] && (!no_repeat || f != g)) {
            for (int q = 0; q < 4; ++q) {
                const int a = q >> 1;
                const int b = q & 1;
                if (!allowed_second[a * F + g]) continue;
                for (int w = 0; w < W; ++w) {
                    const std::uint64_t rows = masks[(a * F + f) * W + w]
                                             & masks[(b * F + g) * W + w];
                    total[q] += popcount64(rows);
                    pos[q] += popcount64(rows & positive[w]);
                }
                if (early_stop && total[q] >= min_leaf) {
                    const int err = std::min(pos[q], total[q] - pos[q]);
                    best[q] = 0.25 * root_cost[f] + 0.5 * second_cost[a * F + g]
                            + weight * static_cast<double>(err);
                    action[q] = -1;
                }
            }
            for (int h = 0; h < F; ++h) {
                if (no_repeat && (h == f || h == g)) continue;
                for (int q = 0; q < 4; ++q) {
                    const int a = q >> 1;
                    const int b = q & 1;
                    if (!allowed_second[a * F + g] || !allowed_third[q * F + h]) continue;
                    int n0 = 0, y10 = 0;
                    for (int w = 0; w < W; ++w) {
                        const std::uint64_t left = masks[(a * F + f) * W + w]
                                                 & masks[(b * F + g) * W + w]
                                                 & masks[(0 * F + h) * W + w];
                        n0 += popcount64(left);
                        y10 += popcount64(left & positive[w]);
                    }
                    const int n1 = total[q] - n0;
                    if (n0 < min_leaf || n1 < min_leaf) continue;
                    const int y11 = pos[q] - y10;
                    const int err = std::min(y10, n0 - y10)
                                  + std::min(y11, n1 - y11);
                    const double value = 0.25 * root_cost[f]
                                       + 0.5 * second_cost[a * F + g]
                                       + third_cost[q * F + h]
                                       + weight * static_cast<double>(err);
                    // h is visited in increasing order.  A strict comparison
                    // keeps STOP on a cost tie (fewer splits) and otherwise
                    // keeps the smallest feature id, matching the Python DP.
                    if (value < best[q]) {
                        best[q] = value;
                        action[q] = h;
                    }
                }
            }
        }
        for (int q = 0; q < 4; ++q) {
            const int out = (q * F + f) * F + g;
            output[out] = best[q];
            argmin_h[out] = action[q];
        }
    }
    return 0;
}
