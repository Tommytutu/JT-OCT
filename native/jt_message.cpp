#include <algorithm>
#include <cstdint>
#include <vector>
#include <intrin.h>
#include <omp.h>

// A direct chain-message update.  Each context is the complete active ancestor
// prefix of ONE path cluster, not a subtree or a D3 optimization problem.
extern "C" __declspec(dllexport) int jt_path_costs(
    const std::uint64_t* masks, const std::uint64_t* positive,
    const double* incoming, int F, int W, int D, int root, int q,
    int no_repeat, int min_leaf, double weight, double penalty,
    int start, int count, int incoming_stride, int threads,
    double* output, std::int32_t* tail_action) {
    const double INF = 1.0e300;
    const int B = F + 1;
#pragma omp parallel num_threads(threads > 0 ? threads : omp_get_max_threads())
    {
        std::vector<std::uint64_t> rows(W);
#pragma omp for schedule(static)
        for (int local = 0; local < count; ++local) {
            const int index = start + local;
            output[local] = INF;
            tail_action[local] = -2;
            const double previous = incoming[index / incoming_stride];
            if (previous >= INF / 2) continue;
            int features[5] = {root, 0, 0, 0, 0};
            int code = index;
            for (int level = D - 2; level >= 1; --level) {
                features[level] = code % B;
                code /= B;
            }
            int stop = D - 1;
            bool valid = true;
            for (int level = 1; level < D - 1; ++level) {
                if (features[level] == F) {
                    if (stop == D - 1) stop = level;
                } else {
                    if (stop != D - 1) valid = false;
                    if (no_repeat) {
                        for (int earlier = 0; earlier < level; ++earlier)
                            if (features[earlier] == features[level]) valid = false;
                    }
                }
            }
            if (!valid) continue;
            double allocated = 0;
            for (int level = 0; level < stop; ++level)
                allocated += penalty / (1 << (D - 1 - level));
            int total = 0, pos = 0;
            for (int w = 0; w < W; ++w) {
                std::uint64_t set = ~std::uint64_t(0);
                for (int level = 0; level < stop; ++level) {
                    const int bit = (q >> (D - 2 - level)) & 1;
                    set &= masks[(bit * F + features[level]) * W + w];
                }
                rows[w] = set;
                total += static_cast<int>(__popcnt64(set));
                pos += static_cast<int>(__popcnt64(set & positive[w]));
            }
            if (stop != D - 1) {
                if (total >= min_leaf) {
                    output[local] = previous + allocated + weight * std::min(pos, total-pos)
                                    / (1 << (D - 1 - stop));
                    tail_action[local] = -3;
                }
                continue;
            }
            double best = total >= min_leaf ? weight * std::min(pos,total-pos) : INF;
            int action = best < INF ? -1 : -2;
            if (best != 0.0) {
                for (int h = 0; h < F; ++h) {
                    bool used = false;
                    if (no_repeat) for (int level = 0; level < D - 1; ++level)
                        if (features[level] == h) used = true;
                    if (used) continue;
                    int n0 = 0, y0 = 0;
                    for (int w = 0; w < W; ++w) {
                        const std::uint64_t left = rows[w] & masks[h * W + w];
                        n0 += static_cast<int>(__popcnt64(left));
                        y0 += static_cast<int>(__popcnt64(left & positive[w]));
                    }
                    const int n1 = total - n0;
                    if (n0 < min_leaf || n1 < min_leaf) continue;
                    const int y1 = pos - y0;
                    const double candidate = penalty + weight *
                        (std::min(y0,n0-y0)+std::min(y1,n1-y1));
                    if (candidate < best) { best = candidate; action = h; }
                }
            }
            if (best < INF / 2) {
                output[local] = previous + allocated + best;
                tail_action[local] = action;
            }
        }
    }
    return 0;
}

// Multiclass variant; binary entry point is unchanged.
extern "C" __declspec(dllexport) int jt_path_costs_multiclass(
    const std::uint64_t* masks, const std::uint64_t* positive,
    const double* incoming, int F, int W, int classes, int D, int root, int q,
    int no_repeat, int min_leaf, double weight, double penalty,
    int start, int count, int incoming_stride, int threads,
    double* output, std::int32_t* tail_action) {
    const double INF = 1.0e300;
    const int B = F + 1;
#pragma omp parallel num_threads(threads > 0 ? threads : omp_get_max_threads())
    {
        std::vector<std::uint64_t> rows(W);
        std::vector<int> totals(classes), left_counts(classes);
#pragma omp for schedule(static)
        for (int local = 0; local < count; ++local) {
            const int index = start + local;
            output[local] = INF;
            tail_action[local] = -2;
            const double previous = incoming[index / incoming_stride];
            if (previous >= INF / 2) continue;
            int features[5] = {root, 0, 0, 0, 0};
            int code = index;
            for (int level = D - 2; level >= 1; --level) {
                features[level] = code % B;
                code /= B;
            }
            int stop = D - 1;
            bool valid = true;
            for (int level = 1; level < D - 1; ++level) {
                if (features[level] == F) {
                    if (stop == D - 1) stop = level;
                } else {
                    if (stop != D - 1) valid = false;
                    if (no_repeat) {
                        for (int earlier = 0; earlier < level; ++earlier)
                            if (features[earlier] == features[level]) valid = false;
                    }
                }
            }
            if (!valid) continue;
            double allocated = 0;
            for (int level = 0; level < stop; ++level)
                allocated += penalty / (1 << (D - 1 - level));
            int total = 0;
            std::fill(totals.begin(), totals.end(), 0);
            for (int w = 0; w < W; ++w) {
                std::uint64_t set = ~std::uint64_t(0);
                for (int level = 0; level < stop; ++level) {
                    const int bit = (q >> (D - 2 - level)) & 1;
                    set &= masks[(bit * F + features[level]) * W + w];
                }
                rows[w] = set;
                total += static_cast<int>(__popcnt64(set));
                for (int k=0; k<classes; ++k) totals[k] += static_cast<int>(__popcnt64(set & positive[k*W+w]));
            }
            if (stop != D - 1) {
                if (total >= min_leaf) {
                    output[local] = previous + allocated + weight * (total - *std::max_element(totals.begin(), totals.end()))
                                    / (1 << (D - 1 - stop));
                    tail_action[local] = -3;
                }
                continue;
            }
            double best = total >= min_leaf ? weight * (total - *std::max_element(totals.begin(), totals.end())) : INF;
            int action = best < INF ? -1 : -2;
            if (best != 0.0) {
                for (int h = 0; h < F; ++h) {
                    bool used = false;
                    if (no_repeat) for (int level = 0; level < D - 1; ++level)
                        if (features[level] == h) used = true;
                    if (used) continue;
                    int n0 = 0;
                    std::fill(left_counts.begin(), left_counts.end(), 0);
                    for (int w = 0; w < W; ++w) {
                        const std::uint64_t left = rows[w] & masks[h * W + w];
                        n0 += static_cast<int>(__popcnt64(left));
                        for (int k=0; k<classes; ++k) left_counts[k] += static_cast<int>(__popcnt64(left & positive[k*W+w]));
                    }
                    const int n1 = total - n0;
                    if (n0 < min_leaf || n1 < min_leaf) continue;
                    int max_left=0, max_right=0;
                    for (int k=0; k<classes; ++k) {
                        max_left=std::max(max_left,left_counts[k]);
                        max_right=std::max(max_right,totals[k]-left_counts[k]);
                    }
                    const double candidate = penalty + weight *
                        (total-max_left-max_right);
                    if (candidate < best) { best = candidate; action = h; }
                }
            }
            if (best < INF / 2) {
                output[local] = previous + allocated + best;
                tail_action[local] = action;
            }
        }
    }
    return 0;
}
