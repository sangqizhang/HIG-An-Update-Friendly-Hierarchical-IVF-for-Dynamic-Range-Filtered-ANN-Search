#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace ivf_topk {

// 按 scores[id] 升序选取 top-k 簇 id（partial_sort，与 nlist 规模下 heap 相比常数更小）
inline void select_smallest_k_indices(
    const std::vector<float>& scores,
    int n_valid,
    int k,
    std::vector<int>& out) {
    out.clear();
    if (n_valid <= 0 || k <= 0 || scores.empty()) {
        return;
    }
    const int n = std::min(n_valid, static_cast<int>(scores.size()));
    const int top_k = std::min(k, n);
    out.resize(static_cast<size_t>(n));
    std::iota(out.begin(), out.end(), 0);
    std::partial_sort(
        out.begin(),
        out.begin() + top_k,
        out.end(),
        [&](int a, int b) {
            return scores[static_cast<size_t>(a)] < scores[static_cast<size_t>(b)];
        });
    out.resize(static_cast<size_t>(top_k));
}

// 在 candidates 子集上按 scores[cid] 取 top-k，结果写回 candidates（原地缩小）
inline void select_smallest_k_inplace(
    const std::vector<float>& scores,
    std::vector<int>& candidates,
    int k) {
    if (candidates.empty() || k <= 0) {
        candidates.clear();
        return;
    }
    const int top_k = std::min(k, static_cast<int>(candidates.size()));
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + top_k,
        candidates.end(),
        [&](int a, int b) {
            return scores[static_cast<size_t>(a)] < scores[static_cast<size_t>(b)];
        });
    candidates.resize(static_cast<size_t>(top_k));
}

// 按 score_of(id) 升序取 top-k；score 在比较时 lazy 求值（配合 dist 缓存可避免 union 全量预计算）
template <typename ScoreFn>
inline void select_smallest_k_lazy_inplace(
    std::vector<int>& candidates,
    int k,
    ScoreFn&& score_of) {
    if (candidates.empty() || k <= 0) {
        candidates.clear();
        return;
    }
    const int top_k = std::min(k, static_cast<int>(candidates.size()));
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + top_k,
        candidates.end(),
        [&](int a, int b) {
            return score_of(a) < score_of(b);
        });
    candidates.resize(static_cast<size_t>(top_k));
}

}  // namespace ivf_topk
