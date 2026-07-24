
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/lsm_ivf_list.h"
#include "../include/ada_ivf_core.h"  // 复用 AdaIVFCore 作为 fine 层
#include "../include/ada_ivf_pybind_common.h"
#include "../include/ivf_topk_utils.h"
#include "../include/simd_utils.h"     // SIMD优化函数

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>
#include <numeric>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <deque>
#include <queue>
#include <cstdio>
#include <cstring>
#include <string>


// C++11 不支持 std::make_unique，需要手动实现
#if __cplusplus < 201402L
namespace std {
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// 核心实现类（不直接暴露给 Python）
// ---------------------------------------------------------------------------

class HierarchicalAdaIVFCore {
public:
    HierarchicalAdaIVFCore(int n_fine_clusters,//fine层簇数
                           int n_coarse_clusters,//粗层簇数
                           int n_probe,
                           size_t max_cluster_size,//最大簇大小
                           float recluster_threshold,//重聚类阈值
                           float centroid_update_threshold,//质心更新阈值
                           int n_probe_f = -1)  // 细层n_probe，-1表示不限制（向后兼容）
        : n_fine_clusters_(n_fine_clusters),
          n_coarse_clusters_(n_coarse_clusters),
          base_n_probe_(n_probe),
          n_probe_f_(n_probe_f > 0 ? n_probe_f : -1),  // 修复P3：使用原子变量，只接受正数，否则为-1（不限制）
          max_cluster_size_(max_cluster_size),
          recluster_threshold_(recluster_threshold),
          centroid_update_threshold_(centroid_update_threshold),
          dimension_(0),
          is_trained_(false) {
        // 修复P3：原子变量已在初始化列表中初始化，无需额外操作
        if (n_fine_clusters_ <= 0 || n_coarse_clusters_ <= 0) {
            throw std::runtime_error("n_fine_clusters 和 n_coarse_clusters 必须为正");
        }
        if (n_coarse_clusters_ > n_fine_clusters_) {
            // 简单保护：粗聚类不能多于细聚类
            n_coarse_clusters_ = n_fine_clusters_;
        }

        // 使用 AdaIVFCore 作为 fine 层（复用 SIMD、OpenMP 等优化）
        // 注意：已删除 recluster_ratio 参数（与单层实现保持一致）
        // fine 层 AdaIVFCore::n_probe_：与最终扫描预算 n_probe_c×n_probe_f 同量级（供内部维护语义）
        int fine_n_probe = n_fine_clusters_;
        if (n_probe_f > 0 && base_n_probe_ > 0) {
            fine_n_probe = std::min(n_fine_clusters_, base_n_probe_ * n_probe_f);
        } else if (n_probe_f > 0) {
            fine_n_probe = std::min(n_fine_clusters_, n_probe_f);
        }
        fine_index_ = std::make_unique<AdaIVFCore>(
            n_fine_clusters_, fine_n_probe,  // n_clusters, n_probe (fine 层探测预算)
            max_cluster_size_, recluster_threshold_);
        fine_index_->set_enable_maintenance(false);

        fine_to_coarse_.assign(static_cast<size_t>(n_fine_clusters_), 0);
        local_kmeans_ = std::make_unique<SimpleKMeans>();
        
        // 修复P2：移除固定间隔检查，改为自适应触发
        const char* env_dbg = std::getenv("HIER_ADA_IVF_DEBUG");
        debug_log_ = (env_dbg != nullptr) && (std::string(env_dbg) == "1");
    }

    // 使用训练向量（learn_vectors）构建 fine / coarse 质心。
    // 边界约定：scalar 不进入训练目标；scalar 只在插入后形成 synopsis，
    // 并在查询期用于 range prune / filter-aware graph routing。
    void train(const float* data, size_t n_vectors, int dimension, const float* scalars = nullptr) {
        (void)scalars;
        if (n_vectors == 0) {
            throw std::runtime_error("训练向量不能为空");
        }
        dimension_ = dimension;
        is_trained_ = false;

        // 1) 使用 AdaIVFCore 训练 fine 层
        fine_index_->train(data, n_vectors, dimension_, nullptr);
        if (!fine_index_->is_trained()) {
            throw std::runtime_error("Fine 层训练失败");
        }

        // 2) 从 fine_index_ 获取 fine 质心（用于构建 coarse 层）
        // 现在可以使用 get_centroids() 方法获取
        fine_centroids_ = fine_index_->get_centroids();
        if (static_cast<int>(fine_centroids_.size()) < n_fine_clusters_) {
            n_fine_clusters_ = static_cast<int>(fine_centroids_.size());
        }

        // 3) 使用 fine 质心训练 coarse 质心
        {
            size_t n_fine = fine_centroids_.size();
            if (n_fine == 0) {
                throw std::runtime_error("细粒度质心为空，无法训练顶层");
            }

            // 优化：coarse层训练使用更少的迭代次数（fine质心数量少，收敛快）
            // 将 fine_centroids_ 展平为连续数组
            std::vector<float> flat_fine(n_fine * static_cast<size_t>(dimension_));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < n_fine; ++i) {
                std::memcpy(
                    flat_fine.data() + i * static_cast<size_t>(dimension_),
                    fine_centroids_[i].data(),
                    static_cast<size_t>(dimension_) * sizeof(float));
            }

            int n_coarse_train = std::min(n_coarse_clusters_, static_cast<int>(n_fine));
            std::vector<uint32_t> fine_ids(n_fine);
            std::iota(fine_ids.begin(), fine_ids.end(), 0);

            // 使用与单层训练相同的高标准迭代次数（最大 40 次）
            int coarse_max_iter = 20;
            if (n_fine >= 500000 || n_coarse_train >= 500) {
                coarse_max_iter = 40;
            } else if (n_fine >= 200000 || n_coarse_train >= 300) {
                coarse_max_iter = 35;
            } else if (n_fine >= 100000 || n_coarse_train >= 200) {
                coarse_max_iter = 30;
            } else if (n_fine < 10000 && n_coarse_train < 100) {
                coarse_max_iter = 20;
            } else {
                coarse_max_iter = 20;
            }
            local_kmeans_->train_with_max_iter(flat_fine.data(), n_fine, dimension_, n_coarse_train, coarse_max_iter);
            coarse_centroids_ = local_kmeans_->get_centroids();
            if (static_cast<int>(coarse_centroids_.size()) < n_coarse_clusters_) {
                n_coarse_clusters_ = static_cast<int>(coarse_centroids_.size());
            }
        }

        // 4) 建立 fine -> coarse 映射（无图模式跳过 seed 预计算）
        build_fine_to_coarse_mapping(enable_graph_routing_.load());

        // Module B：仅图路由开启时构建 fine centroid graph
        if (enable_graph_routing_.load()) {
            rebuild_fine_centroid_graph(/*force=*/true);
            fine_centroids_graph_history_ = fine_centroids_;
        }
        
        // 5) 保存初始 fine 质心作为历史（用于后续变化检测）
        fine_centroids_history_ = fine_centroids_;
        // 保存训练时的初始质心（用于计算epsilon，永远不更新）
        fine_centroids_initial_ = fine_centroids_;

        is_trained_ = true;
    }

    void train_coarse_first(const float* data, size_t n_vectors, int dimension, const float* scalars) {
        if (data == nullptr || n_vectors == 0) {
            throw std::runtime_error("empty training vectors");
        }
        SimpleKMeans coarse_kmeans;
        coarse_kmeans.set_random_seed(fine_index_->get_kmeans_seed());
        coarse_kmeans.set_use_kmeanspp(fine_index_->get_use_kmeanspp());
        coarse_kmeans.set_attr_lambda(0.0f);
        coarse_kmeans.set_scalar_span_lambda(0.0f);
        coarse_kmeans.train_with_max_iter(data, n_vectors, dimension, n_coarse_clusters_, 30, nullptr);
        coarse_centroids_ = coarse_kmeans.get_centroids();
        if (static_cast<int>(coarse_centroids_.size()) < n_coarse_clusters_) {
            n_coarse_clusters_ = static_cast<int>(coarse_centroids_.size());
        }

        std::vector<std::vector<uint32_t>> coarse_members(static_cast<size_t>(n_coarse_clusters_));
        for (size_t i = 0; i < n_vectors; ++i) {
            const float* v = data + i * static_cast<size_t>(dimension);
            float best = std::numeric_limits<float>::max();
            int best_cid = 0;
            for (int cid = 0; cid < n_coarse_clusters_; ++cid) {
                float d = l2_distance_sq(v, coarse_centroids_[static_cast<size_t>(cid)].data());
                if (d < best) {
                    best = d;
                    best_cid = cid;
                }
            }
            coarse_members[static_cast<size_t>(best_cid)].push_back(static_cast<uint32_t>(i));
        }

        std::vector<int> fine_budget(static_cast<size_t>(n_coarse_clusters_), 0);
        std::vector<std::pair<double, int>> remainders;
        int assigned = 0;
        for (int cid = 0; cid < n_coarse_clusters_; ++cid) {
            const size_t cnt = coarse_members[static_cast<size_t>(cid)].size();
            if (cnt == 0) continue;
            double exact = static_cast<double>(cnt) * static_cast<double>(n_fine_clusters_) /
                           static_cast<double>(std::max<size_t>(1, n_vectors));
            int base = std::max(1, static_cast<int>(std::floor(exact)));
            fine_budget[static_cast<size_t>(cid)] = base;
            assigned += base;
            remainders.emplace_back(exact - static_cast<double>(base), cid);
        }
        std::sort(remainders.begin(), remainders.end(),
                  [](const std::pair<double,int>& a, const std::pair<double,int>& b) { return a.first > b.first; });
        size_t rr = 0;
        while (assigned < n_fine_clusters_ && !remainders.empty()) {
            fine_budget[static_cast<size_t>(remainders[rr % remainders.size()].second)]++;
            assigned++;
            rr++;
        }
        std::sort(remainders.begin(), remainders.end(),
                  [](const std::pair<double,int>& a, const std::pair<double,int>& b) { return a.first < b.first; });
        rr = 0;
        while (assigned > n_fine_clusters_ && !remainders.empty()) {
            int cid = remainders[rr % remainders.size()].second;
            if (fine_budget[static_cast<size_t>(cid)] > 1) {
                fine_budget[static_cast<size_t>(cid)]--;
                assigned--;
            }
            rr++;
            if (rr > remainders.size() * 4 && assigned > n_fine_clusters_) break;
        }

        fine_centroids_.clear();
        std::vector<float> fine_scalar_centroids;
        std::vector<float> fine_scalar_mins;
        std::vector<float> fine_scalar_maxs;
        fine_centroids_.reserve(static_cast<size_t>(n_fine_clusters_));
        fine_scalar_centroids.reserve(static_cast<size_t>(n_fine_clusters_));
        fine_scalar_mins.reserve(static_cast<size_t>(n_fine_clusters_));
        fine_scalar_maxs.reserve(static_cast<size_t>(n_fine_clusters_));

        float global_scalar_min = 0.0f;
        float global_scalar_max = 1.0f;
        if (scalars != nullptr && n_vectors > 0) {
            global_scalar_min = scalars[0];
            global_scalar_max = scalars[0];
            for (size_t i = 1; i < n_vectors; ++i) {
                global_scalar_min = std::min(global_scalar_min, scalars[i]);
                global_scalar_max = std::max(global_scalar_max, scalars[i]);
            }
            if (!(global_scalar_min < global_scalar_max)) {
                global_scalar_max = global_scalar_min + 1.0f;
            }
        }
        const float global_scalar_span = std::max(1e-6f, global_scalar_max - global_scalar_min);

        for (int cid = 0; cid < n_coarse_clusters_; ++cid) {
            auto& members = coarse_members[static_cast<size_t>(cid)];
            int k_local = fine_budget[static_cast<size_t>(cid)];
            if (members.empty() || k_local <= 0) continue;
            // Keep partition geometry-first. Mode=2 no longer uses the strong scalar-band
            // refine path; scalar should guide query-time navigation, not dominate clustering.
            const bool use_scalar_band_refine = false;
            bool use_selective_scalar_split = false;
            if (!use_scalar_band_refine && scalars != nullptr && coarse_first_mode_ == 2 &&
                fine_index_->get_scalar_span_lambda() > 0.0f && k_local >= 4 &&
                members.size() >= static_cast<size_t>(k_local) * 16u) {
                std::vector<float> local_scalar_values;
                local_scalar_values.reserve(members.size());
                for (uint32_t idx : members) local_scalar_values.push_back(scalars[idx]);
                std::sort(local_scalar_values.begin(), local_scalar_values.end());
                const float local_span = local_scalar_values.back() - local_scalar_values.front();
                const size_t q25_idx = local_scalar_values.size() / 4u;
                const size_t q75_idx = (local_scalar_values.size() * 3u) / 4u;
                const float local_iqr = local_scalar_values[q75_idx] - local_scalar_values[q25_idx];
                float max_gap = 0.0f;
                for (size_t gi = 1; gi < local_scalar_values.size(); ++gi) {
                    max_gap = std::max(max_gap, local_scalar_values[gi] - local_scalar_values[gi - 1]);
                }
                const float span_ratio = local_span / global_scalar_span;
                const float iqr_ratio = local_iqr / std::max(1e-6f, local_span);
                const float gap_ratio = max_gap / std::max(1e-6f, local_span);
                constexpr int kGateBins = 16;
                std::array<uint32_t, kGateBins> gate_hist{};
                for (float sv : local_scalar_values) {
                    int bin = static_cast<int>(((sv - global_scalar_min) / global_scalar_span) * kGateBins);
                    if (bin < 0) bin = 0;
                    if (bin >= kGateBins) bin = kGateBins - 1;
                    gate_hist[static_cast<size_t>(bin)] += 1u;
                }
                double entropy = 0.0;
                uint32_t max_bin_count = 0;
                const double inv_local_n = 1.0 / static_cast<double>(local_scalar_values.size());
                for (uint32_t c : gate_hist) {
                    max_bin_count = std::max(max_bin_count, c);
                    if (c == 0) continue;
                    const double p = static_cast<double>(c) * inv_local_n;
                    entropy -= p * std::log(p);
                }
                const float entropy_norm = static_cast<float>(entropy / std::log(static_cast<double>(kGateBins)));
                const float max_bin_frac = static_cast<float>(static_cast<double>(max_bin_count) * inv_local_n);
                const bool scalar_distribution_concentrated =
                    entropy_norm <= 0.82f || max_bin_frac >= 0.20f || gap_ratio >= 0.03f;
                use_selective_scalar_split =
                    span_ratio >= 0.35f && iqr_ratio >= 0.25f && gap_ratio >= 0.015f &&
                    scalar_distribution_concentrated;
            }
            if (use_scalar_band_refine) {
                std::sort(members.begin(), members.end(), [scalars](uint32_t a, uint32_t b) { return scalars[a] < scalars[b]; });
                const int n_bands = std::min(
                    k_local,
                    std::max(2, static_cast<int>(std::sqrt(static_cast<double>(std::max(1, k_local))))));
                std::vector<size_t> band_begin(static_cast<size_t>(n_bands), 0);
                std::vector<size_t> band_end(static_cast<size_t>(n_bands), 0);
                std::vector<int> band_budget(static_cast<size_t>(n_bands), 1);
                std::vector<std::pair<double, int>> band_remainders;
                int budget_sum = 0;
                for (int b = 0; b < n_bands; ++b) {
                    size_t begin = static_cast<size_t>(b) * members.size() / static_cast<size_t>(n_bands);
                    size_t end = static_cast<size_t>(b + 1) * members.size() / static_cast<size_t>(n_bands);
                    if (end <= begin) end = std::min(members.size(), begin + 1);
                    band_begin[static_cast<size_t>(b)] = begin;
                    band_end[static_cast<size_t>(b)] = end;
                    const size_t band_n = end - begin;
                    const double exact = static_cast<double>(k_local) * static_cast<double>(band_n) /
                                         static_cast<double>(std::max<size_t>(1, members.size()));
                    int base = std::max(1, static_cast<int>(std::floor(exact)));
                    band_budget[static_cast<size_t>(b)] = base;
                    budget_sum += base;
                    band_remainders.emplace_back(exact - static_cast<double>(base), b);
                }
                std::sort(band_remainders.begin(), band_remainders.end(),
                          [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                              return a.first > b.first;
                          });
                size_t rr_band = 0;
                while (budget_sum < k_local && !band_remainders.empty()) {
                    int b = band_remainders[rr_band % band_remainders.size()].second;
                    band_budget[static_cast<size_t>(b)]++;
                    budget_sum++;
                    rr_band++;
                }
                std::sort(band_remainders.begin(), band_remainders.end(),
                          [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                              return a.first < b.first;
                          });
                rr_band = 0;
                while (budget_sum > k_local && !band_remainders.empty()) {
                    int b = band_remainders[rr_band % band_remainders.size()].second;
                    if (band_budget[static_cast<size_t>(b)] > 1) {
                        band_budget[static_cast<size_t>(b)]--;
                        budget_sum--;
                    }
                    rr_band++;
                    if (rr_band > band_remainders.size() * 4 && budget_sum > k_local) break;
                }

                for (int b = 0; b < n_bands; ++b) {
                    const size_t begin = band_begin[static_cast<size_t>(b)];
                    const size_t end = band_end[static_cast<size_t>(b)];
                    if (end <= begin) continue;
                    const size_t band_n = end - begin;
                    int kb = std::min<int>(std::max(1, band_budget[static_cast<size_t>(b)]),
                                           static_cast<int>(band_n));
                    if (kb <= 1) {
                        std::vector<float> c(static_cast<size_t>(dimension), 0.0f);
                        double ssum = 0.0;
                        float smin = scalars[members[begin]];
                        float smax = smin;
                        for (size_t pos = begin; pos < end; ++pos) {
                            uint32_t idx = members[pos];
                            const float* v = data + static_cast<size_t>(idx) * static_cast<size_t>(dimension);
                            for (int d = 0; d < dimension; ++d) c[static_cast<size_t>(d)] += v[d];
                            ssum += static_cast<double>(scalars[idx]);
                            smin = std::min(smin, scalars[idx]);
                            smax = std::max(smax, scalars[idx]);
                        }
                        const float inv = 1.0f / static_cast<float>(std::max<size_t>(1, band_n));
                        for (int d = 0; d < dimension; ++d) c[static_cast<size_t>(d)] *= inv;
                        fine_centroids_.push_back(std::move(c));
                        fine_scalar_centroids.push_back(static_cast<float>(ssum * static_cast<double>(inv)));
                        fine_scalar_mins.push_back(smin);
                        fine_scalar_maxs.push_back(smax);
                    } else {
                        std::vector<float> local_data(band_n * static_cast<size_t>(dimension));
                        std::vector<float> local_scalars(band_n, 0.0f);
                        for (size_t i = 0; i < band_n; ++i) {
                            uint32_t idx = members[begin + i];
                            std::memcpy(local_data.data() + i * static_cast<size_t>(dimension),
                                        data + static_cast<size_t>(idx) * static_cast<size_t>(dimension),
                                        static_cast<size_t>(dimension) * sizeof(float));
                            local_scalars[i] = scalars[idx];
                        }
                        SimpleKMeans local;
                        local.set_random_seed(fine_index_->get_kmeans_seed() +
                                              static_cast<uint32_t>((cid + 1) * 131 + b));
                        local.set_use_kmeanspp(fine_index_->get_use_kmeanspp());
                        local.set_attr_lambda(0.0f);
                        local.set_scalar_span_lambda(0.0f);
                        local.train_with_max_iter(local_data.data(), band_n, dimension, kb, 20, local_scalars.data());
                        const auto& cents = local.get_centroids();
                        const auto& scs = local.get_scalar_centroids();
                        const auto& smins = local.get_scalar_mins();
                        const auto& smaxs = local.get_scalar_maxs();
                        for (size_t li = 0; li < cents.size(); ++li) {
                            fine_centroids_.push_back(cents[li]);
                            fine_scalar_centroids.push_back(li < scs.size() ? scs[li] : 0.0f);
                            fine_scalar_mins.push_back(li < smins.size() ? smins[li] : scalars[members[begin]]);
                            fine_scalar_maxs.push_back(li < smaxs.size() ? smaxs[li] : scalars[members[end - 1]]);
                        }
                    }
                }
            } else if (use_selective_scalar_split) {
                std::sort(members.begin(), members.end(), [scalars](uint32_t a, uint32_t b) { return scalars[a] < scalars[b]; });
                std::vector<size_t> boundaries(static_cast<size_t>(k_local + 1), 0);
                boundaries[0] = 0;
                boundaries[static_cast<size_t>(k_local)] = members.size();
                for (int b = 0; b < k_local; ++b) {
                    size_t begin = static_cast<size_t>(b) * members.size() / static_cast<size_t>(k_local);
                    size_t end = static_cast<size_t>(b + 1) * members.size() / static_cast<size_t>(k_local);
                    if (end <= begin) end = std::min(members.size(), begin + 1);
                    boundaries[static_cast<size_t>(b)] = begin;
                    boundaries[static_cast<size_t>(b + 1)] = end;
                }

                const size_t local_n = members.size();
                std::vector<double> prefix_norm2(local_n + 1, 0.0);
                std::vector<double> prefix_sum((local_n + 1) * static_cast<size_t>(dimension), 0.0);
                for (size_t i = 0; i < local_n; ++i) {
                    const float* v = data + static_cast<size_t>(members[i]) * static_cast<size_t>(dimension);
                    double sq = 0.0;
                    prefix_norm2[i + 1] = prefix_norm2[i];
                    for (int d = 0; d < dimension; ++d) {
                        const double vd = static_cast<double>(v[d]);
                        prefix_sum[(i + 1) * static_cast<size_t>(dimension) + static_cast<size_t>(d)] =
                            prefix_sum[i * static_cast<size_t>(dimension) + static_cast<size_t>(d)] + vd;
                        sq += vd * vd;
                    }
                    prefix_norm2[i + 1] += sq;
                }

                const float scalar_domain = std::max(1e-6f, scalars[members.back()] - scalars[members.front()]);
                const float span_lambda = std::max(0.0f, fine_index_->get_scalar_span_lambda());
                auto segment_objective = [&](size_t begin, size_t end) -> double {
                    if (end <= begin) {
                        return std::numeric_limits<double>::infinity();
                    }
                    const double count = static_cast<double>(end - begin);
                    double sum_norm2 = prefix_norm2[end] - prefix_norm2[begin];
                    double mean_sq = 0.0;
                    for (int d = 0; d < dimension; ++d) {
                        const double seg_sum =
                            prefix_sum[end * static_cast<size_t>(dimension) + static_cast<size_t>(d)] -
                            prefix_sum[begin * static_cast<size_t>(dimension) + static_cast<size_t>(d)];
                        mean_sq += seg_sum * seg_sum;
                    }
                    double sse = sum_norm2 - mean_sq / std::max(1.0, count);
                    if (span_lambda > 0.0f) {
                        const float span = scalars[members[end - 1]] - scalars[members[begin]];
                        const double norm_span = static_cast<double>(span / scalar_domain);
                        sse += static_cast<double>(span_lambda) * norm_span * norm_span * count;
                    }
                    return sse;
                };

                const size_t base_window = std::max<size_t>(8, members.size() / static_cast<size_t>(std::max(1, k_local * 8)));
                for (int iter = 0; iter < 2; ++iter) {
                    bool moved = false;
                    for (int b = 1; b < k_local; ++b) {
                        const size_t left_begin = boundaries[static_cast<size_t>(b - 1)];
                        const size_t right_end = boundaries[static_cast<size_t>(b + 1)];
                        const size_t cur = boundaries[static_cast<size_t>(b)];
                        const size_t low = std::max(left_begin + 1, cur > base_window ? cur - base_window : left_begin + 1);
                        const size_t high = std::min(right_end - 1, cur + base_window);
                        if (low > high) continue;
                        size_t best_split = cur;
                        double best_obj = segment_objective(left_begin, cur) + segment_objective(cur, right_end);
                        for (size_t split = low; split <= high; ++split) {
                            double obj = segment_objective(left_begin, split) + segment_objective(split, right_end);
                            if (obj < best_obj) {
                                best_obj = obj;
                                best_split = split;
                            }
                        }
                        if (best_split != cur) {
                            boundaries[static_cast<size_t>(b)] = best_split;
                            moved = true;
                        }
                    }
                    if (!moved) break;
                }

                for (int b = 0; b < k_local; ++b) {
                    size_t begin = boundaries[static_cast<size_t>(b)];
                    size_t end = boundaries[static_cast<size_t>(b + 1)];
                    if (end <= begin) end = std::min(members.size(), begin + 1);
                    std::vector<float> c(static_cast<size_t>(dimension), 0.0f);
                    double ssum = 0.0;
                    float smin = scalars[members[begin]];
                    float smax = smin;
                    for (size_t pos = begin; pos < end; ++pos) {
                        uint32_t idx = members[pos];
                        const float* v = data + static_cast<size_t>(idx) * static_cast<size_t>(dimension);
                        for (int d = 0; d < dimension; ++d) c[static_cast<size_t>(d)] += v[d];
                        ssum += static_cast<double>(scalars[idx]);
                        smin = std::min(smin, scalars[idx]);
                        smax = std::max(smax, scalars[idx]);
                    }
                    float inv = 1.0f / static_cast<float>(std::max<size_t>(1, end - begin));
                    for (int d = 0; d < dimension; ++d) c[static_cast<size_t>(d)] *= inv;
                    fine_centroids_.push_back(std::move(c));
                    fine_scalar_centroids.push_back(static_cast<float>(ssum * static_cast<double>(inv)));
                    fine_scalar_mins.push_back(smin);
                    fine_scalar_maxs.push_back(smax);
                }
            } else {
                std::vector<float> local_data(members.size() * static_cast<size_t>(dimension));
                std::vector<float> local_scalars(members.size(), 0.0f);
                for (size_t i = 0; i < members.size(); ++i) {
                    uint32_t idx = members[i];
                    std::memcpy(local_data.data() + i * static_cast<size_t>(dimension),
                                data + static_cast<size_t>(idx) * static_cast<size_t>(dimension),
                                static_cast<size_t>(dimension) * sizeof(float));
                    local_scalars[i] = scalars ? scalars[idx] : 0.0f;
                }
                SimpleKMeans local;
                local.set_random_seed(fine_index_->get_kmeans_seed() + static_cast<uint32_t>(cid + 1));
                local.set_use_kmeanspp(fine_index_->get_use_kmeanspp());
                const bool mode2_vector_fallback = (coarse_first_mode_ == 2);
                local.set_attr_lambda(mode2_vector_fallback ? 0.0f : fine_index_->get_attr_lambda());
                local.set_scalar_span_lambda(mode2_vector_fallback ? 0.0f : fine_index_->get_scalar_span_lambda());
                local.train_with_max_iter(local_data.data(), members.size(), dimension, k_local, 20, scalars ? local_scalars.data() : nullptr);
                const auto& cents = local.get_centroids();
                const auto& scs = local.get_scalar_centroids();
                const auto& smins = local.get_scalar_mins();
                const auto& smaxs = local.get_scalar_maxs();
                for (size_t li = 0; li < cents.size(); ++li) {
                    fine_centroids_.push_back(cents[li]);
                    fine_scalar_centroids.push_back(li < scs.size() ? scs[li] : 0.0f);
                    fine_scalar_mins.push_back(li < smins.size() ? smins[li] : 0.0f);
                    fine_scalar_maxs.push_back(li < smaxs.size() ? smaxs[li] : 0.0f);
                }
            }
        }

        while (static_cast<int>(fine_centroids_.size()) < n_fine_clusters_) {
            int cid = static_cast<int>(fine_centroids_.size()) % std::max(1, n_coarse_clusters_);
            fine_centroids_.push_back(coarse_centroids_[static_cast<size_t>(cid)]);
            fine_scalar_centroids.push_back(0.0f);
            fine_scalar_mins.push_back(0.0f);
            fine_scalar_maxs.push_back(0.0f);
        }
        if (static_cast<int>(fine_centroids_.size()) > n_fine_clusters_) {
            fine_centroids_.resize(static_cast<size_t>(n_fine_clusters_));
            fine_scalar_centroids.resize(static_cast<size_t>(n_fine_clusters_));
            fine_scalar_mins.resize(static_cast<size_t>(n_fine_clusters_));
            fine_scalar_maxs.resize(static_cast<size_t>(n_fine_clusters_));
        }

        fine_index_->train_with_centroids(data, n_vectors, dimension_, scalars,
                                          fine_centroids_, fine_scalar_centroids,
                                          fine_scalar_mins, fine_scalar_maxs);
        build_fine_to_coarse_mapping(enable_graph_routing_.load());
        if (enable_graph_routing_.load()) {
            rebuild_fine_centroid_graph(true);
            fine_centroids_graph_history_ = fine_centroids_;
        }
        fine_centroids_history_ = fine_centroids_;
        fine_centroids_initial_ = fine_centroids_;
        is_trained_ = true;
    }

    // 插入基础向量（base_vectors）
    void add(const float* data,
             size_t n_vectors,
             const int* ids,
             bool auto_recluster,
             const float* scalars) {
        if (!is_trained_) {
            throw std::runtime_error("索引未训练，无法插入向量");
        }
        if (dimension_ <= 0) {
            throw std::runtime_error("维度未初始化");
        }
        last_graph_rebuild_time_s_ = 0.0;

        // 调试：记录插入前状态
        size_t n_vectors_before = fine_index_->get_n_vectors();
        if (debug_log_) {
            std::cerr << "[层次化插入调试] 开始插入 " << n_vectors
                      << " 个向量, 当前索引向量数: " << n_vectors_before << std::endl;
        }

        // 直接使用 AdaIVFCore 的 add 方法（复用其优化）
        // [优化·问题6-E] 分段计时：HIER_ADA_IVF_PROFILE_INSERT=1 或 debug_log_ 时输出
        const char* profile_env = std::getenv("HIER_ADA_IVF_PROFILE_INSERT");
        const bool profile_insert = debug_log_ || (profile_env != nullptr && profile_env[0] == '1');
        auto add_start = std::chrono::high_resolution_clock::now();
        fine_index_->add(data, n_vectors, ids, auto_recluster, scalars);
        auto add_end = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - add_start).count();
        if (debug_log_) {
            std::cerr << "[层次化插入调试] Fine层插入完成, 耗时: " << add_duration << "ms" << std::endl;
        }

        // 图同步节流计数：仅维护开启时累计
        if (enable_maintenance_) {
            inserted_since_graph_rebuild_ += n_vectors;
            inserted_since_centroid_refresh_ += n_vectors;
            inserted_since_coarse_check_ += n_vectors;
            inserted_since_coarse_rebuild_ += n_vectors;
        }

        long long ms_graph_sync = 0;
        long long ms_coarse_check = 0;
        
        // 关键优化：不要每次 add 都全量拉取 fine 质心（非常重）。
        // - 只在“需要做 coarse 检查 / 建图判断 / coarse 重建”时再刷新。
        auto refresh_fine_centroids_if_needed = [&](bool force) {
            if (!force && inserted_since_centroid_refresh_ < centroid_refresh_min_insertions_) return;
            auto centroid_start = std::chrono::high_resolution_clock::now();
            fine_centroids_ = fine_index_->get_centroids();
            inserted_since_centroid_refresh_ = 0;
            auto centroid_end = std::chrono::high_resolution_clock::now();
            if (debug_log_) {
                auto centroid_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(centroid_end - centroid_start).count();
                std::cerr << "[层次化插入调试] 更新fine质心完成, 耗时: " << centroid_duration
                          << "ms, 质心数: " << fine_centroids_.size() << std::endl;
            }
        };

        // 图 / coarse 维护：仅 enable_maintenance_ 且 auto_recluster 时运行（纯 IVF 插入路径零维护开销）
        if (enable_maintenance_ && auto_recluster) {
            auto graph_sync_start = std::chrono::high_resolution_clock::now();
            const bool hit_insert = (inserted_since_graph_rebuild_ >= graph_rebuild_min_insertions_);
            if (hit_insert) {
                refresh_fine_centroids_if_needed(/*force=*/true);
                float delta = compute_graph_centroid_change_mean();
                if (delta > graph_rebuild_epsilon_) {
                    auto g0 = std::chrono::high_resolution_clock::now();
                    rebuild_fine_centroid_graph(/*force=*/true);
                    auto g1 = std::chrono::high_resolution_clock::now();
                    auto gms = std::chrono::duration_cast<std::chrono::milliseconds>(g1 - g0).count();
                    last_graph_rebuild_time_s_ += std::chrono::duration<double>(g1 - g0).count();
                    if (debug_log_) {
                        std::cerr << "[层次化插入调试] 图同步更新: delta=" << delta
                                  << ", 耗时=" << gms << "ms"
                                  << ", hit_insert=" << (hit_insert ? 1 : 0)
                                  << ", inserted_since_last=" << inserted_since_graph_rebuild_
                                  << std::endl;
                    }
                }
            }
            ms_graph_sync = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - graph_sync_start).count();
        }
        
        // 修复epsilon NaN问题：如果fine_centroids_history_为空，初始化它
        if (enable_maintenance_ && fine_centroids_history_.empty() && !fine_centroids_.empty()) {
            fine_centroids_history_ = fine_centroids_;
            if (debug_log_) {
                std::cerr << "[层次化插入调试] 初始化fine_centroids_history_, 质心数: " << fine_centroids_history_.size() << std::endl;
            }
        }
        
        if (enable_maintenance_ && auto_recluster) {
            if (inserted_since_coarse_check_ >= coarse_check_every_insertions_) {
                inserted_since_coarse_check_ = 0;
                refresh_fine_centroids_if_needed(/*force=*/true);
                const int rebuilds_before = coarse_global_rebuild_count_;
                auto check_start = std::chrono::high_resolution_clock::now();
                check_and_update_coarse_layer();
                auto check_end = std::chrono::high_resolution_clock::now();
                ms_coarse_check = std::chrono::duration_cast<std::chrono::milliseconds>(
                    check_end - check_start).count();
                if (coarse_global_rebuild_count_ > rebuilds_before) {
                    last_graph_rebuild_time_s_ += std::chrono::duration<double>(
                        check_end - check_start).count();
                }
                if (debug_log_) {
                    std::cerr << "[层次化插入调试] Coarse层检查完成, 耗时: " << ms_coarse_check << "ms" << std::endl;
                }
            }
        } else if (debug_log_) {
            std::cerr << "[层次化插入调试] 维护已关闭（enable_maintenance=false 或 auto_recluster=false），跳过图/coarse 维护" << std::endl;
        }

        if (profile_insert) {
            std::cerr << "[层次化插入Profile] fine_add=" << add_duration << "ms"
                      << " graph_sync=" << ms_graph_sync << "ms"
                      << " coarse_check=" << ms_coarse_check << "ms"
                      << " n=" << n_vectors << std::endl;
        }
        
        size_t n_vectors_after = fine_index_->get_n_vectors();
        if (debug_log_) {
            std::cerr << "[层次化插入调试] 插入完成, 索引向量数: " << n_vectors_before << " -> " << n_vectors_after << std::endl;
        }
    }

    // -------------------------------------------------------------------------
    // batch_search：Python 查询入口
    // 流程：同步 live 质心 → 逐 query 调用 search_single → 返回 top-k (dist², id)
    // OpenMP：设环境变量 HIER_ADA_IVF_OMP=1 可并行多条 query
    // -------------------------------------------------------------------------
    std::vector<std::vector<std::pair<float, int>>> batch_search(
        const float* queries,
        size_t n_queries,
        int k,
        int base_n_probe,
        int base_search_k,
        float range_min,
        float range_max,
        bool use_mid_filtering) const {
        (void)use_mid_filtering;  // 标量范围过滤在 AdaIVFCore::gather_cluster_candidate_ids 内实现
        std::vector<std::vector<std::pair<float, int>>> all_results;
        all_results.resize(n_queries);

        if (!is_trained_ || fine_index_->get_n_vectors() == 0) {
            return all_results;
        }

        // 每批一次：引用 fine 层 live 质心做路由（const 引用，避免 ~2MB 按值拷贝）
        const auto& live_centroids = fine_index_->get_centroids();
        const std::vector<std::vector<float>>& route_centroids =
            live_centroids.empty() ? fine_centroids_ : live_centroids;

        int n_probe = (base_n_probe > 0) ? base_n_probe : base_n_probe_;
        if (n_probe > n_coarse_clusters_) n_probe = n_coarse_clusters_;
        if (n_probe <= 0) n_probe = std::min(4, n_coarse_clusters_);

        int budget = base_search_k > 0 ? base_search_k : 0;  // 0 表示无预算限制

        // [优化·问题3-A] 批内复用 scratch：每批只 ensure_size 一次，避免每条 query malloc
        QuerySearchScratch batch_scratch;
        batch_scratch.ensure_size(n_fine_clusters_, n_coarse_clusters_);

        const char* env_omp = std::getenv("HIER_ADA_IVF_OMP");
        // 默认多核并行 batch 查询；设 HIER_ADA_IVF_OMP=0 可关闭。
        // range benchmark 常出现每组只有 1 条 query，此时进入 OpenMP 会反复分配线程 scratch，反而更慢。
        const bool use_omp = (n_queries > 1) && !(env_omp != nullptr && env_omp[0] == '0');
        if (use_omp) {
#ifdef _OPENMP
            const int n_threads = omp_get_max_threads();
            std::vector<QuerySearchScratch> omp_scratches(static_cast<size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                omp_scratches[static_cast<size_t>(t)].ensure_size(n_fine_clusters_, n_coarse_clusters_);
            }
#pragma omp parallel for
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                QuerySearchScratch& scratch = omp_scratches[static_cast<size_t>(omp_get_thread_num())];
                all_results[qi] = search_single(q, k, n_probe, budget, range_min, range_max, scratch, route_centroids);
            }
#else
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                all_results[qi] = search_single(q, k, n_probe, budget, range_min, range_max, batch_scratch, route_centroids);
            }
#endif
        } else {
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                all_results[qi] = search_single(q, k, n_probe, budget, range_min, range_max, batch_scratch, route_centroids);
            }
        }

        return all_results;
    }

    // 与 AdaIVFCore::batch_search_ranges 对齐：每条 query 使用 ranges[i*2], ranges[i*2+1]
    std::vector<std::vector<std::pair<float, int>>> batch_search_ranges(
        const float* queries,
        const float* ranges,
        size_t n_queries,
        int k,
        int base_n_probe,
        int base_search_k,
        bool use_mid_filtering) const {
        (void)use_mid_filtering;
        if (ranges == nullptr) {
            return batch_search(
                queries, n_queries, k, base_n_probe, base_search_k,
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::max(),
                use_mid_filtering);
        }

        std::vector<std::vector<std::pair<float, int>>> all_results(n_queries);
        if (!is_trained_ || fine_index_->get_n_vectors() == 0) {
            return all_results;
        }

        const auto& live_centroids = fine_index_->get_centroids();
        const std::vector<std::vector<float>>& route_centroids =
            live_centroids.empty() ? fine_centroids_ : live_centroids;

        int n_probe = (base_n_probe > 0) ? base_n_probe : base_n_probe_;
        if (n_probe > n_coarse_clusters_) n_probe = n_coarse_clusters_;
        if (n_probe <= 0) n_probe = std::min(4, n_coarse_clusters_);

        int budget = base_search_k > 0 ? base_search_k : 0;

        QuerySearchScratch batch_scratch;
        batch_scratch.ensure_size(n_fine_clusters_, n_coarse_clusters_);

        const char* env_omp = std::getenv("HIER_ADA_IVF_OMP");
        const bool use_omp = (n_queries > 1) && !(env_omp != nullptr && env_omp[0] == '0');
        if (use_omp) {
#ifdef _OPENMP
            const int n_threads = omp_get_max_threads();
            std::vector<QuerySearchScratch> omp_scratches(static_cast<size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                omp_scratches[static_cast<size_t>(t)].ensure_size(n_fine_clusters_, n_coarse_clusters_);
            }
#pragma omp parallel for schedule(static)
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                const float rmin = ranges[qi * 2];
                const float rmax = ranges[qi * 2 + 1];
                QuerySearchScratch& scratch = omp_scratches[static_cast<size_t>(omp_get_thread_num())];
                all_results[qi] = search_single(q, k, n_probe, budget, rmin, rmax, scratch, route_centroids);
            }
#else
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                const float rmin = ranges[qi * 2];
                const float rmax = ranges[qi * 2 + 1];
                all_results[qi] = search_single(q, k, n_probe, budget, rmin, rmax, batch_scratch, route_centroids);
            }
#endif
        } else {
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                const float rmin = ranges[qi * 2];
                const float rmax = ranges[qi * 2 + 1];
                all_results[qi] = search_single(q, k, n_probe, budget, rmin, rmax, batch_scratch, route_centroids);
            }
        }

        return all_results;
    }

    AdaIVFCore::BatchSearchResult batch_search_with_stats(
        const float* queries,
        size_t n_queries,
        int k,
        int base_n_probe,
        int base_search_k,
        float range_min,
        float range_max,
        bool use_mid_filtering) const {
        (void)use_mid_filtering;
        AdaIVFCore::BatchSearchResult out;
        out.results.resize(n_queries);
        out.per_query_stats.resize(n_queries);
        if (!is_trained_ || fine_index_->get_n_vectors() == 0) {
            return out;
        }

        const auto& live_centroids = fine_index_->get_centroids();
        const std::vector<std::vector<float>>& route_centroids =
            live_centroids.empty() ? fine_centroids_ : live_centroids;

        int n_probe = (base_n_probe > 0) ? base_n_probe : base_n_probe_;
        if (n_probe > n_coarse_clusters_) n_probe = n_coarse_clusters_;
        if (n_probe <= 0) n_probe = std::min(4, n_coarse_clusters_);
        int budget = base_search_k > 0 ? base_search_k : 0;

        QuerySearchScratch batch_scratch;
        batch_scratch.ensure_size(n_fine_clusters_, n_coarse_clusters_);
        const char* env_omp = std::getenv("HIER_ADA_IVF_OMP");
        const bool use_omp = (n_queries > 1) && !(env_omp != nullptr && env_omp[0] == '0');
        if (use_omp) {
#ifdef _OPENMP
            const int n_threads = omp_get_max_threads();
            std::vector<QuerySearchScratch> omp_scratches(static_cast<size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                omp_scratches[static_cast<size_t>(t)].ensure_size(n_fine_clusters_, n_coarse_clusters_);
            }
#pragma omp parallel for schedule(static)
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                QuerySearchScratch& scratch = omp_scratches[static_cast<size_t>(omp_get_thread_num())];
                out.results[qi] = search_single(
                    q, k, n_probe, budget, range_min, range_max, scratch, route_centroids,
                    &out.per_query_stats[qi]);
            }
#else
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                out.results[qi] = search_single(
                    q, k, n_probe, budget, range_min, range_max, batch_scratch, route_centroids,
                    &out.per_query_stats[qi]);
            }
#endif
        } else {
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                out.results[qi] = search_single(
                    q, k, n_probe, budget, range_min, range_max, batch_scratch, route_centroids,
                    &out.per_query_stats[qi]);
            }
        }
        return out;
    }

    AdaIVFCore::BatchSearchResult batch_search_ranges_with_stats(
        const float* queries,
        const float* ranges,
        size_t n_queries,
        int k,
        int base_n_probe,
        int base_search_k,
        bool use_mid_filtering) const {
        if (ranges == nullptr) {
            return batch_search_with_stats(
                queries, n_queries, k, base_n_probe, base_search_k,
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::max(),
                use_mid_filtering);
        }
        (void)use_mid_filtering;
        AdaIVFCore::BatchSearchResult out;
        out.results.resize(n_queries);
        out.per_query_stats.resize(n_queries);
        if (!is_trained_ || fine_index_->get_n_vectors() == 0) {
            return out;
        }

        const auto& live_centroids = fine_index_->get_centroids();
        const std::vector<std::vector<float>>& route_centroids =
            live_centroids.empty() ? fine_centroids_ : live_centroids;
        int n_probe = (base_n_probe > 0) ? base_n_probe : base_n_probe_;
        if (n_probe > n_coarse_clusters_) n_probe = n_coarse_clusters_;
        if (n_probe <= 0) n_probe = std::min(4, n_coarse_clusters_);
        int budget = base_search_k > 0 ? base_search_k : 0;

        QuerySearchScratch batch_scratch;
        batch_scratch.ensure_size(n_fine_clusters_, n_coarse_clusters_);
        const char* env_omp = std::getenv("HIER_ADA_IVF_OMP");
        const bool use_omp = (n_queries > 1) && !(env_omp != nullptr && env_omp[0] == '0');
        if (use_omp) {
#ifdef _OPENMP
            const int n_threads = omp_get_max_threads();
            std::vector<QuerySearchScratch> omp_scratches(static_cast<size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                omp_scratches[static_cast<size_t>(t)].ensure_size(n_fine_clusters_, n_coarse_clusters_);
            }
#pragma omp parallel for schedule(static)
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                const float rmin = ranges[qi * 2];
                const float rmax = ranges[qi * 2 + 1];
                QuerySearchScratch& scratch = omp_scratches[static_cast<size_t>(omp_get_thread_num())];
                out.results[qi] = search_single(
                    q, k, n_probe, budget, rmin, rmax, scratch, route_centroids,
                    &out.per_query_stats[qi]);
            }
#else
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                const float rmin = ranges[qi * 2];
                const float rmax = ranges[qi * 2 + 1];
                out.results[qi] = search_single(
                    q, k, n_probe, budget, rmin, rmax, batch_scratch, route_centroids,
                    &out.per_query_stats[qi]);
            }
#endif
        } else {
            for (size_t qi = 0; qi < n_queries; ++qi) {
                const float* q = queries + qi * static_cast<size_t>(dimension_);
                const float rmin = ranges[qi * 2];
                const float rmax = ranges[qi * 2 + 1];
                out.results[qi] = search_single(
                    q, k, n_probe, budget, rmin, rmax, batch_scratch, route_centroids,
                    &out.per_query_stats[qi]);
            }
        }
        return out;
    }

    AdaIVFCore* fine_core() { return fine_index_.get(); }
    const AdaIVFCore* fine_core() const { return fine_index_.get(); }

    size_t n_vectors() const { 
        return fine_index_->get_n_vectors();
    }

    AdaIVFCore::ClusterSizeStats get_cluster_size_stats() const {
        return fine_index_->get_cluster_size_stats();
    }

    std::vector<size_t> get_all_cluster_sizes() const {
        return fine_index_->get_all_cluster_sizes();
    }

    int global_rebuild_count() const {
        return coarse_global_rebuild_count_;
    }
    int coarse_rebuild_count() const { return coarse_global_rebuild_count_; }
    int coarse_soft_sync_count() const { return coarse_soft_sync_count_; }
    int coarse_soft_moved_fines_last() const { return coarse_soft_moved_fines_last_; }
    int coarse_soft_moved_fines_total() const { return coarse_soft_moved_fines_total_; }
    float coarse_soft_moved_ratio_last() const { return coarse_soft_moved_ratio_last_; }
    float coarse_soft_rel_shift_last() const { return coarse_soft_rel_shift_last_; }
    float coarse_soft_avg_radius_last() const { return coarse_soft_avg_radius_last_; }
    size_t adaptive_fanout_query_count() const { return adaptive_fanout_query_count_.load(); }
    size_t adaptive_fanout_extra_coarse_total() const { return adaptive_fanout_extra_coarse_total_.load(); }
    size_t adaptive_fanout_extra_target_total() const { return adaptive_fanout_extra_target_total_.load(); }
    size_t fine_refresh_count() const { return fine_index_->get_fine_refresh_count(); }
    size_t local_recluster_count() const { return fine_index_->get_local_recluster_count(); }
    size_t migrated_vector_count() const { return fine_index_->get_migrated_vector_count(); }
    float maintenance_G_before() const { return fine_index_->get_last_maintenance_G_before(); }
    float maintenance_G_after() const { return fine_index_->get_last_maintenance_G_after(); }
    
    // 修复P3：动态设置n_probe_f（使用原子变量保证线程安全）
    void set_n_probe_f(int n_probe_f) {
        n_probe_f_.store((n_probe_f > 0) ? n_probe_f : -1);
    }
    
    int get_n_probe_f() const {
        return n_probe_f_.load();
    }
    
    // 获取失衡指标（用于Python接口）
    struct HierarchicalImbalanceMetrics {
        float sigma_c;        // 粗层分区大小标准差（64 coarse 簇）
        float sigma_f;        // 细层分区大小标准差（4096 fine 簇，与单层 σ 同义，用于对比）
        float epsilon_c;       // 重建误差（质心变化均值）
        float epsilon_c_prime; // 漂移误差（质心变化标准差）
        float G_c;             // 粗层全局失衡指标
        float G;               // 细层全局失衡指标（从fine_index_获取，用于对比）
        float avg_coarse_size; // 粗层非空簇平均向量数（与sigma_c口径一致，用于cluster_size_cv）
        float cluster_size_cv; // 粗层簇大小变异系数 = sigma_c / avg_coarse_size
    };
    
    // 获取失衡指标实现（内联定义）
    HierarchicalImbalanceMetrics get_imbalance_metrics() const {
        HierarchicalImbalanceMetrics metrics;
        metrics.sigma_c = 0.0f;
        metrics.sigma_f = 0.0f;
        metrics.epsilon_c = 0.0f;
        metrics.epsilon_c_prime = 0.0f;
        metrics.G_c = 0.0f;
        metrics.G = 0.0f;
        metrics.avg_coarse_size = 0.0f;
        metrics.cluster_size_cv = 0.0f;

        if (!enable_maintenance_) {
            return metrics;
        }
        
        if (!is_trained_ || fine_centroids_.empty() || coarse_centroids_.empty()) {
            return metrics;
        }
        
        // 1. 计算分区大小标准差 σ_c（按向量数量，与 compute_global_imbalance_indicator_v2 一致）
        std::vector<size_t> coarse_cluster_sizes = get_coarse_cluster_sizes_by_vectors();
        
        float avg_size = 0.0f;
        size_t count = 0;
        for (size_t cid = 0; cid < static_cast<size_t>(n_coarse_clusters_); ++cid) {
            if (coarse_cluster_sizes[cid] > 0) {
                avg_size += static_cast<float>(coarse_cluster_sizes[cid]);
                count++;
            }
        }
        if (count == 0) return metrics;
        avg_size /= static_cast<float>(count);
        
        float variance = 0.0f;
        for (size_t cid = 0; cid < static_cast<size_t>(n_coarse_clusters_); ++cid) {
            if (coarse_cluster_sizes[cid] > 0) {
                float diff = static_cast<float>(coarse_cluster_sizes[cid]) - avg_size;
                variance += diff * diff;
            }
        }
        metrics.sigma_c = std::sqrt(variance / static_cast<float>(count));
        
        // 1b. 粗层 cluster_size_cv（与 sigma_c 使用相同 avg_size 口径）
        metrics.avg_coarse_size = avg_size;
        metrics.cluster_size_cv = (avg_size > 0) ? (metrics.sigma_c / avg_size) : 0.0f;
        
        // 2–3. 细层 σ_f、ε、ε'、G：全部取自 fine_index_（AdaIVFCore），与单层动态实验同一套定义，
        // 避免此前「ε 用层次化质心漂移、G 用 fine 内部」混用导致 CSV 中 G ≠ (σ_f+ε+ε')/norm。
        // 粗层 G_c 仍用 compute_global_imbalance_indicator_v2（层次化质心 + 粗层统计），与此处细层指标独立。
        auto fine_metrics = fine_index_->get_imbalance_metrics();
        metrics.sigma_f = fine_metrics.sigma;
        metrics.epsilon_c = fine_metrics.epsilon;
        metrics.epsilon_c_prime = fine_metrics.epsilon_prime;
        metrics.G = fine_metrics.G;
        
        // 5. 计算全局失衡指标 G_c（使用v2版本）
        metrics.G_c = compute_global_imbalance_indicator_v2();
        
        return metrics;
    }

    // ========================================================================
    // Module A（标量 LSM）
    // Module B：图搜索路由（enable_graph_routing）；簇维护（enable_maintenance_，默认关）
    // ========================================================================

    // ========================================================================
    // 删除支持（Tombstone Delete）
    // ========================================================================
    void remove(const int* ids, size_t n) {
        if (fine_index_) fine_index_->remove(ids, n);
    }
    size_t get_n_deleted() const { return fine_index_ ? fine_index_->get_n_deleted() : 0; }
    size_t get_n_live_vectors() const { return fine_index_ ? fine_index_->get_n_live_vectors() : 0; }
    void compact_deleted_vectors() { if (fine_index_) fine_index_->compact_deleted_vectors(); }
    size_t compact_deleted_vectors_step(size_t max_clusters) {
        return fine_index_ ? fine_index_->compact_deleted_vectors_step(max_clusters) : 0;
    }
    void set_keep_list_packed_payload(bool keep) { if (fine_index_) fine_index_->set_keep_list_packed_payload(keep); }
    bool get_keep_list_packed_payload() const { return fine_index_ ? fine_index_->get_keep_list_packed_payload() : true; }

    // ========================================================================
    // 直方图剪枝
    // ========================================================================
    void set_enable_histogram_prune(bool v) { if (fine_index_) fine_index_->set_enable_histogram_prune(v); }
    bool get_enable_histogram_prune() const { return fine_index_ ? fine_index_->get_enable_histogram_prune() : false; }
    void set_histogram_prune_threshold(float t) { if (fine_index_) fine_index_->set_histogram_prune_threshold(t); }
    float get_histogram_prune_threshold() const { return fine_index_ ? fine_index_->get_histogram_prune_threshold() : 0.005f; }

    // ========================================================================
    // 级联探测早停
    // ========================================================================
    void set_enable_cascade_early_exit(bool v) { if (fine_index_) fine_index_->set_enable_cascade_early_exit(v); }
    bool get_enable_cascade_early_exit() const { return fine_index_ ? fine_index_->get_enable_cascade_early_exit() : false; }
    void set_cascade_early_exit_alpha(float alpha) { if (fine_index_) fine_index_->set_cascade_early_exit_alpha(alpha); }
    float get_cascade_early_exit_alpha() const { return fine_index_ ? fine_index_->get_cascade_early_exit_alpha() : 1.0f; }

    // ========================================================================
    // 选择率自适应双路由
    // ========================================================================
    void set_enable_adaptive_routing(bool v) { enable_adaptive_routing_ = v; }
    bool get_enable_adaptive_routing() const { return enable_adaptive_routing_; }
    /// 选择率估计 > high_threshold 时走快路径（不激活图路由）
    void set_adaptive_routing_high_threshold(float t) { adaptive_routing_high_threshold_ = std::min(1.0f, std::max(0.0f, t)); }
    float get_adaptive_routing_high_threshold() const { return adaptive_routing_high_threshold_; }

    void set_enable_maintenance(bool v) {
        enable_maintenance_ = v;
        if (fine_index_) {
            fine_index_->set_enable_maintenance(v);
        }
    }
    bool get_enable_maintenance() const { return enable_maintenance_; }

    void set_enable_graph_routing(bool v) { enable_graph_routing_.store(bool(v)); }
    bool get_enable_graph_routing() const { return enable_graph_routing_.load(); }
    double last_graph_rebuild_time_s() const { return last_graph_rebuild_time_s_; }

    void set_graph_degree(int k) { graph_degree_ = std::max(4, k); }
    int get_graph_degree() const { return graph_degree_; }
    void set_seed_L(int l) { seed_L_ = std::max(1, l); }
    int get_seed_L() const { return seed_L_; }
    void set_beam_width(int w) { beam_width_ = std::max(1, w); }
    int get_beam_width() const { return beam_width_; }
    void set_early_stop_patience(int p) { early_stop_patience_ = std::max(1, p); }
    int get_early_stop_patience() const { return early_stop_patience_; }
    void set_early_stop_relax(float r) { early_stop_relax_ = std::max(1.0f, r); }
    float get_early_stop_relax() const { return early_stop_relax_; }
    void set_graph_repair_budget(int v) { graph_repair_budget_ = std::max(0, v); }
    int get_graph_repair_budget() const { return graph_repair_budget_; }
    void set_enable_filter_aware_routing(bool v) { enable_filter_aware_routing_ = v; }
    bool get_enable_filter_aware_routing() const { return enable_filter_aware_routing_; }
    void set_filter_selectivity_epsilon(float v) { filter_selectivity_epsilon_ = std::max(0.0f, v); }
    float get_filter_selectivity_epsilon() const { return filter_selectivity_epsilon_; }
    void set_enable_adaptive_fanout(bool v) { enable_adaptive_fanout_ = v; }
    bool get_enable_adaptive_fanout() const { return enable_adaptive_fanout_; }
    void set_filter_fanout_temperature_ratio(float v) { filter_fanout_temperature_ratio_ = std::max(1e-3f, v); }
    float get_filter_fanout_temperature_ratio() const { return filter_fanout_temperature_ratio_; }

    /// 在同一质心/分区上，按当前 graph_degree_ / seed_L_ 重建 seeds + fine 质心图
    void rebuild_graph_routing_structures() {
        rebuild_coarse_seed_fines();
        rebuild_fine_centroid_graph(/*force=*/true);
    }

    void save_graph_routing_cache(const std::string& key) {
        GraphRoutingCacheEntry entry;
        entry.fine_graph = fine_graph_;
        entry.coarse_seed_fines = coarse_seed_fines_;
        entry.graph_degree = graph_degree_;
        entry.seed_L = seed_L_;
        graph_routing_cache_[key] = std::move(entry);
    }

    bool load_graph_routing_cache(const std::string& key) {
        auto it = graph_routing_cache_.find(key);
        if (it == graph_routing_cache_.end()) return false;
        fine_graph_ = it->second.fine_graph;
        rebuild_fine_graph_flat_storage();
        coarse_seed_fines_ = it->second.coarse_seed_fines;
        graph_degree_ = it->second.graph_degree;
        seed_L_ = it->second.seed_L;
        return true;
    }

    void clear_graph_routing_cache() { graph_routing_cache_.clear(); }

    size_t graph_routing_cache_size() const { return graph_routing_cache_.size(); }

    void set_enable_scalar_filter_lsm(bool v) { fine_index_->set_enable_scalar_filter_lsm(v); }
    bool get_enable_scalar_filter_lsm() const { return fine_index_->get_enable_scalar_filter_lsm(); }
    void set_enable_scalar_range_prune(bool v) { fine_index_->set_enable_scalar_range_prune(v); }
    bool get_enable_scalar_range_prune() const { return fine_index_->get_enable_scalar_range_prune(); }
    void set_lsm_merge_threshold(size_t t) { fine_index_->set_lsm_merge_threshold(t); }
    size_t get_lsm_merge_threshold() const { return fine_index_->get_lsm_merge_threshold(); }

    /// 保存/导出前建议调用：将所有 IVF 桶的无序段并入有序段
    void flush_all_lsm_segments() { fine_index_->flush_all_lsm_segments(); }

    // Deprecated no-op training knobs kept for Python/script compatibility.
    // Scalar-guided graph is query-time only; training remains pure vector KMeans.
    void set_attr_lambda(float) { fine_index_->set_attr_lambda(0.0f); }
    float get_attr_lambda() const { return 0.0f; }
    void set_scalar_span_lambda(float) { fine_index_->set_scalar_span_lambda(0.0f); }
    float get_scalar_span_lambda() const { return 0.0f; }
    void set_enable_coarse_first_training(bool) { enable_coarse_first_training_ = false; }
    bool get_enable_coarse_first_training() const { return false; }
    void set_coarse_first_mode(int) { coarse_first_mode_ = 2; }
    int get_coarse_first_mode() const { return 2; }
    void set_kmeans_seed(uint32_t v) {
        fine_index_->set_kmeans_seed(v);
        if (local_kmeans_) local_kmeans_->set_random_seed(v);
    }
    uint32_t get_kmeans_seed() const { return fine_index_->get_kmeans_seed(); }
    void set_use_kmeanspp(bool v) {
        fine_index_->set_use_kmeanspp(v);
        if (local_kmeans_) local_kmeans_->set_use_kmeanspp(v);
    }
    bool get_use_kmeanspp() const { return fine_index_->get_use_kmeanspp(); }

    // Module PQ：透传到 fine 层 AdaIVFCore
    void set_enable_pq_compression(bool v) { fine_index_->set_enable_pq_compression(v); }
    bool get_enable_pq_compression() const { return fine_index_->get_enable_pq_compression(); }
    void set_pq_m(int m) { fine_index_->set_pq_m(m); }
    int get_pq_m() const { return fine_index_->get_pq_m(); }
    void set_pq_ksub(int k) { fine_index_->set_pq_ksub(k); }
    int get_pq_ksub() const { return fine_index_->get_pq_ksub(); }
    void set_pq_release_floats(bool v) { fine_index_->set_pq_release_floats(v); }
    bool get_pq_release_floats() const { return fine_index_->get_pq_release_floats(); }
    bool is_pq_trained() const { return fine_index_->is_pq_trained(); }
    void train_pq_from_samples(const float* data, size_t n_vectors, int dimension) {
        fine_index_->train_pq_from_samples(data, n_vectors, dimension);
    }

    std::vector<std::vector<float>> get_fine_centroids() const {
        return fine_index_->get_centroids();
    }

    /// 诊断：vec_id 为 fine 层内部连续 id（按序插入时通常等于 SIFT 行号）
    int find_fine_cluster_for_vec_id(int vec_id) const {
        if (vec_id < 0) {
            return -1;
        }
        return fine_index_->find_cluster_for_vec_id(static_cast<uint32_t>(vec_id));
    }

    /// 大规模实验维护档位：throughput（默认）| balanced | large_recall
    void set_maintenance_profile(const std::string& profile) {
        if (profile == "balanced") {
            adaptive_config_.base_tau = 0.50f;
            adaptive_config_.sensitivity = 2.0f;
            adaptive_config_.min_tau = 0.38f;
            adaptive_config_.max_tau = 0.82f;
            coarse_soft_drift_tau_ = 0.06f;
            fine_remap_rel_improvement_ = 0.12f;
            fine_remap_abs_margin_ratio_ = 0.04f;
            coarse_rebuild_min_insertions_ = 300000;
            graph_rebuild_min_insertions_ = 500000;
            centroid_refresh_min_insertions_ = 500000;
            coarse_check_every_insertions_ = 200000;
        } else if (profile == "large_recall") {
            adaptive_config_.base_tau = 0.42f;
            adaptive_config_.sensitivity = 1.8f;
            adaptive_config_.min_tau = 0.35f;
            adaptive_config_.max_tau = 0.72f;
            coarse_soft_drift_tau_ = 0.04f;
            fine_remap_rel_improvement_ = 0.08f;
            fine_remap_abs_margin_ratio_ = 0.03f;
            coarse_rebuild_min_insertions_ = 200000;
            graph_rebuild_min_insertions_ = 300000;
            centroid_refresh_min_insertions_ = 300000;
            coarse_check_every_insertions_ = 150000;
        } else {
            // throughput：与历史默认一致（插入优先，少触发 coarse 重建）
            adaptive_config_.base_tau = 0.6f;
            adaptive_config_.sensitivity = 3.0f;
            adaptive_config_.min_tau = 0.4f;
            adaptive_config_.max_tau = 0.95f;
            coarse_soft_drift_tau_ = 0.08f;
            fine_remap_rel_improvement_ = 0.18f;
            fine_remap_abs_margin_ratio_ = 0.06f;
            coarse_rebuild_min_insertions_ = 500000;
            graph_rebuild_min_insertions_ = 500000;
            centroid_refresh_min_insertions_ = 500000;
            coarse_check_every_insertions_ = 200000;
        }
    }

    void set_coarse_check_every_insertions(size_t n) {
        if (n > 0) {
            coarse_check_every_insertions_ = n;
        }
    }

    /// 跳过 G_c 判定，强制 coarse 全局重建（重训 coarse + 重映射 + 重建图）
    void force_coarse_rebuild() {
        if (!is_trained_) {
            return;
        }
        fine_centroids_ = fine_index_->get_centroids();
        if (fine_centroids_history_.empty() && !fine_centroids_.empty()) {
            fine_centroids_history_ = fine_centroids_;
        }
        update_coarse_layer_from_fine();
        fine_centroids_history_ = fine_centroids_;
        coarse_global_rebuild_count_++;
        gc_history_.values.clear();
        inserted_since_coarse_check_ = 0;
        inserted_since_coarse_rebuild_ = 0;
    }

private:
    // [优化·问题3-A/B/D] 批内复用 scratch + 世代标记，避免每条 query 堆分配/清零 O(n_fine) 数组
    struct GraphBeamNode {
        float dsq;
        int fid;
    };

    struct QuerySearchScratch {
        std::vector<float> dist_cache;
        std::vector<uint32_t> dist_stamp;
        std::vector<float> sel_cache;
        std::vector<uint32_t> sel_stamp;
        std::vector<float> scalar_utility_cache;
        std::vector<uint32_t> scalar_utility_stamp;
        std::vector<uint32_t> visit_stamp;
        std::vector<uint32_t> top_stamp;
        std::vector<uint32_t> protected_seed_stamp;
        std::vector<uint32_t> coarse_selected_stamp;
        std::vector<float> coarse_scores;
        std::vector<int> coarse_order;
        std::vector<int> candidate_fine_ids;
        // 图 beam 复用缓冲，避免每条 query 堆分配 seeds/explored/pq
        std::vector<int> graph_seeds;
        std::vector<int> graph_explored;
        std::vector<GraphBeamNode> graph_beam_nodes;
        std::vector<GraphBeamNode> graph_fallback_nodes;
        std::vector<GraphBeamNode> graph_top_nodes;
        std::vector<int> graph_top_fids;
        std::vector<float> graph_top_dists;
        uint32_t generation = 1u;

        void ensure_size(int n_fine, int n_coarse = 0) {
            const size_t nf = static_cast<size_t>(std::max(0, n_fine));
            if (dist_cache.size() < nf) {
                dist_cache.resize(nf, 0.0f);
                dist_stamp.resize(nf, 0u);
                sel_cache.resize(nf, 1.0f);
                sel_stamp.resize(nf, 0u);
                scalar_utility_cache.resize(nf, 1.0f);
                scalar_utility_stamp.resize(nf, 0u);
                visit_stamp.resize(nf, 0u);
                top_stamp.resize(nf, 0u);
                protected_seed_stamp.resize(nf, 0u);
            }
            const size_t nc = static_cast<size_t>(std::max(0, n_coarse));
            if (nc > 0) {
                if (coarse_scores.size() < nc) {
                    coarse_scores.resize(nc);
                }
                if (coarse_order.size() < nc) {
                    coarse_order.resize(nc);
                }
                if (coarse_selected_stamp.size() < nc) {
                    coarse_selected_stamp.resize(nc, 0u);
                }
            }
        }

        uint32_t bump_generation() {
            ++generation;
            if (generation == 0u) {
                std::fill(dist_stamp.begin(), dist_stamp.end(), 0u);
                std::fill(sel_stamp.begin(), sel_stamp.end(), 0u);
                std::fill(scalar_utility_stamp.begin(), scalar_utility_stamp.end(), 0u);
                std::fill(visit_stamp.begin(), visit_stamp.end(), 0u);
                std::fill(top_stamp.begin(), top_stamp.end(), 0u);
                std::fill(protected_seed_stamp.begin(), protected_seed_stamp.end(), 0u);
                std::fill(coarse_selected_stamp.begin(), coarse_selected_stamp.end(), 0u);
                generation = 1u;
            }
            return generation;
        }
    };

    // [优化·问题4-A] n_fine 不超过此值时不走 beam 图（小 nlist 图导航常数开销大于收益）
    int n_fine_clusters_;
    int n_coarse_clusters_;
    int base_n_probe_;
    std::atomic<int> n_probe_f_;  // 修复P3：使用原子变量保证线程安全，细层n_probe，-1表示不限制
    size_t max_cluster_size_;
    float recluster_threshold_;
    float centroid_update_threshold_;

    int dimension_;
    bool is_trained_;

    // Fine 层：使用 AdaIVFCore（复用 SIMD、OpenMP 等优化）
    std::unique_ptr<AdaIVFCore> fine_index_;

    // 新 Module B：图搜索路由开关
    std::atomic<bool> enable_graph_routing_{false};
    // 簇维护（fine 重聚类 / coarse 重建 / 插入期图同步）；默认关，纯 IVF 基线
    bool enable_maintenance_{false};

    // Coarse 层：聚类结构
    std::vector<std::vector<float>> fine_centroids_;    // [fine_id][dim]（用于 coarse 层构建）
    std::vector<std::vector<float>> fine_centroids_history_;  // 上一次的 fine 质心（用于计算变化）
    std::vector<std::vector<float>> fine_centroids_initial_;  // 训练时的初始 fine 质心（用于计算epsilon，永远不更新）
    std::vector<std::vector<float>> coarse_centroids_;  // [coarse_id][dim]
    std::vector<int> fine_to_coarse_;                   // [fine_id] -> coarse_id
    // 优化：预先构建coarse到fine的反向索引，避免每次查询都遍历所有fine聚类
    std::vector<std::vector<int>> coarse_to_fine_;      // [coarse_id] -> vector of fine_ids

    // ---------------------------------------------------------------------
    // Module B（新策略）：fine centroid graph + coarse seeds + beam search
    // 用于 coarse→fine 图 beam 路由
    // ---------------------------------------------------------------------
    int seed_L_ = 4;                 // 每个 coarse cluster 取 Top-L fine 作为 seeds（代表点初始化入口）
    int graph_degree_ = 8;          // fine centroid graph 的 kNN 度
    int graph_neighbor_coarses_ = 8; // 建图时每个 coarse 纳入的邻近粗簇数（跨 coarse 桥接）
    int beam_width_ = 8;            // 每节点扩展上限；实际 eff_beam=min(beam_width_, graph_degree_)
    int early_stop_patience_ = 8;   // early stop：连续无改进扩展次数
    float early_stop_relax_ = 1.05f; // early stop：允许的相对退化
    int graph_repair_budget_ = 8;    // 0=自动取 max(2*seed_L, ceil(target/4))
    int n_probe_f_default_ = 4;      // 当 n_probe_f=-1（不限制）时，用于目标簇数的兜底
    bool enable_filter_aware_routing_ = false; // range 查询时用轻量 ScalarSynopsis 修正路由分数
    bool enable_adaptive_fanout_ = false;      // 默认关闭，作为独立消融项验证
    float filter_selectivity_epsilon_ = 0.25f; // selectivity bias 强度；距离仍为主导
    float filter_fanout_temperature_ratio_ = 0.15f; // coarse 自适应 fanout 的距离温度比例
    float adaptive_fanout_boundary_margin_ratio_ = 0.005f; // kth 与 next coarse 距离差低于该比例视为边界 query
    size_t adaptive_fanout_late_vectors_ = 7000000;       // DEEP 后期漂移风险阈值，低于此只对边界 query 加预算
    int adaptive_fanout_boundary_extra_coarse_ = 1;       // 边界 query 额外 coarse 覆盖
    int adaptive_fanout_late_extra_coarse_ = 2;           // 后期/软同步风险额外 coarse 覆盖
    size_t adaptive_fanout_deep_late_vectors_ = 8500000;  // DEEP 深后期：漂移进一步累积
    size_t adaptive_fanout_tail_vectors_ = 9000000;       // DEEP 尾段：强制更高 coarse 覆盖
    int adaptive_fanout_deep_late_extra_coarse_ = 2;      // 深后期基础额外 coarse 覆盖
    int adaptive_fanout_tail_extra_coarse_ = 3;           // 尾段基础额外 coarse 覆盖
    mutable std::atomic<size_t> adaptive_fanout_query_count_{0};
    mutable std::atomic<size_t> adaptive_fanout_extra_coarse_total_{0};
    mutable std::atomic<size_t> adaptive_fanout_extra_target_total_{0};

    // 运行时结构
    std::vector<std::vector<int>> coarse_seed_fines_;   // [coarse_id] -> seeds fine_ids（预计算）
    std::vector<std::vector<int>> coarse_neighbor_coarses_; // [coarse_id] -> 质心最近的其它 coarse
    std::vector<std::vector<int>> fine_graph_;          // [fine_id] -> neighbor fine_ids（kNN 图）
    std::vector<int> fine_graph_offsets_;                // 查询热路径使用的 CSR offsets
    std::vector<int> fine_graph_neighbors_;              // 查询热路径使用的连续邻接表

    struct GraphRoutingCacheEntry {
        std::vector<std::vector<int>> fine_graph;
        std::vector<std::vector<int>> coarse_seed_fines;
        int graph_degree;
        int seed_L;
    };
    std::unordered_map<std::string, GraphRoutingCacheEntry> graph_routing_cache_;

    // 质心更新后同步更新图：用“变化幅度 + 时间间隔”节流，避免每次 add 都全量重建
    float graph_rebuild_epsilon_ = 1e-3f;       // 质心平均漂移超过该阈值才重建
    size_t graph_rebuild_min_insertions_ = 500000; // 
    size_t inserted_since_graph_rebuild_ = 0;
    double last_graph_rebuild_time_s_{0.0};
    size_t centroid_refresh_min_insertions_ = 500000; // 
    size_t inserted_since_centroid_refresh_ = 0;
    size_t coarse_check_every_insertions_ = 200000;   // 
    size_t inserted_since_coarse_check_ = 0;
    size_t coarse_rebuild_min_insertions_ = 300000;
    size_t inserted_since_coarse_rebuild_ = 0;
    bool debug_log_{false};
    std::vector<std::vector<float>> fine_centroids_graph_history_; // 上次建图时的质心快照

    // 选择率自适应双路由
    bool enable_adaptive_routing_{false};
    float adaptive_routing_high_threshold_{0.15f};  // 选择率 > 15% 视为"宽范围"，跳过图路由
    bool enable_coarse_first_training_{false};      // 训练消融：先 vector coarse，再 coarse 内 scalar-aware fine split
    int coarse_first_mode_{2};                      // 1=local scalar kmeans, 2=scalar quantile split

    std::unique_ptr<SimpleKMeans> local_kmeans_;
    
    // 方案1：自适应阈值
    struct GcHistory {
        std::deque<float> values;           // 最近 G_c 值
        size_t max_history = 10;             // 最多保存10个历史值
    } gc_history_;
    
    struct AdaptiveThresholdConfig {
        float base_tau = 0.6f;               // 更保守：减少 coarse 重建触发，提升插入吞吐
        float sensitivity = 3.0f;            // 更不敏感：需要更显著偏离历史才重建
        float min_tau = 0.4f;                // 抬高下限，避免冷启动阶段频繁重建
        float max_tau = 0.95f;               // 允许阈值抬到更高（更不易触发）
    } adaptive_config_;
    
    float coarse_soft_drift_tau_ = 0.06f;        // coarse 质心相对当前半径的软同步阈值
    float coarse_centroid_ema_alpha_ = 0.35f;    // soft M-step EMA，抑制边界抖动
    float fine_remap_rel_improvement_ = 0.12f;  // fine 迁移需带来的相对距离改善
    float fine_remap_abs_margin_ratio_ = 0.04f; // fine 迁移需超过 coarse 半径比例 margin
    float fine_remap_min_moved_ratio_ = 0.0f; // soft remap 后：任一 fine 迁移即刷新 fine graph（更积极）
    int coarse_soft_sync_count_ = 0;
    int coarse_soft_moved_fines_last_ = 0;
    int coarse_soft_moved_fines_total_ = 0;
    float coarse_soft_moved_ratio_last_ = 0.0f;
    float coarse_soft_rel_shift_last_ = 0.0f;
    float coarse_soft_avg_radius_last_ = 0.0f;

    int coarse_global_rebuild_count_ = 0;    // coarse层全局重建累计次数
    float l2_distance_sq(const float* a, const float* b) const {
        if (a == nullptr || b == nullptr || dimension_ <= 0) return 0.0f;
#if defined(__AVX__)
        const float* pa = a;
        const float* pb = b;
        int d = 0;
        __m256 sum256 = _mm256_setzero_ps();

        // 主路径：每次处理 8 维
        for (; d + 8 <= dimension_; d += 8) {
            __m256 va = _mm256_loadu_ps(pa + d);
            __m256 vb = _mm256_loadu_ps(pb + d);
            __m256 diff = _mm256_sub_ps(va, vb);//并行做 8 个差值。
#if defined(__FMA__)
            sum256 = _mm256_fmadd_ps(diff, diff, sum256);//一条融合指令完成
#else
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff, diff));//拆成 mul + add 两条指令
#endif
        }

        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        float dist = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(sum128, sum128), sum128));

        // 尾部标量处理
        for (; d < dimension_; ++d) {
            float diff = pa[d] - pb[d];
            dist += diff * diff;
        }
        return dist;
#else
        float dist = 0.0f;
        for (int d = 0; d < dimension_; ++d) {
            float diff = a[d] - b[d];
            dist += diff * diff;
        }
        return dist;
#endif
    }

    // [优化·问题4-D] beam 邻居方向判定 dot((nb-cur), (q-cur))，AVX 加速
    float centroid_alignment_dot(const float* cur_c, const float* nb_c, const float* query) const {
        if (cur_c == nullptr || nb_c == nullptr || query == nullptr || dimension_ <= 0) {
            return 0.0f;
        }
#if defined(__AVX__)
        int d = 0;
        __m256 sum256 = _mm256_setzero_ps();
        for (; d + 8 <= dimension_; d += 8) {
            __m256 v_nb = _mm256_loadu_ps(nb_c + d);
            __m256 v_cur = _mm256_loadu_ps(cur_c + d);
            __m256 v_q = _mm256_loadu_ps(query + d);
            __m256 diff_nb = _mm256_sub_ps(v_nb, v_cur);
            __m256 diff_q = _mm256_sub_ps(v_q, v_cur);
#if defined(__FMA__)
            sum256 = _mm256_fmadd_ps(diff_nb, diff_q, sum256);
#else
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff_nb, diff_q));
#endif
        }
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        float dot = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(sum128, sum128), sum128));
        for (; d < dimension_; ++d) {
            dot += (nb_c[d] - cur_c[d]) * (query[d] - cur_c[d]);
        }
        return dot;
#else
        float dot = 0.0f;
        for (int d = 0; d < dimension_; ++d) {
            dot += (nb_c[d] - cur_c[d]) * (query[d] - cur_c[d]);
        }
        return dot;
#endif
    }

    void build_fine_to_coarse_mapping(bool build_graph_seeds = true) {
        size_t n_fine = fine_centroids_.size();
        size_t n_coarse = coarse_centroids_.size();
        if (n_fine == 0 || n_coarse == 0) return;

        fine_to_coarse_.assign(n_fine, 0);
        coarse_to_fine_.assign(n_coarse, std::vector<int>());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t fid = 0; fid < n_fine; ++fid) {
            float best = std::numeric_limits<float>::max();
            int best_cid = 0;
            const float* fcent = fine_centroids_[fid].data();
            for (size_t cid = 0; cid < n_coarse; ++cid) {
                float dist = l2_distance_sq(fcent, coarse_centroids_[cid].data());
                if (dist < best) {
                    best = dist;
                    best_cid = static_cast<int>(cid);
                }
            }
            fine_to_coarse_[fid] = best_cid;
        }

        for (size_t fid = 0; fid < n_fine; ++fid) {
            coarse_to_fine_[static_cast<size_t>(fine_to_coarse_[fid])].push_back(static_cast<int>(fid));
        }

        if (build_graph_seeds) {
            rebuild_coarse_seed_fines();
        }
    }

    void rebuild_coarse_seed_fines() {
        const size_t n_fine = fine_centroids_.size();
        const size_t n_coarse = coarse_centroids_.size();
        coarse_seed_fines_.assign(n_coarse, {});
        if (n_fine == 0 || n_coarse == 0) return;

        const int L = std::max(1, seed_L_);
        const int pool_size = std::max(L, std::min(graph_degree_, L * 4));
        for (size_t cid = 0; cid < n_coarse; ++cid) {
            std::vector<std::pair<float, int>> tmp;
            tmp.reserve(coarse_to_fine_[cid].size());
            for (int fid : coarse_to_fine_[cid]) {
                if (fid < 0 || static_cast<size_t>(fid) >= n_fine) continue;
                float dsq = l2_distance_sq(fine_centroids_[static_cast<size_t>(fid)].data(),
                                           coarse_centroids_[cid].data());
                tmp.emplace_back(dsq, fid);
            }
            if (tmp.empty()) continue;
            const int take = std::min(pool_size, static_cast<int>(tmp.size()));
            std::partial_sort(tmp.begin(), tmp.begin() + take, tmp.end(),
                              [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first < b.first; });
            coarse_seed_fines_[cid].reserve(static_cast<size_t>(take));
            for (int i = 0; i < take; ++i) coarse_seed_fines_[cid].push_back(tmp[static_cast<size_t>(i)].second);
        }
    }

    void rebuild_coarse_neighbor_lists() {
        const size_t nc = coarse_centroids_.size();
        coarse_neighbor_coarses_.assign(nc, {});
        if (nc <= 1) {
            return;
        }
        const int top_r = std::max(1, graph_neighbor_coarses_);
        for (size_t c = 0; c < nc; ++c) {
            std::vector<std::pair<float, int>> tmp;
            tmp.reserve(nc - 1);
            for (size_t c2 = 0; c2 < nc; ++c2) {
                if (c2 == c) {
                    continue;
                }
                tmp.emplace_back(
                    l2_distance_sq(coarse_centroids_[c].data(), coarse_centroids_[c2].data()),
                    static_cast<int>(c2));
            }
            const int take = std::min(top_r, static_cast<int>(tmp.size()));
            if (take <= 0) {
                continue;
            }
            std::partial_sort(
                tmp.begin(), tmp.begin() + take, tmp.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first < b.first;
                });
            auto& out = coarse_neighbor_coarses_[c];
            out.reserve(static_cast<size_t>(take));
            for (int i = 0; i < take; ++i) {
                out.push_back(tmp[static_cast<size_t>(i)].second);
            }
        }
    }

    static void pick_top_k_neighbor_ids(
        std::vector<std::pair<float, int>>& dists,
        int k,
        std::vector<int>& out) {
        out.clear();
        if (dists.empty() || k <= 0) {
            return;
        }
        const int take = std::min(k, static_cast<int>(dists.size()));
        if (take < static_cast<int>(dists.size())) {
            std::partial_sort(
                dists.begin(), dists.begin() + take, dists.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first < b.first;
                });
            dists.resize(static_cast<size_t>(take));
        } else {
            std::sort(
                dists.begin(), dists.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first < b.first;
                });
        }
        out.reserve(static_cast<size_t>(take));
        for (int t = 0; t < take; ++t) {
            out.push_back(dists[static_cast<size_t>(t)].second);
        }
    }

    void rebuild_fine_graph_flat_storage() {
        fine_graph_offsets_.assign(fine_graph_.size() + 1, 0);
        size_t total = 0;
        for (size_t i = 0; i < fine_graph_.size(); ++i) {
            fine_graph_offsets_[i] = static_cast<int>(total);
            total += fine_graph_[i].size();
        }
        fine_graph_offsets_[fine_graph_.size()] = static_cast<int>(total);
        fine_graph_neighbors_.clear();
        fine_graph_neighbors_.reserve(total);
        for (const auto& nbrs : fine_graph_) {
            fine_graph_neighbors_.insert(fine_graph_neighbors_.end(), nbrs.begin(), nbrs.end());
        }
    }

    void rebuild_fine_centroid_graph(bool force) {
        const size_t n = fine_centroids_.size();
        if (!force && !fine_graph_.empty() && fine_graph_.size() == n) {
            return;
        }
        fine_graph_.assign(n, {});
        if (n == 0) {
            return;
        }
        const int deg = std::max(4, graph_degree_);

        // HIER_GRAPH_FULL_KNN=1 时回退全库 brute-force（消融/对照）
        const char* full_env = std::getenv("HIER_GRAPH_FULL_KNN");
        const bool use_full_knn =
            (full_env != nullptr) && (full_env[0] == '1' || full_env[0] == 'y' || full_env[0] == 'Y');

        if (use_full_knn) {
            if (debug_log_) {
                std::cerr << "[建图] 全量 O(n^2) kNN, n=" << n << ", deg=" << deg << std::endl;
            }
#pragma omp parallel
            {
                thread_local std::vector<std::pair<float, int>> dists_tls;
#pragma omp for schedule(static)
                for (long long i = 0; i < static_cast<long long>(n); ++i) {
                    auto& dists = dists_tls;
                    dists.clear();
                    const size_t ii = static_cast<size_t>(i);
                    const float* ci = fine_centroids_[ii].data();
                    if (n > 1) {
                        dists.reserve(n - 1);
                    }
                    for (size_t j = 0; j < n; ++j) {
                        if (j == ii) {
                            continue;
                        }
                        dists.emplace_back(l2_distance_sq(ci, fine_centroids_[j].data()), static_cast<int>(j));
                    }
                    pick_top_k_neighbor_ids(dists, deg, fine_graph_[ii]);
                }
            }
        } else {
            // 近似建图：簇内候选 + 全局 seed 桥接 + 邻近 coarse 的 fine（O(n·C_local) 量级）
            if (coarse_seed_fines_.size() != coarse_centroids_.size()) {
                rebuild_coarse_seed_fines();
            }
            rebuild_coarse_neighbor_lists();

            std::vector<int> global_seeds;
            global_seeds.reserve(coarse_seed_fines_.size() * static_cast<size_t>(seed_L_) + 8);
            for (const auto& seeds : coarse_seed_fines_) {
                for (int fid : seeds) {
                    global_seeds.push_back(fid);
                }
            }
            std::sort(global_seeds.begin(), global_seeds.end());
            global_seeds.erase(std::unique(global_seeds.begin(), global_seeds.end()), global_seeds.end());

            if (debug_log_) {
                std::cerr << "[建图] 局部+桥接 kNN, n=" << n << ", deg=" << deg
                          << ", neighbor_coarses=" << graph_neighbor_coarses_
                          << ", global_seeds=" << global_seeds.size() << std::endl;
            }

#pragma omp parallel
            {
                thread_local std::vector<int> candidate_ids_tls;
                thread_local std::vector<std::pair<float, int>> dists_tls;
#pragma omp for schedule(static)
                for (long long i = 0; i < static_cast<long long>(n); ++i) {
                    const size_t ii = static_cast<size_t>(i);
                    const int fid = static_cast<int>(ii);
                    auto& cands = candidate_ids_tls;
                    cands.clear();

                    const int my_coarse =
                        (ii < fine_to_coarse_.size()) ? fine_to_coarse_[ii] : 0;

                    // 1) 同 coarse 内全部 fine（局部连通）
                    if (my_coarse >= 0 && static_cast<size_t>(my_coarse) < coarse_to_fine_.size()) {
                        for (int j : coarse_to_fine_[static_cast<size_t>(my_coarse)]) {
                            if (j != fid) {
                                cands.push_back(j);
                            }
                        }
                    }

                    // 2) 全局 seed 桥接（跨 coarse 入口，允许跳到其它粗区代表点）
                    for (int s : global_seeds) {
                        if (s != fid) {
                            cands.push_back(s);
                        }
                    }

                    // 3) 质心最近的若干其它 coarse：纳入其全部 fine（跨区短程边）
                    if (my_coarse >= 0 && static_cast<size_t>(my_coarse) < coarse_neighbor_coarses_.size()) {
                        for (int nc : coarse_neighbor_coarses_[static_cast<size_t>(my_coarse)]) {
                            if (nc < 0 || static_cast<size_t>(nc) >= coarse_to_fine_.size()) {
                                continue;
                            }
                            for (int j : coarse_to_fine_[static_cast<size_t>(nc)]) {
                                if (j != fid) {
                                    cands.push_back(j);
                                }
                            }
                        }
                    }

                    std::sort(cands.begin(), cands.end());
                    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

                    auto& dists = dists_tls;
                    dists.clear();
                    dists.reserve(cands.size());
                    const float* ci = fine_centroids_[ii].data();
                    for (int j : cands) {
                        if (j < 0 || static_cast<size_t>(j) >= n) {
                            continue;
                        }
                        dists.emplace_back(
                            l2_distance_sq(ci, fine_centroids_[static_cast<size_t>(j)].data()), j);
                    }
                    pick_top_k_neighbor_ids(dists, deg, fine_graph_[ii]);
                }
            }
        }

        rebuild_fine_graph_flat_storage();
        fine_centroids_graph_history_ = fine_centroids_;
        inserted_since_graph_rebuild_ = 0;
    }

    float compute_graph_centroid_change_mean() const {
        if (fine_centroids_graph_history_.empty()) return std::numeric_limits<float>::infinity();
        const size_t n = std::min(fine_centroids_graph_history_.size(), fine_centroids_.size());
        if (n == 0) return 0.0f;
        double sum = 0.0;
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i) {
            if (fine_centroids_graph_history_[i].size() != fine_centroids_[i].size()) continue;
            double dsq = 0.0;
            for (int d = 0; d < dimension_; ++d) {
                const double diff = static_cast<double>(fine_centroids_[i][d]) -
                                    static_cast<double>(fine_centroids_graph_history_[i][d]);
                dsq += diff * diff;
            }
            sum += std::sqrt(dsq);
            ++cnt;
        }
        if (cnt == 0) return 0.0f;
        return static_cast<float>(sum / static_cast<double>(cnt));
    }

    // -------------------------------------------------------------------------
    // search_single：单条 query 的完整层次化检索
    // 阶段1 coarse：对 n_coarse 个粗质心算 L2²，取 top n_probe 粗簇
    // 阶段2 fine 路由（由 Module B / enable_graph_routing 二选一）：
    //   - B关：合并 top n_probe_c 粗簇内全部 fine，再全局 partial_sort 取 top (n_probe_c×n_probe_f)
    //   - B开：coarse seeds + fine 质心图 beam search，最终取 top (n_probe_c×n_probe_f)
    // 阶段3 列表扫描：AdaIVFCore::search_in_clusters
    // -------------------------------------------------------------------------
    std::vector<std::pair<float, int>> search_single(
        const float* query,
        int k,
        int n_probe,
        int budget,
        float range_min,
        float range_max,
        QuerySearchScratch& scratch,
        const std::vector<std::vector<float>>& route_centroids,
        AdaIVFCore::SearchStats* stats = nullptr) const {
        if (!is_trained_ || fine_index_->get_n_vectors() == 0) {
            return {};
        }

        scratch.ensure_size(n_fine_clusters_, n_coarse_clusters_);
        const uint32_t gen = scratch.bump_generation();
        const size_t n_fine_sz = static_cast<size_t>(n_fine_clusters_);
        const size_t route_n = route_centroids.size();

        auto dist_q_fid = [&](int fid) -> float {
            if (fid < 0 || static_cast<size_t>(fid) >= n_fine_sz) {
                return std::numeric_limits<float>::max();
            }
            const size_t idx = static_cast<size_t>(fid);
            // [优化·问题3-B/D] 世代标记 + 单数组缓存，去掉 dist_cached 与每 query 清零
            if (scratch.dist_stamp[idx] == gen) {
                return scratch.dist_cache[idx];
            }
            const float* cent = (idx < route_n && !route_centroids[idx].empty())
                                    ? route_centroids[idx].data()
                                    : fine_centroids_[idx].data();
            float dsq = l2_distance_sq(query, cent);
            scratch.dist_cache[idx] = dsq;
            scratch.dist_stamp[idx] = gen;
            return dsq;
        };

        bool use_filter_aware = enable_filter_aware_routing_ &&
            std::isfinite(range_min) && std::isfinite(range_max) &&
            range_min > -1e9f && range_max < 1e9f && range_min <= range_max;
        // Query-level effectiveness gate. Min/max overlap is often too coarse on
        // vector-trained clusters: when scalar contrast is weak, keep the graph
        // path purely geometric and avoid extra utility lookups/reranking.
        bool scalar_route_effective = use_filter_aware;
        bool scalar_cluster_prune_effective = use_filter_aware;

        auto selectivity_fid = [&](int fid) -> float {
            if (!use_filter_aware) {
                return 1.0f;
            }
            if (fid < 0 || static_cast<size_t>(fid) >= n_fine_sz) {
                return 0.0f;
            }
            const size_t idx = static_cast<size_t>(fid);
            if (scratch.sel_stamp[idx] == gen) {
                return scratch.sel_cache[idx];
            }
            float sel = fine_index_->estimate_cluster_selectivity(fid, range_min, range_max);
            if (sel <= 0.0f && fine_index_->cluster_may_match_range(fid, range_min, range_max)) {
                // Histogram is only a routing estimate. Keep a small overlap prior so
                // sparse-bin false zeros do not demote bridge/nearby clusters to pure fallback.
                sel = 0.005f;
            }
            if (sel < 0.0f) sel = 0.0f;
            if (sel > 1.0f) sel = 1.0f;
            scratch.sel_cache[idx] = sel;
            scratch.sel_stamp[idx] = gen;
            return sel;
        };

        auto scalar_utility_fid = [&](int fid) -> float {
            if (!use_filter_aware) {
                return 1.0f;
            }
            if (fid < 0 || static_cast<size_t>(fid) >= n_fine_sz) {
                return 0.0f;
            }
            const size_t idx = static_cast<size_t>(fid);
            if (scratch.scalar_utility_stamp[idx] == gen) {
                return scratch.scalar_utility_cache[idx];
            }
            float utility = fine_index_->estimate_cluster_scalar_utility(fid, range_min, range_max);
            if (utility <= 0.0f && fine_index_->cluster_may_match_range(fid, range_min, range_max)) {
                utility = 0.005f;
            }
            if (utility < 0.0f) utility = 0.0f;
            if (utility > 1.0f) utility = 1.0f;
            scratch.scalar_utility_cache[idx] = utility;
            scratch.scalar_utility_stamp[idx] = gen;
            return utility;
        };

        auto route_score_fid = [&](int fid) -> float {
            const float dsq = dist_q_fid(fid);
            if (!use_filter_aware) {
                return dsq;
            }
            if (!scalar_route_effective) {
                return dsq;
            }
            const float utility = scalar_utility_fid(fid);
            // Scalar is only a weak tie-breaker. The previous small base
            // denominator could turn utility into a dominant global metric and
            // demote geometrically close clusters.
            const float boost = 1.0f + 0.25f * filter_selectivity_epsilon_ * utility;
            return dsq / boost;
        };

        auto scalar_may_match_fid = [&](int fid) -> bool {
            if (!use_filter_aware) return true;
            return fine_index_->cluster_may_match_range(fid, range_min, range_max);
        };

        auto gate_filter_aware_for_candidates = [&](const std::vector<int>& ids) {
            if (!use_filter_aware || ids.empty()) return;
            const int sample_n = std::min(64, static_cast<int>(ids.size()));
            const int stride = std::max(1, static_cast<int>(ids.size()) / std::max(1, sample_n));
            int sampled = 0;
            int overlap_count = 0;
            int positive_count = 0;
            float sel_sum = 0.0f;
            float sel_max = 0.0f;
            for (size_t pos = 0; pos < ids.size() && sampled < sample_n; pos += static_cast<size_t>(stride)) {
                const int fid = ids[pos];
                if (fid < 0 || fid >= n_fine_clusters_) continue;
                const float sel = scalar_utility_fid(fid);
                const bool overlap = sel > 0.0f;
                overlap_count += overlap ? 1 : 0;
                positive_count += sel > 0.0051f ? 1 : 0;
                sel_sum += sel;
                sel_max = std::max(sel_max, sel);
                ++sampled;
            }
            if (sampled <= 0) return;
            const float overlap_ratio = static_cast<float>(overlap_count) / static_cast<float>(sampled);
            const float positive_ratio = static_cast<float>(positive_count) / static_cast<float>(sampled);
            const float avg_sel = sel_sum / static_cast<float>(sampled);
            const float contrast = sel_max / std::max(1e-6f, avg_sel);
            if (overlap_ratio > 0.85f || positive_ratio > 0.65f ||
                (overlap_ratio > 0.70f && contrast < 1.30f)) {
                use_filter_aware = false;
                scalar_route_effective = false;
                scalar_cluster_prune_effective = false;
            } else {
                scalar_cluster_prune_effective = overlap_ratio <= 0.80f;
            }
        };

        auto is_visited = [&](int fid) -> bool {
            if (fid < 0 || static_cast<size_t>(fid) >= n_fine_sz) return true;
            return scratch.visit_stamp[static_cast<size_t>(fid)] == gen;
        };
        auto mark_visited = [&](int fid) {
            if (fid < 0 || static_cast<size_t>(fid) >= n_fine_sz) return;
            scratch.visit_stamp[static_cast<size_t>(fid)] = gen;
        };

        // 1) coarse：选 top n_probe 粗簇（与单层共用 partial_sort top-k）
        auto& coarse_scores = scratch.coarse_scores;
        for (int cid = 0; cid < n_coarse_clusters_; ++cid) {
            coarse_scores[static_cast<size_t>(cid)] =
                l2_distance_sq(query, coarse_centroids_[static_cast<size_t>(cid)].data());
        }

        auto& coarse_top_ids = scratch.coarse_order;
        const int base_coarse_top_k = std::min(n_probe, n_coarse_clusters_);
        int coarse_top_k = base_coarse_top_k;
        ivf_topk::select_smallest_k_indices(coarse_scores, n_coarse_clusters_, coarse_top_k, coarse_top_ids);

        const int current_n_probe_f_raw = n_probe_f_.load();
        const int per_coarse_f =
            (current_n_probe_f_raw > 0) ? current_n_probe_f_raw : n_probe_f_default_;
        if (enable_adaptive_fanout_ && base_coarse_top_k > 0 && base_coarse_top_k < n_coarse_clusters_) {
            float kth_score = 0.0f;
            float next_score = std::numeric_limits<float>::max();
            for (int i = 0; i < base_coarse_top_k; ++i) {
                kth_score = std::max(kth_score, coarse_scores[static_cast<size_t>(coarse_top_ids[static_cast<size_t>(i)])]);
            }
            for (int cid = 0; cid < n_coarse_clusters_; ++cid) {
                bool selected = false;
                for (int i = 0; i < base_coarse_top_k; ++i) {
                    if (coarse_top_ids[static_cast<size_t>(i)] == cid) {
                        selected = true;
                        break;
                    }
                }
                if (!selected) {
                    next_score = std::min(next_score, coarse_scores[static_cast<size_t>(cid)]);
                }
            }
            const float denom = std::max(1e-6f, std::max(1.0f, kth_score));
            const bool boundary_risk =
                std::isfinite(next_score) &&
                (next_score - kth_score) <= adaptive_fanout_boundary_margin_ratio_ * denom;
            const size_t cur_vectors = fine_index_->get_n_vectors();
            const bool late_risk =
                cur_vectors >= adaptive_fanout_late_vectors_ ||
                coarse_soft_sync_count_ >= 24 ||
                coarse_soft_rel_shift_last_ >= coarse_soft_drift_tau_ * 0.75f;
            int extra_coarse = 0;
            if (late_risk && boundary_risk) {
                extra_coarse = adaptive_fanout_boundary_extra_coarse_;
                if (cur_vectors >= adaptive_fanout_deep_late_vectors_) {
                    extra_coarse = std::max(extra_coarse, adaptive_fanout_deep_late_extra_coarse_);
                }
                if (cur_vectors >= adaptive_fanout_tail_vectors_) {
                    extra_coarse = std::max(extra_coarse, adaptive_fanout_tail_extra_coarse_);
                }
            }
            if (extra_coarse > 0) {
                const int new_coarse_top_k = std::min(n_coarse_clusters_, base_coarse_top_k + extra_coarse);
                if (new_coarse_top_k > coarse_top_k) {
                    coarse_top_k = new_coarse_top_k;
                    ivf_topk::select_smallest_k_indices(
                        coarse_scores, n_coarse_clusters_, coarse_top_k, coarse_top_ids);
                    adaptive_fanout_query_count_.fetch_add(1, std::memory_order_relaxed);
                    adaptive_fanout_extra_coarse_total_.fetch_add(
                        static_cast<size_t>(coarse_top_k - base_coarse_top_k), std::memory_order_relaxed);
                    adaptive_fanout_extra_target_total_.fetch_add(
                        static_cast<size_t>(coarse_top_k - base_coarse_top_k) *
                        static_cast<size_t>(std::max(1, per_coarse_f)),
                        std::memory_order_relaxed);
                }
            }
        }
        const int* coarse_top_ptr = coarse_top_ids.data();
        // 最终列表扫描预算 = n_probe_c × n_probe_f（与单层 n_probe 对齐，如 8×10=80）
        const int target = std::min(
            std::max(1, coarse_top_k * per_coarse_f),
            n_fine_clusters_);

        // 自适应路由：只用便宜的 min/max overlap 做早期宽范围判断。
        // 旧实现会在这里额外估计 histogram selectivity，和后续 filter-aware gate
        // 重复计算；对高 overlap 查询尤其容易纯增开销。
        bool force_flat_routing = false;
        if (enable_adaptive_routing_ && enable_graph_routing_.load() && use_filter_aware) {
            int overlap_cnt = 0;
            int sample_cnt = 0;
            const int sample_limit = std::min(std::max(4, coarse_top_k * 2), 16);
            for (int i = 0; i < coarse_top_k && sample_cnt < sample_limit; ++i) {
                const int cid = coarse_top_ptr[i];
                if (cid < 0 || static_cast<size_t>(cid) >= coarse_to_fine_.size()) continue;
                for (int fid : coarse_to_fine_[static_cast<size_t>(cid)]) {
                    if (fid < 0 || fid >= n_fine_clusters_) continue;
                    overlap_cnt += scalar_may_match_fid(fid) ? 1 : 0;
                    ++sample_cnt;
                    if (sample_cnt >= sample_limit) break;
                }
            }
            if (sample_cnt > 0) {
                const float overlap_ratio = static_cast<float>(overlap_cnt) / static_cast<float>(sample_cnt);
                if (overlap_ratio > 0.90f) {
                    force_flat_routing = true;
                    use_filter_aware = false;
                    scalar_route_effective = false;
                    scalar_cluster_prune_effective = false;
                }
            }
        }

        // 2) fine 路由：B关 union+全局 top-(n_probe_c×n_probe_f)；B开图 beam search
        auto& candidate_fine_ids = scratch.candidate_fine_ids;
        candidate_fine_ids.clear();
        if (!enable_graph_routing_.load() || force_flat_routing) {
            // B 关：合并 top n_probe 粗簇内 fine id（先收集 union，再按需算质心距）
            candidate_fine_ids.reserve(static_cast<size_t>(target) * 8);
            for (int i = 0; i < coarse_top_k; ++i) {
                const int cid = coarse_top_ptr[i];
                if (cid < 0 || static_cast<size_t>(cid) >= coarse_to_fine_.size()) {
                    continue;
                }
                for (int fid : coarse_to_fine_[static_cast<size_t>(cid)]) {
                    if (fid < 0 || fid >= n_fine_clusters_) {
                        continue;
                    }
                    candidate_fine_ids.push_back(fid);
                }
            }
            if (!candidate_fine_ids.empty()) {
                std::sort(candidate_fine_ids.begin(), candidate_fine_ids.end());
                candidate_fine_ids.erase(
                    std::unique(candidate_fine_ids.begin(), candidate_fine_ids.end()),
                    candidate_fine_ids.end());

                gate_filter_aware_for_candidates(candidate_fine_ids);
                if (use_filter_aware) {
                    std::vector<int> filtered;
                    filtered.reserve(candidate_fine_ids.size());
                    size_t route_pruned = 0;
                    for (int fid : candidate_fine_ids) {
                        if (scalar_may_match_fid(fid)) {
                            filtered.push_back(fid);
                        } else {
                            ++route_pruned;
                        }
                    }
                    const size_t original_count = candidate_fine_ids.size();
                    const bool low_prune_yield = (route_pruned * 20 < original_count);
                    if (low_prune_yield) {
                        scalar_route_effective = false;
                        const int take = std::min(target, static_cast<int>(candidate_fine_ids.size()));
                        if (take < static_cast<int>(candidate_fine_ids.size())) {
                            fine_index_->write_cluster_dist_sq_for_ids(
                                query, candidate_fine_ids, scratch.dist_cache);
                            for (int fid : candidate_fine_ids) {
                                if (fid >= 0 && fid < n_fine_clusters_) {
                                    scratch.dist_stamp[static_cast<size_t>(fid)] = gen;
                                }
                            }
                            ivf_topk::select_smallest_k_inplace(
                                scratch.dist_cache, candidate_fine_ids, take);
                            candidate_fine_ids.resize(static_cast<size_t>(take));
                        }
                    } else {
                        if (stats) stats->num_pruned_clusters += route_pruned;
                        if (filtered.empty()) {
                            candidate_fine_ids.clear();
                        } else {
                            const int take = std::min(target, static_cast<int>(filtered.size()));
                            if (take < static_cast<int>(filtered.size())) {
                                ivf_topk::select_smallest_k_lazy_inplace(filtered, take, route_score_fid);
                                filtered.resize(static_cast<size_t>(take));
                            }
                            candidate_fine_ids.swap(filtered);
                        }
                    }
                } else {
                    const int take = std::min(target, static_cast<int>(candidate_fine_ids.size()));
                    if (take < static_cast<int>(candidate_fine_ids.size())) {
                        constexpr int kLazyUnionFactor = 2;
                        const bool use_lazy = static_cast<int>(candidate_fine_ids.size()) > take * kLazyUnionFactor;
                        if (use_lazy) {
                            ivf_topk::select_smallest_k_lazy_inplace(
                                candidate_fine_ids, take, route_score_fid);
                        } else {
                            fine_index_->write_cluster_dist_sq_for_ids(
                                query, candidate_fine_ids, scratch.dist_cache);
                            for (int fid : candidate_fine_ids) {
                                if (fid >= 0 && fid < n_fine_clusters_) {
                                    scratch.dist_stamp[static_cast<size_t>(fid)] = gen;
                                }
                            }
                            ivf_topk::select_smallest_k_inplace(
                                scratch.dist_cache, candidate_fine_ids, take);
                        }
                        candidate_fine_ids.resize(static_cast<size_t>(take));
                    }
                }
            }
        } else {
            // B 开：基础双层候选 + coarse 外优先的残差图修复。
            // TopPC 内 fine 只构成基础候选和少量过桥节点，增量结果只接收 TopPC 外 fine。
            for (int i = 0; i < coarse_top_k; ++i) {
                const int cid = coarse_top_ptr[i];
                if (cid >= 0 && cid < n_coarse_clusters_) {
                    scratch.coarse_selected_stamp[static_cast<size_t>(cid)] = gen;
                }
            }
            auto is_selected_coarse = [&](int cid) -> bool {
                return cid >= 0 && cid < n_coarse_clusters_ &&
                    scratch.coarse_selected_stamp[static_cast<size_t>(cid)] == gen;
            };
            auto is_external_fine = [&](int fid) -> bool {
                if (fid < 0 || static_cast<size_t>(fid) >= fine_to_coarse_.size()) return false;
                return !is_selected_coarse(fine_to_coarse_[static_cast<size_t>(fid)]);
            };

            candidate_fine_ids.reserve(static_cast<size_t>(target) * 8);
            for (int i = 0; i < coarse_top_k; ++i) {
                const int cid = coarse_top_ptr[i];
                if (cid < 0 || static_cast<size_t>(cid) >= coarse_to_fine_.size()) continue;
                for (int fid : coarse_to_fine_[static_cast<size_t>(cid)]) {
                    if (fid >= 0 && fid < n_fine_clusters_) {
                        candidate_fine_ids.push_back(fid);
                    }
                }
            }
            gate_filter_aware_for_candidates(candidate_fine_ids);
            // Keep graph traversal geometric by default. The scan layer already
            // performs exact min/max range pruning, so scalar-guided beam order is
            // worth its utility lookups only when the sampled candidate set shows
            // both substantial no-overlap mass and strong utility contrast.
            bool graph_use_scalar_navigation = false;
            if (use_filter_aware && !candidate_fine_ids.empty()) {
                const int sample_n = std::min(32, static_cast<int>(candidate_fine_ids.size()));
                const int stride = std::max(1, static_cast<int>(candidate_fine_ids.size()) / sample_n);
                int sampled = 0;
                int overlap_count = 0;
                float utility_sum = 0.0f;
                float utility_max = 0.0f;
                for (size_t pos = 0; pos < candidate_fine_ids.size() && sampled < sample_n; pos += static_cast<size_t>(stride)) {
                    const int fid = candidate_fine_ids[pos];
                    if (fid < 0 || fid >= n_fine_clusters_) continue;
                    const bool overlap = scalar_may_match_fid(fid);
                    const float utility = overlap ? scalar_utility_fid(fid) : 0.0f;
                    overlap_count += overlap ? 1 : 0;
                    utility_sum += utility;
                    utility_max = std::max(utility_max, utility);
                    ++sampled;
                }
                if (sampled > 0) {
                    const float overlap_ratio = static_cast<float>(overlap_count) / static_cast<float>(sampled);
                    const float avg_utility = utility_sum / static_cast<float>(sampled);
                    const float contrast = utility_max / std::max(1e-6f, avg_utility);
                    const float non_overlap_ratio = 1.0f - overlap_ratio;
                    graph_use_scalar_navigation =
                        non_overlap_ratio >= 0.60f &&
                        contrast >= 2.00f &&
                        utility_max >= 0.02f;
                }
            }
            scalar_route_effective = graph_use_scalar_navigation;
            auto graph_score_fid = [&](int fid) -> float {
                // Graph navigation itself remains geometry-primary. Scalar is applied only
                // as local seed/neighbor tie-break and scan scheduling, not as a global
                // distance divisor for graph result ownership.
                (void)graph_use_scalar_navigation;
                return dist_q_fid(fid);
            };

            if (!candidate_fine_ids.empty() && static_cast<int>(candidate_fine_ids.size()) > target) {
                ivf_topk::select_smallest_k_lazy_inplace(
                    candidate_fine_ids, target, graph_score_fid);
                candidate_fine_ids.resize(static_cast<size_t>(target));
            }

            const int auto_repair = std::max(seed_L_ * 2, (target + 3) / 4);
            const int repair_budget = std::min(
                target, graph_repair_budget_ > 0 ? graph_repair_budget_ : auto_repair);
            const int max_expand = std::max(1, repair_budget + early_stop_patience_);
            const int eff_beam = std::max(1, std::min(beam_width_, graph_degree_));

            auto scalar_positive_fid = [&](int fid) -> bool {
                if (!use_filter_aware) return true;
                if (!graph_use_scalar_navigation) return true;
                return scalar_utility_fid(fid) > 0.0f;
            };

            // 扩展优先级：几何分数 + neighbor utility gradient（相对父节点的 utility 提升）。
            // 分数越小越优先；仅改变 traversal 顺序，不硬剪枝阴性邻居。
            auto expand_priority = [&](int parent_fid, int nb_fid) -> float {
                const float base = dist_q_fid(nb_fid);
                if (!graph_use_scalar_navigation) {
                    return base;
                }
                const float u_nb = scalar_utility_fid(nb_fid);
                const float u_parent = (parent_fid >= 0) ? scalar_utility_fid(parent_fid) : 0.0f;
                const float gradient = std::max(0.0f, u_nb - u_parent);
                // Weak local tie-break: scalar can adjust neighbors within the vector margin,
                // but cannot globally replace the geometric graph metric.
                const float boost = 1.0f + 0.20f * u_nb + 0.35f * gradient;
                return base / boost;
            };
            constexpr float kScalarGradientVectorMargin = 1.15f;
            auto guarded_expand_priority = [&](int parent_fid, int nb_fid, float best_vec_dsq) -> float {
                const float vec_dsq = dist_q_fid(nb_fid);
                if (!graph_use_scalar_navigation || !std::isfinite(best_vec_dsq) ||
                    vec_dsq > best_vec_dsq * kScalarGradientVectorMargin) {
                    return vec_dsq;
                }
                return expand_priority(parent_fid, nb_fid);
            };

            // 每个 coarse 离线保存较大的入口池；查询时用 scalar-aware score 重排，
            // 并做 coarse 轮询多样性，避免 seeds 全挤在同一 coarse。
            auto& seeds = scratch.graph_seeds;
            seeds.clear();
            seeds.reserve(static_cast<size_t>(coarse_top_k) * static_cast<size_t>(seed_L_) + 8);
            auto& seed_nodes = scratch.graph_top_nodes;
            std::vector<std::vector<int>> per_coarse_seeds(
                static_cast<size_t>(std::max(0, coarse_top_k)));
            for (int i = 0; i < coarse_top_k; ++i) {
                const int cid = coarse_top_ptr[i];
                if (cid < 0 || static_cast<size_t>(cid) >= coarse_seed_fines_.size()) continue;
                seed_nodes.clear();
                for (int fid : coarse_seed_fines_[static_cast<size_t>(cid)]) {
                    if (fid < 0 || fid >= n_fine_clusters_) continue;
                    seed_nodes.push_back(GraphBeamNode{dist_q_fid(fid), fid});
                }
                const int take = std::min(seed_L_, static_cast<int>(seed_nodes.size()));
                if (take <= 0) continue;
                if (take < static_cast<int>(seed_nodes.size())) {
                    std::partial_sort(
                        seed_nodes.begin(), seed_nodes.begin() + take, seed_nodes.end(),
                        [](const GraphBeamNode& a, const GraphBeamNode& b) {
                            return a.dsq < b.dsq;
                        });
                } else {
                    std::sort(seed_nodes.begin(), seed_nodes.end(),
                              [](const GraphBeamNode& a, const GraphBeamNode& b) {
                                  return a.dsq < b.dsq;
                              });
                }
                auto& bucket = per_coarse_seeds[static_cast<size_t>(i)];
                bucket.reserve(static_cast<size_t>(take));
                for (int j = 0; j < take; ++j) {
                    const int fid = seed_nodes[static_cast<size_t>(j)].fid;
                    bucket.push_back(fid);
                    scratch.protected_seed_stamp[static_cast<size_t>(fid)] = gen;
                }
            }
            // round-robin 跨 coarse 取 seed，保证 frontier 起点多样性
            {
                size_t max_len = 0;
                for (const auto& b : per_coarse_seeds) {
                    max_len = std::max(max_len, b.size());
                }
                for (size_t round = 0; round < max_len; ++round) {
                    for (const auto& b : per_coarse_seeds) {
                        if (round < b.size()) {
                            seeds.push_back(b[round]);
                        }
                    }
                }
            }
            if (seeds.empty()) {
                const int take = std::min(seed_L_, static_cast<int>(candidate_fine_ids.size()));
                seed_nodes.clear();
                for (int fid : candidate_fine_ids) {
                    seed_nodes.push_back(GraphBeamNode{graph_score_fid(fid), fid});
                }
                if (take > 0 && take < static_cast<int>(seed_nodes.size())) {
                    std::partial_sort(
                        seed_nodes.begin(), seed_nodes.begin() + take, seed_nodes.end(),
                        [](const GraphBeamNode& a, const GraphBeamNode& b) {
                            return a.dsq < b.dsq;
                        });
                }
                for (int j = 0; j < take; ++j) seeds.push_back(seed_nodes[static_cast<size_t>(j)].fid);
            }
            std::sort(seeds.begin(), seeds.end());
            seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

            struct MinFrontierCmp {
                bool operator()(const GraphBeamNode& a, const GraphBeamNode& b) const {
                    return a.dsq > b.dsq;
                }
            };
            struct MaxResultCmp {
                bool operator()(const GraphBeamNode& a, const GraphBeamNode& b) const {
                    return a.dsq < b.dsq;
                }
            };

            // 固定容量 max-heap 保存最终 target 个簇，基础候选提供 recall 下界。
            auto& top_nodes = scratch.graph_top_nodes;
            top_nodes.clear();
            top_nodes.reserve(static_cast<size_t>(target) + 1);
            for (int fid : candidate_fine_ids) {
                const float score = graph_score_fid(fid);
                top_nodes.push_back(GraphBeamNode{score, fid});
                scratch.top_stamp[static_cast<size_t>(fid)] = gen;
            }
            std::make_heap(top_nodes.begin(), top_nodes.end(), MaxResultCmp{});

            auto consider_result = [&](int fid, float score) {
                if (fid < 0 || fid >= n_fine_clusters_) return false;
                const size_t idx = static_cast<size_t>(fid);
                if (scratch.top_stamp[idx] == gen) return false;
                if (static_cast<int>(top_nodes.size()) < target) {
                    top_nodes.push_back(GraphBeamNode{score, fid});
                    scratch.top_stamp[idx] = gen;
                    std::push_heap(top_nodes.begin(), top_nodes.end(), MaxResultCmp{});
                    return true;
                }
                if (top_nodes.empty() || score >= top_nodes.front().dsq) return false;
                std::pop_heap(top_nodes.begin(), top_nodes.end(), MaxResultCmp{});
                const int evicted = top_nodes.back().fid;
                if (evicted >= 0 && evicted < n_fine_clusters_) {
                    scratch.top_stamp[static_cast<size_t>(evicted)] = 0u;
                }
                top_nodes.back() = GraphBeamNode{score, fid};
                scratch.top_stamp[idx] = gen;
                std::push_heap(top_nodes.begin(), top_nodes.end(), MaxResultCmp{});
                return true;
            };

            auto& predicate_frontier = scratch.graph_beam_nodes;
            auto& fallback_frontier = scratch.graph_fallback_nodes;
            predicate_frontier.clear();
            fallback_frontier.clear();
            const int frontier_cap = std::max(
                32, static_cast<int>(seeds.size()) + repair_budget * 2 + eff_beam * 2);
            predicate_frontier.reserve(static_cast<size_t>(frontier_cap * 2));
            fallback_frontier.reserve(static_cast<size_t>(frontier_cap * 2));

            // frontier diversity：限制同一 coarse 在 frontier 中的占比，避免局部塌缩。
            std::vector<int> frontier_coarse_count(
                static_cast<size_t>(std::max(1, n_coarse_clusters_)), 0);
            const int max_per_coarse_frontier = std::max(
                2, (frontier_cap + std::max(1, coarse_top_k) - 1) / std::max(1, coarse_top_k) + 1);

            auto push_frontier_node = [&](int fid, float priority_score) {
                if (fid < 0 || fid >= n_fine_clusters_ || is_visited(fid)) return false;
                const int cid = (static_cast<size_t>(fid) < fine_to_coarse_.size())
                                    ? fine_to_coarse_[static_cast<size_t>(fid)]
                                    : -1;
                if (cid >= 0 && cid < n_coarse_clusters_ &&
                    frontier_coarse_count[static_cast<size_t>(cid)] >= max_per_coarse_frontier) {
                    // 多样性门控：该 coarse 已饱和时，仅允许高 utility 外部节点挤入
                    if (!(graph_use_scalar_navigation && is_external_fine(fid) &&
                          scalar_utility_fid(fid) >= 0.35f)) {
                        return false;
                    }
                }
                mark_visited(fid);
                if (cid >= 0 && cid < n_coarse_clusters_) {
                    frontier_coarse_count[static_cast<size_t>(cid)] += 1;
                }
                if (scalar_positive_fid(fid)) {
                    predicate_frontier.push_back(GraphBeamNode{priority_score, fid});
                    std::push_heap(predicate_frontier.begin(), predicate_frontier.end(), MinFrontierCmp{});
                } else {
                    // fallback 仍用几何距离作桥，但入队优先级可带 gradient
                    fallback_frontier.push_back(GraphBeamNode{priority_score, fid});
                    std::push_heap(fallback_frontier.begin(), fallback_frontier.end(), MinFrontierCmp{});
                }
                return true;
            };
            for (int fid : seeds) {
                const float seed_priority =
                    (fid >= 0 && static_cast<size_t>(fid) < scratch.protected_seed_stamp.size() &&
                     scratch.protected_seed_stamp[static_cast<size_t>(fid)] == gen)
                        ? dist_q_fid(fid)
                        : guarded_expand_priority(-1, fid, dist_q_fid(fid));
                push_frontier_node(fid, seed_priority);
            }

            auto trim_one_frontier = [&](std::vector<GraphBeamNode>& frontier) {
                if (static_cast<int>(frontier.size()) <= frontier_cap * 2) return;
                std::nth_element(
                    frontier.begin(), frontier.begin() + frontier_cap, frontier.end(),
                    [](const GraphBeamNode& a, const GraphBeamNode& b) {
                        return a.dsq < b.dsq;
                    });
                // 重建 coarse 计数（trim 后）
                for (size_t i = static_cast<size_t>(frontier_cap); i < frontier.size(); ++i) {
                    const int fid = frontier[i].fid;
                    if (fid < 0 || static_cast<size_t>(fid) >= fine_to_coarse_.size()) continue;
                    const int cid = fine_to_coarse_[static_cast<size_t>(fid)];
                    if (cid >= 0 && cid < n_coarse_clusters_ &&
                        frontier_coarse_count[static_cast<size_t>(cid)] > 0) {
                        frontier_coarse_count[static_cast<size_t>(cid)] -= 1;
                    }
                }
                frontier.resize(static_cast<size_t>(frontier_cap));
                std::make_heap(frontier.begin(), frontier.end(), MinFrontierCmp{});
            };

            const bool graph_ok =
                fine_graph_offsets_.size() == fine_centroids_.size() + 1 &&
                !fine_graph_neighbors_.empty();
            int external_expanded = 0;
            int gateway_expanded = 0;
            int no_improve = 0;
            int total_expanded = 0;
            const int gateway_budget = std::max(2, std::min(
                static_cast<int>(seeds.size()), coarse_top_k * std::max(1, seed_L_ / 2)));
            const int internal_gateway_width = 2;
            // 邻居候选缓冲：按 utility gradient 重排后再取 beam，而不是按图边存储顺序。
            std::vector<std::pair<float, int>> nb_rank;
            nb_rank.reserve(static_cast<size_t>(std::max(8, graph_degree_)));

            while ((!predicate_frontier.empty() || !fallback_frontier.empty()) &&
                   (external_expanded < max_expand || gateway_expanded < gateway_budget)) {
                const bool prefer_fallback =
                    !fallback_frontier.empty() &&
                    (predicate_frontier.empty() || ((total_expanded + 1) % 5 == 0));
                auto& active_frontier = prefer_fallback ? fallback_frontier : predicate_frontier;
                std::pop_heap(active_frontier.begin(), active_frontier.end(), MinFrontierCmp{});
                const GraphBeamNode cur = active_frontier.back();
                active_frontier.pop_back();
                {
                    const int cid = (static_cast<size_t>(cur.fid) < fine_to_coarse_.size())
                                        ? fine_to_coarse_[static_cast<size_t>(cur.fid)]
                                        : -1;
                    if (cid >= 0 && cid < n_coarse_clusters_ &&
                        frontier_coarse_count[static_cast<size_t>(cid)] > 0) {
                        frontier_coarse_count[static_cast<size_t>(cid)] -= 1;
                    }
                }
                ++total_expanded;
                if (stats) {
                    stats->graph_nodes_visited += 1;
                    if (scalar_positive_fid(cur.fid)) {
                        stats->graph_scalar_positive_nodes += 1;
                    } else {
                        stats->graph_scalar_negative_nodes += 1;
                    }
                    if (prefer_fallback) {
                        stats->graph_fallback_expansions += 1;
                    }
                }

                const float result_score = graph_score_fid(cur.fid);
                const bool cur_external = is_external_fine(cur.fid);
                bool improved = false;
                if (cur_external) {
                    ++external_expanded;
                    improved = consider_result(cur.fid, result_score);
                    if (improved) no_improve = 0;
                    else ++no_improve;
                    if (no_improve >= early_stop_patience_ &&
                        external_expanded >= repair_budget) break;
                } else {
                    // TopPC 内节点只作为通往边界外的 gateway，不参与增量替换。
                    ++gateway_expanded;
                    if (gateway_expanded > gateway_budget) continue;
                }
                if (!graph_ok) break;

                const int begin = fine_graph_offsets_[static_cast<size_t>(cur.fid)];
                const int finish = fine_graph_offsets_[static_cast<size_t>(cur.fid) + 1];

                // 第一遍：收集 TopPC 外邻居，按 expand_priority（utility gradient）排序后取 beam。
                nb_rank.clear();
                float best_external_vec = std::numeric_limits<float>::max();
                for (int pos = begin; pos < finish; ++pos) {
                    const int nb_fid = fine_graph_neighbors_[static_cast<size_t>(pos)];
                    if (nb_fid < 0 || nb_fid >= n_fine_clusters_ ||
                        is_visited(nb_fid) || !is_external_fine(nb_fid)) continue;
                    best_external_vec = std::min(best_external_vec, dist_q_fid(nb_fid));
                }
                for (int pos = begin; pos < finish; ++pos) {
                    const int nb_fid = fine_graph_neighbors_[static_cast<size_t>(pos)];
                    if (nb_fid < 0 || nb_fid >= n_fine_clusters_ ||
                        is_visited(nb_fid) || !is_external_fine(nb_fid)) continue;
                    nb_rank.emplace_back(guarded_expand_priority(cur.fid, nb_fid, best_external_vec), nb_fid);
                }
                if (static_cast<int>(nb_rank.size()) > eff_beam) {
                    std::partial_sort(
                        nb_rank.begin(), nb_rank.begin() + eff_beam, nb_rank.end(),
                        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                            return a.first < b.first;
                        });
                    nb_rank.resize(static_cast<size_t>(eff_beam));
                } else {
                    std::sort(nb_rank.begin(), nb_rank.end(),
                              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                                  return a.first < b.first;
                              });
                }
                int external_pushed = 0;
                for (const auto& item : nb_rank) {
                    if (push_frontier_node(item.second, item.first)) {
                        ++external_pushed;
                    }
                }

                // 内部 gateway：同样按 utility gradient 选，而不是边存储顺序。
                if (external_pushed < eff_beam && gateway_expanded < gateway_budget) {
                    nb_rank.clear();
                    float best_internal_vec = std::numeric_limits<float>::max();
                    for (int pos = begin; pos < finish; ++pos) {
                        const int nb_fid = fine_graph_neighbors_[static_cast<size_t>(pos)];
                        if (nb_fid < 0 || nb_fid >= n_fine_clusters_ ||
                            is_visited(nb_fid) || is_external_fine(nb_fid)) continue;
                        if (scratch.top_stamp[static_cast<size_t>(nb_fid)] == gen) continue;
                        best_internal_vec = std::min(best_internal_vec, dist_q_fid(nb_fid));
                    }
                    for (int pos = begin; pos < finish; ++pos) {
                        const int nb_fid = fine_graph_neighbors_[static_cast<size_t>(pos)];
                        if (nb_fid < 0 || nb_fid >= n_fine_clusters_ ||
                            is_visited(nb_fid) || is_external_fine(nb_fid)) continue;
                        if (scratch.top_stamp[static_cast<size_t>(nb_fid)] == gen) continue;
                        nb_rank.emplace_back(guarded_expand_priority(cur.fid, nb_fid, best_internal_vec), nb_fid);
                    }
                    const int take_internal = std::min(internal_gateway_width, static_cast<int>(nb_rank.size()));
                    if (take_internal > 0) {
                        if (take_internal < static_cast<int>(nb_rank.size())) {
                            std::partial_sort(
                                nb_rank.begin(), nb_rank.begin() + take_internal, nb_rank.end(),
                                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                                    return a.first < b.first;
                                });
                        }
                        for (int i = 0; i < take_internal; ++i) {
                            push_frontier_node(nb_rank[static_cast<size_t>(i)].second,
                                               nb_rank[static_cast<size_t>(i)].first);
                        }
                    }
                }
                trim_one_frontier(predicate_frontier);
                trim_one_frontier(fallback_frontier);
            }

            std::sort_heap(top_nodes.begin(), top_nodes.end(), MaxResultCmp{});
            candidate_fine_ids.clear();
            candidate_fine_ids.reserve(top_nodes.size());
            for (const auto& node : top_nodes) candidate_fine_ids.push_back(node.fid);
        }

        if (scalar_cluster_prune_effective && candidate_fine_ids.size() > 1) {
            std::vector<int> overlap_candidates;
            overlap_candidates.reserve(candidate_fine_ids.size());
            size_t route_pruned = 0;
            for (int fid : candidate_fine_ids) {
                if (scalar_may_match_fid(fid)) {
                    overlap_candidates.push_back(fid);
                } else {
                    ++route_pruned;
                }
            }
            if (!overlap_candidates.empty() && route_pruned > 0) {
                if (stats) stats->num_pruned_clusters += route_pruned;
                candidate_fine_ids.swap(overlap_candidates);
            }
        }

        // 3) 列表扫描：block-first 重排 + adaptive per-cluster scan budget
        if (candidate_fine_ids.empty()) {
            return {};
        }

        int effective_budget = (budget > 0) ? budget : (k * 100);
        if (stats) {
            std::vector<int> uniq = candidate_fine_ids;
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            stats->num_candidate_clusters += uniq.size();
        }

        // scalar utility rerank：先按纯几何恢复候选簇主序，70% protected prefix 保 recall，
        // 后 30% 用 scalar utility 重新调度，让预算更早花到预计有效簇。
        if (scalar_route_effective && candidate_fine_ids.size() > 1) {
            std::stable_sort(
                candidate_fine_ids.begin(), candidate_fine_ids.end(),
                [&](int a, int b) { return dist_q_fid(a) < dist_q_fid(b); });
            const int protected_n = std::max(
                1, std::min(static_cast<int>(candidate_fine_ids.size()),
                            (static_cast<int>(candidate_fine_ids.size()) * 7 + 9) / 10));
            if (protected_n < static_cast<int>(candidate_fine_ids.size())) {
                std::stable_sort(
                    candidate_fine_ids.begin() + protected_n,
                    candidate_fine_ids.end(),
                    [&](int a, int b) {
                        const float ua = scalar_utility_fid(a);
                        const float ub = scalar_utility_fid(b);
                        if (std::fabs(ua - ub) > 1e-6f) return ua > ub;
                        return dist_q_fid(a) < dist_q_fid(b);
                    });
            }
        }

        return fine_index_->search_in_clusters_with_stats(
            query, k, effective_budget, candidate_fine_ids, range_min, range_max, stats);
    }

private:
    // 计算质心变化的统计分布（相对于上次 coarse 层同步的基准）
    // 修复：使用 fine_centroids_history_ 作为参考，使 epsilon 在 coarse 重建后正确重置并随后的插入增长
    // 若 history 为空则用 initial，保证训练后第一批有基准
    float compute_centroid_change_mean() const {
        const std::vector<std::vector<float>>* reference_centroids = nullptr;
        if (!fine_centroids_history_.empty()) {
            reference_centroids = &fine_centroids_history_;
        } else if (!fine_centroids_initial_.empty()) {
            reference_centroids = &fine_centroids_initial_;
        }
        
        if (reference_centroids == nullptr || fine_centroids_.empty()) {
            return 0.0f;
        }
        
        float total_change = 0.0f;
        size_t count = 0;
        
        size_t n_fine = std::min(fine_centroids_.size(), reference_centroids->size());
        for (size_t i = 0; i < n_fine; ++i) {
            float change = l2_distance_sq(
                fine_centroids_[i].data(),
                (*reference_centroids)[i].data()
            );
            total_change += std::sqrt(change);  // 转换为实际距离
            ++count;
        }
        
        return count > 0 ? total_change / static_cast<float>(count) : 0.0f;
    }
    
    float compute_centroid_change_std(float mean_change) const {
        const std::vector<std::vector<float>>* reference_centroids = nullptr;
        if (!fine_centroids_history_.empty()) {
            reference_centroids = &fine_centroids_history_;
        } else if (!fine_centroids_initial_.empty()) {
            reference_centroids = &fine_centroids_initial_;
        }
        
        if (reference_centroids == nullptr || fine_centroids_.empty()) {
            return 0.0f;
        }
        
        float variance = 0.0f;
        size_t count = 0;
        
        size_t n_fine = std::min(fine_centroids_.size(), reference_centroids->size());
        for (size_t i = 0; i < n_fine; ++i) {
            float change = std::sqrt(l2_distance_sq(
                fine_centroids_[i].data(),
                (*reference_centroids)[i].data()
            ));
            float diff = change - mean_change;
            variance += diff * diff;
            ++count;
        }
        
        return count > 0 ? std::sqrt(variance / static_cast<float>(count)) : 0.0f;
    }
    
    // 自适应 centroid_update_threshold（基于质心变化统计）
    float compute_adaptive_centroid_update_threshold() const {
        if (fine_centroids_history_.empty() || !is_trained_) {
            return centroid_update_threshold_;  // 回退到固定阈值
        }
        
        float mean_change = compute_centroid_change_mean();
        float std_change = compute_centroid_change_std(mean_change);
        
        // 使用 mean + 0.5 * std 作为阈值（超过此值才更新 coarse 层）
        float adaptive_threshold = mean_change + 0.5f * std_change;
        
        // 确保阈值在合理范围内（不低于固定阈值的50%，不高于固定阈值的200%）
        float min_threshold = centroid_update_threshold_ * 0.5f;
        float max_threshold = centroid_update_threshold_ * 2.0f;
        adaptive_threshold = std::max(min_threshold, std::min(adaptive_threshold, max_threshold));
        
        return adaptive_threshold;
    }
    
    // 辅助函数：按向量数量计算各粗簇大小（与 get_imbalance_metrics 一致）
    std::vector<size_t> get_coarse_cluster_sizes_by_vectors() const {
        std::vector<size_t> coarse_cluster_sizes(n_coarse_clusters_, 0);
        std::vector<size_t> fine_cluster_sizes = fine_index_->get_all_cluster_sizes();
        bool use_fine_to_coarse = (fine_to_coarse_.size() == static_cast<size_t>(n_fine_clusters_));
        if (use_fine_to_coarse && fine_cluster_sizes.size() == fine_to_coarse_.size()) {
            for (size_t fid = 0; fid < fine_to_coarse_.size(); ++fid) {
                int cid = fine_to_coarse_[static_cast<int>(fid)];
                if (cid >= 0 && cid < n_coarse_clusters_ && fid < fine_cluster_sizes.size()) {
                    coarse_cluster_sizes[static_cast<size_t>(cid)] += fine_cluster_sizes[fid];
                }
            }
        } else {
            for (size_t fid = 0; fid < fine_centroids_.size() && fid < fine_cluster_sizes.size(); ++fid) {
                float best_dist = std::numeric_limits<float>::max();
                int best_cid = 0;
                const float* fcent = fine_centroids_[fid].data();
                for (int cid = 0; cid < n_coarse_clusters_; ++cid) {
                    if (static_cast<size_t>(cid) >= coarse_centroids_.size()) continue;
                    float dist = l2_distance_sq(fcent, coarse_centroids_[static_cast<size_t>(cid)].data());
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cid = cid;
                    }
                }
                if (best_cid >= 0 && best_cid < n_coarse_clusters_) {
                    coarse_cluster_sizes[static_cast<size_t>(best_cid)] += fine_cluster_sizes[fid];
                }
            }
        }
        return coarse_cluster_sizes;
    }

    // 方案2：分层归一化版本 - 计算粗层全局失衡指标 G_c(σ_c, ε_c, ε_c')
    // 分别归一化三个分量，解决量纲不一致问题
    // 修复：σ_c 使用向量数量（与 get_imbalance_metrics 一致），而非 fine 聚类数量
    float compute_global_imbalance_indicator_v2() const {
        if (!enable_maintenance_) {
            return 0.0f;
        }
        if (!is_trained_ || fine_centroids_.empty() || coarse_centroids_.empty()) {
            return 0.0f;
        }
        
        // 1. 计算分区大小标准差 σ_c（使用向量数量，与 get_imbalance_metrics 一致）
        std::vector<size_t> coarse_cluster_sizes = get_coarse_cluster_sizes_by_vectors();
        
        float avg_size = 0.0f;
        size_t count = 0;
        for (size_t cid = 0; cid < static_cast<size_t>(n_coarse_clusters_); ++cid) {
            if (coarse_cluster_sizes[cid] > 0) {
                avg_size += static_cast<float>(coarse_cluster_sizes[cid]);
                count++;
            }
        }
        if (count == 0) return 0.0f;
        avg_size /= static_cast<float>(count);
        
        float variance = 0.0f;
        for (size_t cid = 0; cid < static_cast<size_t>(n_coarse_clusters_); ++cid) {
            if (coarse_cluster_sizes[cid] > 0) {
                float diff = static_cast<float>(coarse_cluster_sizes[cid]) - avg_size;
                variance += diff * diff;
            }
        }
        float sigma_c = std::sqrt(variance / static_cast<float>(count));
        // 使用变异系数归一化（范围通常在 [0, 1]）
        float cv_size = (avg_size > 0) ? (sigma_c / avg_size) : 0.0f;
        
        // 2. 计算重建误差 ε_c（相对变化）
        float epsilon_c = compute_centroid_change_mean();
        float relative_epsilon = 0.0f;
        // 使用与 compute_centroid_change_mean 相同的参考质心（history 优先）
        const std::vector<std::vector<float>>* ref = !fine_centroids_history_.empty() ?
            &fine_centroids_history_ : (!fine_centroids_initial_.empty() ? &fine_centroids_initial_ : nullptr);
        if (ref && !ref->empty() && epsilon_c > 0) {
            float avg_norm = 0.0f;
            size_t n_fine = std::min(ref->size(), fine_centroids_.size());
            for (size_t i = 0; i < n_fine; ++i) {
                float norm_sq = 0.0f;
                for (int d = 0; d < dimension_; ++d) {
                    float v = (*ref)[i][static_cast<size_t>(d)];
                    norm_sq += v * v;
                }
                avg_norm += std::sqrt(norm_sq);
            }
            if (n_fine > 0) {
                avg_norm /= static_cast<float>(n_fine);
                relative_epsilon = (avg_norm > 0) ? (epsilon_c / avg_norm) : 0.0f;
            }
        }
        
        // 3. 计算漂移误差 ε_c'（相对化）
        float epsilon_c_prime = compute_centroid_change_std(epsilon_c);
        float relative_epsilon_prime = (epsilon_c > 0) ? 
            (epsilon_c_prime / epsilon_c) : 0.0f;
        
        // 4. 加权组合（各分量都在 [0, 1] 范围）
        const float w_size = 0.4f;      // 分区失衡权重
        const float w_drift = 0.4f;     // 质心漂移权重
        const float w_variance = 0.2f;  // 漂移方差权重
        
        float G_c = w_size * std::min(cv_size, 1.0f) +
                    w_drift * std::min(relative_epsilon, 1.0f) +
                    w_variance * std::min(relative_epsilon_prime, 1.0f);
        
        return G_c;  // 范围 [0, 1]，阈值可设为 0.3-0.5
    }
    
    struct CoarseSoftReconcileStats {
        bool applied = false;
        bool mapping_changed = false;
        int moved_fines = 0;
        float moved_ratio = 0.0f;
        float rel_centroid_shift = 0.0f;
        float avg_radius = 0.0f;
    };

    CoarseSoftReconcileStats soft_reconcile_coarse_layer() {
        CoarseSoftReconcileStats stats;
        const size_t n_fine = std::min(fine_centroids_.size(), fine_to_coarse_.size());
        const size_t n_coarse = coarse_centroids_.size();
        if (n_fine == 0 || n_coarse == 0 || dimension_ <= 0) {
            return stats;
        }

        std::vector<size_t> fine_sizes = fine_index_->get_all_cluster_sizes();
        std::vector<std::vector<double>> sums(n_coarse, std::vector<double>(static_cast<size_t>(dimension_), 0.0));
        std::vector<double> weights(n_coarse, 0.0);

        double radius_dsq_sum = 0.0;
        double total_weight = 0.0;
        for (size_t fid = 0; fid < n_fine; ++fid) {
            const int cid = fine_to_coarse_[fid];
            if (cid < 0 || static_cast<size_t>(cid) >= n_coarse) {
                continue;
            }
            const double w = (fid < fine_sizes.size() && fine_sizes[fid] > 0)
                                 ? static_cast<double>(fine_sizes[fid])
                                 : 1.0;
            for (int d = 0; d < dimension_; ++d) {
                sums[static_cast<size_t>(cid)][static_cast<size_t>(d)] +=
                    w * static_cast<double>(fine_centroids_[fid][static_cast<size_t>(d)]);
            }
            weights[static_cast<size_t>(cid)] += w;
            radius_dsq_sum += w * static_cast<double>(
                l2_distance_sq(fine_centroids_[fid].data(), coarse_centroids_[static_cast<size_t>(cid)].data()));
            total_weight += w;
        }
        if (total_weight <= 0.0) {
            return stats;
        }

        const double avg_radius = std::sqrt(std::max(0.0, radius_dsq_sum / total_weight));
        stats.avg_radius = static_cast<float>(avg_radius);
        const double radius_floor = std::max(avg_radius, 1e-6);

        const float soft_tau = coarse_soft_drift_tau_;
        const float ema_alpha = coarse_centroid_ema_alpha_;
        const float remap_rel_threshold = fine_remap_rel_improvement_;
        const float remap_margin_ratio = fine_remap_abs_margin_ratio_;

        std::vector<std::vector<float>> target = coarse_centroids_;
        double shift_sum = 0.0;
        double shift_weight = 0.0;
        for (size_t cid = 0; cid < n_coarse; ++cid) {
            if (weights[cid] <= 0.0) {
                continue;
            }
            double shift_dsq = 0.0;
            for (int d = 0; d < dimension_; ++d) {
                const float v = static_cast<float>(sums[cid][static_cast<size_t>(d)] / weights[cid]);
                const double diff = static_cast<double>(v) -
                                    static_cast<double>(coarse_centroids_[cid][static_cast<size_t>(d)]);
                shift_dsq += diff * diff;
                target[cid][static_cast<size_t>(d)] = v;
            }
            shift_sum += weights[cid] * std::sqrt(shift_dsq);
            shift_weight += weights[cid];
        }

        const double mean_shift = (shift_weight > 0.0) ? (shift_sum / shift_weight) : 0.0;
        stats.rel_centroid_shift = static_cast<float>(mean_shift / radius_floor);
        const bool update_centroids = stats.rel_centroid_shift >= soft_tau;
        if (update_centroids) {
            const float alpha = std::max(0.0f, std::min(1.0f, ema_alpha));
            for (size_t cid = 0; cid < n_coarse; ++cid) {
                if (weights[cid] <= 0.0) {
                    continue;
                }
                for (int d = 0; d < dimension_; ++d) {
                    float& cur = coarse_centroids_[cid][static_cast<size_t>(d)];
                    cur = (1.0f - alpha) * cur + alpha * target[cid][static_cast<size_t>(d)];
                }
            }
            stats.applied = true;
        }

        const float abs_margin_dsq = static_cast<float>(
            remap_margin_ratio * remap_margin_ratio * radius_floor * radius_floor);
        int moved = 0;
        for (size_t fid = 0; fid < n_fine; ++fid) {
            const int old_cid = fine_to_coarse_[fid];
            if (old_cid < 0 || static_cast<size_t>(old_cid) >= n_coarse) {
                continue;
            }
            float old_dsq = l2_distance_sq(
                fine_centroids_[fid].data(), coarse_centroids_[static_cast<size_t>(old_cid)].data());
            float best_dsq = old_dsq;
            int best_cid = old_cid;
            for (size_t cid = 0; cid < n_coarse; ++cid) {
                if (static_cast<int>(cid) == old_cid) {
                    continue;
                }
                float dsq = l2_distance_sq(fine_centroids_[fid].data(), coarse_centroids_[cid].data());
                if (dsq < best_dsq) {
                    best_dsq = dsq;
                    best_cid = static_cast<int>(cid);
                }
            }
            const float improvement = old_dsq - best_dsq;
            const float rel_improvement = (old_dsq > 1e-12f) ? (improvement / old_dsq) : 0.0f;
            if (best_cid != old_cid &&
                improvement > abs_margin_dsq &&
                rel_improvement >= remap_rel_threshold) {
                fine_to_coarse_[fid] = best_cid;
                ++moved;
            }
        }

        if (moved > 0) {
            coarse_to_fine_.assign(n_coarse, std::vector<int>());
            for (size_t fid = 0; fid < n_fine; ++fid) {
                const int cid = fine_to_coarse_[fid];
                if (cid >= 0 && static_cast<size_t>(cid) < n_coarse) {
                    coarse_to_fine_[static_cast<size_t>(cid)].push_back(static_cast<int>(fid));
                }
            }
            stats.applied = true;
            stats.mapping_changed = true;
            stats.moved_fines = moved;
            stats.moved_ratio = static_cast<float>(moved) / static_cast<float>(n_fine);
        }

        coarse_soft_moved_fines_last_ = moved;
        coarse_soft_moved_ratio_last_ = stats.moved_ratio;
        coarse_soft_rel_shift_last_ = stats.rel_centroid_shift;
        coarse_soft_avg_radius_last_ = stats.avg_radius;

        if (stats.applied) {
            ++coarse_soft_sync_count_;
            coarse_soft_moved_fines_total_ += moved;
            rebuild_coarse_seed_fines();
            // 更积极刷图：任一 mapping 变化，或相对漂移达到 soft_tau 的一半，即重建 fine graph，
            // 避免「新 coarse 归属 + 旧边」短暂不一致拖累残差 repair。
            if (enable_graph_routing_.load() &&
                ((stats.mapping_changed &&
                  stats.moved_ratio >= fine_remap_min_moved_ratio_) ||
                 stats.rel_centroid_shift >= soft_tau * 0.5f)) {
                rebuild_fine_centroid_graph(true);
            }
        }
        return stats;
    }

    // 方案1：计算自适应阈值
    float compute_adaptive_tau_Gc() const {
        if (gc_history_.values.size() < 5) {
            return adaptive_config_.base_tau;  // 历史不足，使用默认值
        }
        
        // 计算历史均值和标准差
        float mean = 0.0f, variance = 0.0f;
        for (float v : gc_history_.values) {
            mean += v;
        }
        mean /= static_cast<float>(gc_history_.values.size());
        
        for (float v : gc_history_.values) {
            float diff = v - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(gc_history_.values.size());
        float std_dev = std::sqrt(variance);
        
        // 自适应阈值 = 均值 + sensitivity * 标准差
        // 原理：如果当前 G_c 显著高于历史水平，说明需要重建
        float adaptive_tau = mean + adaptive_config_.sensitivity * std_dev;
        
        // 限制在合理范围内
        return std::max(adaptive_config_.min_tau, 
                       std::min(adaptive_config_.max_tau, adaptive_tau));
    }
    
    // 原始版本（保留用于对比）
    // 计算粗层全局失衡指标 G_c(σ_c, ε_c, ε_c')
    // 其中 σ_c 是分区大小标准差，ε_c 是重建误差，ε_c' 是漂移误差
    float compute_global_imbalance_indicator() const {
        if (!enable_maintenance_) {
            return 0.0f;
        }
        if (!is_trained_ || fine_centroids_.empty() || coarse_centroids_.empty()) {
            return 0.0f;
        }
        
        // 1. 计算分区大小标准差 σ_c
        // 统计每个粗簇包含的细簇数量
        std::vector<size_t> coarse_cluster_sizes(n_coarse_clusters_, 0);
        for (size_t fid = 0; fid < fine_to_coarse_.size(); ++fid) {
            int cid = fine_to_coarse_[fid];
            if (cid >= 0 && cid < n_coarse_clusters_) {
                coarse_cluster_sizes[static_cast<size_t>(cid)]++;
            }
        }
        
        // 计算平均值和标准差
        float avg_size = 0.0f;
        size_t count = 0;
        for (size_t cid = 0; cid < static_cast<size_t>(n_coarse_clusters_); ++cid) {
            if (coarse_cluster_sizes[cid] > 0) {
                avg_size += static_cast<float>(coarse_cluster_sizes[cid]);
                count++;
            }
        }
        if (count == 0) return 0.0f;
        avg_size /= static_cast<float>(count);
        
        float variance = 0.0f;
        for (size_t cid = 0; cid < static_cast<size_t>(n_coarse_clusters_); ++cid) {
            if (coarse_cluster_sizes[cid] > 0) {
                float diff = static_cast<float>(coarse_cluster_sizes[cid]) - avg_size;
                variance += diff * diff;
            }
        }
        float sigma_c = std::sqrt(variance / static_cast<float>(count));  // 分区大小标准差
        
        // 2. 计算重建误差 ε_c（质心变化均值）
        float epsilon_c = compute_centroid_change_mean();
        
        // 3. 计算漂移误差 ε_c'（质心变化标准差）
        float epsilon_c_prime = compute_centroid_change_std(epsilon_c);
        
        // 4. 全局失衡指标：G_c = σ_c + ε_c + ε_c'（归一化）
        // 归一化因子：使用平均值作为基准
        float normalization = std::max(avg_size, 1.0f);
        float G_c = (sigma_c + epsilon_c + epsilon_c_prime) / normalization;
        
        return G_c;
    }
    
    // 检查并更新 coarse 层（如果需要）
    // 使用方案2（分层归一化）+ 方案1（自适应阈值）+ 时间间隔保护
    void check_and_update_coarse_layer() {
        if (!enable_maintenance_) {
            return;
        }
        if (!is_trained_) {
            if (debug_log_) std::cerr << "[Coarse层检查] 跳过: is_trained=" << is_trained_ << std::endl;
            return;
        }
        
        // 修复epsilon NaN问题：如果fine_centroids_history_为空，初始化它
        if (fine_centroids_history_.empty() && !fine_centroids_.empty()) {
            fine_centroids_history_ = fine_centroids_;
            if (debug_log_) std::cerr << "[Coarse层检查] 初始化fine_centroids_history_, 质心数: " << fine_centroids_history_.size() << std::endl;
        }
        
        if (fine_centroids_history_.empty()) {
            if (debug_log_) std::cerr << "[Coarse层检查] 跳过: fine_centroids_history_仍为空" << std::endl;
            return;
        }

        CoarseSoftReconcileStats soft_stats = soft_reconcile_coarse_layer();
        if (soft_stats.applied) {
            fine_centroids_history_ = fine_centroids_;
            if (debug_log_) {
                std::cerr << "[Coarse层软同步] rel_centroid_shift=" << soft_stats.rel_centroid_shift
                          << ", avg_radius=" << soft_stats.avg_radius
                          << ", moved_fines=" << soft_stats.moved_fines
                          << ", moved_ratio=" << soft_stats.moved_ratio << std::endl;
            }
        }
        
        // 仅按插入量保护，保证不同机器和运行速度下触发点一致。
        if (inserted_since_coarse_rebuild_ < coarse_rebuild_min_insertions_) {
            return;
        }
        
        // 计算全局失衡指标 G_c（使用方案2：分层归一化）
        auto compute_start = std::chrono::high_resolution_clock::now();
        float G_c = compute_global_imbalance_indicator_v2();
        auto compute_end = std::chrono::high_resolution_clock::now();
        auto compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
        
        // 记录历史（用于自适应阈值）
        gc_history_.values.push_back(G_c);
        if (gc_history_.values.size() > gc_history_.max_history) {
            gc_history_.values.pop_front();
        }
        
        // 方案1：计算自适应阈值
        float tau_Gc = compute_adaptive_tau_Gc();
        
        // 静态变量记录上一次的G_c值（用于调试输出）
        static float prev_G_c = 0.0f;
        if (std::abs(prev_G_c - G_c) > 0.001f) {
            if (debug_log_) {
                std::cerr << "[Coarse层G值变化] " << prev_G_c << " -> " << G_c 
                          << ", 自适应阈值=" << tau_Gc 
                          << ", 历史样本数=" << gc_history_.values.size()
                          << ", 计算耗时=" << compute_duration << "ms" << std::endl;
            }
            prev_G_c = G_c;
        }
        
        // 额外检查：相对变化率（插入优先：提高阈值，减少 coarse 重建）
        bool significant_increase = false;
        if (gc_history_.values.size() >= 2) {
            float prev = gc_history_.values[gc_history_.values.size() - 2];
            float increase_rate = (prev > 1e-6f) ? ((G_c - prev) / prev) : 0.0f;
            significant_increase = (increase_rate > 1.0f);  // 单次翻倍才认为“快速变化”
            if (significant_increase) {
                if (debug_log_) std::cerr << "[Coarse层快速变化检测] G_c单次增长 " << (increase_rate * 100) << "%, 触发重建" << std::endl;
            }
        }
        
        // 综合决策：使用自适应阈值或检测到显著增长
        if (G_c > tau_Gc || significant_increase) {
            if (debug_log_) {
                std::cerr << "[⚠️ Coarse层全局重建触发] G_c=" << G_c << " > 自适应阈值=" << tau_Gc 
                          << (significant_increase ? " (或检测到快速变化)" : "") << std::endl;
            }
            auto rebuild_start = std::chrono::high_resolution_clock::now();
            // 重新训练 coarse 层（使用最新的 fine 质心）
            update_coarse_layer_from_fine();
            auto rebuild_end = std::chrono::high_resolution_clock::now();
            auto rebuild_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rebuild_end - rebuild_start).count();
            if (debug_log_) std::cerr << "[Coarse层重建完成] 耗时: " << rebuild_duration << "ms" << std::endl;
            
            // 更新历史和时间戳（但不更新初始质心，用于计算epsilon）
            fine_centroids_history_ = fine_centroids_;
            // 注意：fine_centroids_initial_不更新，保持训练时的初始值
            coarse_global_rebuild_count_++;
            inserted_since_coarse_rebuild_ = 0;
            
            // 清空G_c历史（重建后重新开始统计）
            gc_history_.values.clear();
        }
    }
    
    // 从 fine 质心更新 coarse 层
    void update_coarse_layer_from_fine() {
        if (fine_centroids_.empty()) {
            if (debug_log_) std::cerr << "[Coarse层更新] 错误: fine_centroids_为空" << std::endl;
            return;
        }
        
        if (debug_log_) {
            std::cerr << "[Coarse层更新] 开始更新, fine质心数=" << fine_centroids_.size() 
                      << ", coarse聚类数=" << n_coarse_clusters_ << std::endl;
        }
        
        // 将 fine_centroids_ 展平为连续数组
        auto flatten_start = std::chrono::high_resolution_clock::now();
        size_t n_fine = fine_centroids_.size();
        std::vector<float> flat_fine(n_fine * static_cast<size_t>(dimension_));
        for (size_t i = 0; i < n_fine; ++i) {
            for (int d = 0; d < dimension_; ++d) {
                flat_fine[i * static_cast<size_t>(dimension_) + static_cast<size_t>(d)] =
                    fine_centroids_[i][d];
            }
        }
        auto flatten_end = std::chrono::high_resolution_clock::now();
        auto flatten_duration = std::chrono::duration_cast<std::chrono::milliseconds>(flatten_end - flatten_start).count();
        if (debug_log_) std::cerr << "[Coarse层更新] 展平完成, 耗时: " << flatten_duration << "ms" << std::endl;
        
        int n_coarse_train = std::min(n_coarse_clusters_, static_cast<int>(n_fine));
        
        auto train_start = std::chrono::high_resolution_clock::now();
        // 使用与单层训练相同的高标准迭代次数
        int coarse_max_iter = 20;
        if (n_fine >= 500000 || n_coarse_train >= 500) {
            coarse_max_iter = 40;
        } else if (n_fine >= 200000 || n_coarse_train >= 300) {
            coarse_max_iter = 35;
        } else if (n_fine >= 100000 || n_coarse_train >= 200) {
            coarse_max_iter = 30;
        } else if (n_fine < 10000 && n_coarse_train < 100) {
            coarse_max_iter = 20;
        } else {
            coarse_max_iter = 20;
        }
        const std::vector<size_t> fine_sizes = fine_index_->get_all_cluster_sizes();
        std::vector<float> fine_weights(n_fine, 1.0f);
        bool have_nonzero_weight = false;
        for (size_t i = 0; i < n_fine && i < fine_sizes.size(); ++i) {
            fine_weights[i] = static_cast<float>(fine_sizes[i]);
            have_nonzero_weight = have_nonzero_weight || fine_sizes[i] > 0;
        }
        if (have_nonzero_weight) {
            local_kmeans_->train_weighted_with_max_iter(
                flat_fine.data(), fine_weights.data(), n_fine, dimension_,
                n_coarse_train, coarse_max_iter);
        } else {
            local_kmeans_->train_with_max_iter(
                flat_fine.data(), n_fine, dimension_, n_coarse_train, coarse_max_iter);
        }
        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
        if (debug_log_) std::cerr << "[Coarse层更新] K-means训练完成, 耗时: " << train_duration << "ms" << std::endl;
        
        coarse_centroids_ = local_kmeans_->get_centroids();
        if (debug_log_) std::cerr << "[Coarse层更新] 获取coarse质心完成, 质心数: " << coarse_centroids_.size() << std::endl;
        
        // 重新建立 fine -> coarse 映射
        auto mapping_start = std::chrono::high_resolution_clock::now();
        build_fine_to_coarse_mapping(enable_graph_routing_.load());
        if (enable_graph_routing_.load()) {
            rebuild_fine_centroid_graph(/*force=*/true);
        }
        auto mapping_end = std::chrono::high_resolution_clock::now();
        auto mapping_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mapping_end - mapping_start).count();
        if (debug_log_) std::cerr << "[Coarse层更新] 映射建立完成, 耗时: " << mapping_duration << "ms" << std::endl;
    }
};

// ---------------------------------------------------------------------------
// pybind11 包装类
// ---------------------------------------------------------------------------

// 细层搜索视图：直接复用 AdaIVFCore::batch_search（与单层 AdaIVFIndex 同逻辑）
class PyFineIndexView {
public:
    explicit PyFineIndexView(AdaIVFCore* core) : core_(core) {}

    py::list batch_search(py::array_t<float> queries, int k, int max_num_distances,
                           float range_min, float range_max) const {
        return ada_ivf_py::batch_search_to_python(
            core_, queries, k, max_num_distances, range_min, range_max);
    }

    py::list batch_search_ranges(py::array_t<float> queries, py::array_t<float> ranges,
                                 int k, int max_num_distances) const {
        return ada_ivf_py::batch_search_ranges_to_python(
            core_, queries, ranges, k, max_num_distances);
    }

    py::tuple batch_search_with_stats(py::array_t<float> queries, int k, int max_num_distances,
                                      float range_min, float range_max) const {
        return ada_ivf_py::batch_search_with_stats_to_python(
            core_, queries, k, max_num_distances, range_min, range_max);
    }

    py::tuple batch_search_ranges_with_stats(py::array_t<float> queries, py::array_t<float> ranges,
                                             int k, int max_num_distances) const {
        return ada_ivf_py::batch_search_ranges_with_stats_to_python(
            core_, queries, ranges, k, max_num_distances);
    }

    int n_probe() const { return core_ ? core_->get_n_probe() : 0; }
    void set_n_probe(int v) {
        if (core_) core_->set_n_probe(v);
    }
    void set_enable_scalar_range_prune(bool v) { if (core_) core_->set_enable_scalar_range_prune(v); }
    bool get_enable_scalar_range_prune() const { return core_ ? core_->get_enable_scalar_range_prune() : false; }

private:
    AdaIVFCore* core_;
};

class PyHierarchicalAdaIVFIndex {
public:
    PyHierarchicalAdaIVFIndex(int n_fine_clusters,
                              int n_coarse_clusters,
                              int n_probe,
                              size_t max_cluster_size,
                              float recluster_threshold,
                              float centroid_update_threshold,
                              int n_probe_f = -1)  // 细层n_probe，-1表示不限制
        : core_(std::make_unique<HierarchicalAdaIVFCore>(
              n_fine_clusters,
              n_coarse_clusters,
              n_probe,
              max_cluster_size,
              recluster_threshold,
              centroid_update_threshold,
              n_probe_f)) {}

    void train(py::array_t<float> vectors, py::object scalars_obj = py::none()) {
        auto buf = vectors.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("训练向量必须是二维数组 (n, d)");
        }
        size_t n = static_cast<size_t>(buf.shape[0]);
        int d = static_cast<int>(buf.shape[1]);
        const float* data = static_cast<const float*>(buf.ptr);
        const float* scalar_ptr = nullptr;
        py::array_t<float> scalars;
        if (!scalars_obj.is_none()) {
            scalars = py::cast<py::array_t<float>>(scalars_obj);
            auto sbuf = scalars.request();
            if (sbuf.ndim != 1 || static_cast<size_t>(sbuf.shape[0]) != n) {
                throw std::runtime_error("训练 scalars 必须是一维数组，且长度与训练向量一致");
            }
            scalar_ptr = static_cast<const float*>(sbuf.ptr);
        }
        core_->train(data, n, d, scalar_ptr);
    }

    void add(py::array_t<float> vectors,
             py::array_t<int> ids,
             bool auto_recluster,
             py::array_t<float> scalars) {
        auto vbuf = vectors.request();
        if (vbuf.ndim != 2) {
            throw std::runtime_error("插入向量必须是二维数组 (n, d)");
        }
        size_t n = static_cast<size_t>(vbuf.shape[0]);
        const float* data = static_cast<const float*>(vbuf.ptr);

        const int* id_ptr = nullptr;
        if (ids && ids.size() > 0) {
            auto ibuf = ids.request();
            if (ibuf.ndim != 1 || static_cast<size_t>(ibuf.shape[0]) != n) {
                throw std::runtime_error("ids 长度必须与向量数量一致");
            }
            id_ptr = static_cast<const int*>(ibuf.ptr);
        }

        const float* scalar_ptr = nullptr;
        if (scalars && scalars.size() > 0) {
            auto sbuf = scalars.request();
            if (sbuf.ndim != 1 || static_cast<size_t>(sbuf.shape[0]) != n) {
                throw std::runtime_error("scalars 长度必须与向量数量一致");
            }
            scalar_ptr = static_cast<const float*>(sbuf.ptr);
        }

        core_->add(data, n, id_ptr, auto_recluster, scalar_ptr);
    }

    // 返回：List[numpy int32]，每条 query 的 top-k 向量 id（已按距离升序）
    py::list batch_search(py::array_t<float> queries,
                          int k,//每条查询返回 top-k 的候选数
                          int base_n_probe,
                          int base_search_k,
                          py::object range_min_obj = py::none(),
                          py::object range_max_obj = py::none(),
                          bool use_mid_filtering = false) {
        auto qbuf = queries.request();
        if (qbuf.ndim != 2) {
            throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
        }
        size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
        const float* qdata = static_cast<const float*>(qbuf.ptr);

        float range_min = std::numeric_limits<float>::lowest();
        float range_max = std::numeric_limits<float>::max();
        if (!range_min_obj.is_none() && !range_max_obj.is_none()) {
            range_min = range_min_obj.cast<float>();
            range_max = range_max_obj.cast<float>();
        }

        auto results = core_->batch_search(qdata, n_queries, k,
                                           base_n_probe, base_search_k,
                                           range_min, range_max,
                                           use_mid_filtering);
        return ada_ivf_py::batch_results_indices_to_python(results);
    }

    py::list batch_search_ranges(py::array_t<float> queries,
                                 py::array_t<float> ranges,
                                 int k,
                                 int base_n_probe,
                                 int base_search_k,
                                 bool use_mid_filtering = false) {
        auto qbuf = queries.request();
        auto rbuf = ranges.request();
        if (qbuf.ndim != 2) {
            throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
        }
        if (rbuf.ndim != 2 || rbuf.shape[1] != 2 || rbuf.shape[0] != qbuf.shape[0]) {
            throw std::runtime_error("ranges 必须是二维数组 (n_queries, 2)，且长度与 queries 匹配");
        }
        size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
        const float* qdata = static_cast<const float*>(qbuf.ptr);
        const float* rdata = static_cast<const float*>(rbuf.ptr);

        auto results = core_->batch_search_ranges(
            qdata, rdata, n_queries, k, base_n_probe, base_search_k, use_mid_filtering);
        return ada_ivf_py::batch_results_indices_to_python(results);
    }

    py::tuple batch_search_with_stats(py::array_t<float> queries,
                                      int k,
                                      int base_n_probe,
                                      int base_search_k,
                                      py::object range_min_obj = py::none(),
                                      py::object range_max_obj = py::none(),
                                      bool use_mid_filtering = false) {
        auto qbuf = queries.request();
        if (qbuf.ndim != 2) {
            throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
        }
        size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
        const float* qdata = static_cast<const float*>(qbuf.ptr);
        float range_min = std::numeric_limits<float>::lowest();
        float range_max = std::numeric_limits<float>::max();
        if (!range_min_obj.is_none() && !range_max_obj.is_none()) {
            range_min = range_min_obj.cast<float>();
            range_max = range_max_obj.cast<float>();
        }
        auto out = core_->batch_search_with_stats(
            qdata, n_queries, k, base_n_probe, base_search_k,
            range_min, range_max, use_mid_filtering);
        return py::make_tuple(
            ada_ivf_py::batch_results_indices_to_python(out.results),
            ada_ivf_py::aggregate_search_stats_to_python(out.per_query_stats));
    }

    py::tuple batch_search_ranges_with_stats(py::array_t<float> queries,
                                             py::array_t<float> ranges,
                                             int k,
                                             int base_n_probe,
                                             int base_search_k,
                                             bool use_mid_filtering = false) {
        auto qbuf = queries.request();
        auto rbuf = ranges.request();
        if (qbuf.ndim != 2) {
            throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
        }
        if (rbuf.ndim != 2 || rbuf.shape[1] != 2 || rbuf.shape[0] != qbuf.shape[0]) {
            throw std::runtime_error("ranges 必须是二维数组 (n_queries, 2)，且长度与 queries 匹配");
        }
        size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
        const float* qdata = static_cast<const float*>(qbuf.ptr);
        const float* rdata = static_cast<const float*>(rbuf.ptr);
        auto out = core_->batch_search_ranges_with_stats(
            qdata, rdata, n_queries, k, base_n_probe, base_search_k, use_mid_filtering);
        return py::make_tuple(
            ada_ivf_py::batch_results_indices_to_python(out.results),
            ada_ivf_py::aggregate_search_stats_to_python(out.per_query_stats));
    }

    PyFineIndexView fine_index() const {
        return PyFineIndexView(core_->fine_core());
    }

    size_t n_vectors() const {
        return core_->n_vectors();
    }
    
    // 动态设置n_probe_f（用于静态基准测试中复用索引）
    void set_n_probe_f(int n_probe_f) {
        core_->set_n_probe_f(n_probe_f);
    }
    
    int get_n_probe_f() const {
        return core_->get_n_probe_f();
    }

    py::dict get_cluster_size_stats() {
        auto stats = core_->get_cluster_size_stats();
        py::dict result;
        result["min_size"] = stats.min_size;
        result["max_size"] = stats.max_size;
        result["avg_size"] = stats.avg_size;
        return result;
    }

    std::vector<size_t> get_all_cluster_sizes() const {
        return core_->get_all_cluster_sizes();
    }

    int global_rebuild_count() const {
        return core_->global_rebuild_count();
    }
    int coarse_rebuild_count() const { return core_->coarse_rebuild_count(); }
    int coarse_soft_sync_count() const { return core_->coarse_soft_sync_count(); }
    int coarse_soft_moved_fines_last() const { return core_->coarse_soft_moved_fines_last(); }
    int coarse_soft_moved_fines_total() const { return core_->coarse_soft_moved_fines_total(); }
    float coarse_soft_moved_ratio_last() const { return core_->coarse_soft_moved_ratio_last(); }
    float coarse_soft_rel_shift_last() const { return core_->coarse_soft_rel_shift_last(); }
    float coarse_soft_avg_radius_last() const { return core_->coarse_soft_avg_radius_last(); }
    size_t adaptive_fanout_query_count() const { return core_->adaptive_fanout_query_count(); }
    size_t adaptive_fanout_extra_coarse_total() const { return core_->adaptive_fanout_extra_coarse_total(); }
    size_t adaptive_fanout_extra_target_total() const { return core_->adaptive_fanout_extra_target_total(); }
    size_t fine_refresh_count() const { return core_->fine_refresh_count(); }
    size_t local_recluster_count() const { return core_->local_recluster_count(); }
    size_t migrated_vector_count() const { return core_->migrated_vector_count(); }
    float maintenance_G_before() const { return core_->maintenance_G_before(); }
    float maintenance_G_after() const { return core_->maintenance_G_after(); }

    double last_graph_rebuild_time_s() const {
        return core_->last_graph_rebuild_time_s();
    }

    void set_maintenance_profile(const std::string& profile) {
        core_->set_maintenance_profile(profile);
    }

    void set_coarse_check_every_insertions(size_t n) {
        core_->set_coarse_check_every_insertions(n);
    }

    void force_coarse_rebuild() {
        core_->force_coarse_rebuild();
    }
    
    py::dict get_imbalance_metrics() {
        auto metrics = core_->get_imbalance_metrics();
        py::dict result;
        result["sigma_c"] = metrics.sigma_c;
        result["sigma_f"] = metrics.sigma_f;  // 细层 σ，与单层 σ 同义，用于对比
        result["epsilon"] = metrics.epsilon_c;  // 使用epsilon作为键名，与单层保持一致
        result["epsilon_prime"] = metrics.epsilon_c_prime;
        result["G_c"] = metrics.G_c;
        result["G"] = metrics.G;  // fine层全局失衡指标，用于对比（含G=0时的fallback）
        result["avg_coarse_size"] = metrics.avg_coarse_size;  // 与sigma_c口径一致
        result["cluster_size_cv"] = metrics.cluster_size_cv;  // sigma_c / avg_coarse_size
        return result;
    }

    void set_enable_scalar_filter_lsm(bool v) { core_->set_enable_scalar_filter_lsm(v); }
    bool get_enable_scalar_filter_lsm() const { return core_->get_enable_scalar_filter_lsm(); }
    void set_enable_scalar_range_prune(bool v) { core_->set_enable_scalar_range_prune(v); }
    bool get_enable_scalar_range_prune() const { return core_->get_enable_scalar_range_prune(); }
    void set_lsm_merge_threshold(size_t t) { core_->set_lsm_merge_threshold(t); }
    size_t get_lsm_merge_threshold() const { return core_->get_lsm_merge_threshold(); }

    // ---------- 删除支持 ----------
    void remove(py::array_t<int> ids) {
        auto buf = ids.request();
        if (buf.ndim != 1 || buf.size == 0) return;
        core_->remove(static_cast<const int*>(buf.ptr), static_cast<size_t>(buf.size));
    }
    size_t get_n_deleted() const { return core_->get_n_deleted(); }
    size_t get_n_live_vectors() const { return core_->get_n_live_vectors(); }
    void compact_deleted_vectors() { core_->compact_deleted_vectors(); }
    size_t compact_deleted_vectors_step(size_t max_clusters) { return core_->compact_deleted_vectors_step(max_clusters); }
    void set_keep_list_packed_payload(bool keep) { core_->set_keep_list_packed_payload(keep); }
    bool get_keep_list_packed_payload() const { return core_->get_keep_list_packed_payload(); }

    // ---------- 直方图剪枝 ----------
    void set_enable_histogram_prune(bool v) { core_->set_enable_histogram_prune(v); }
    bool get_enable_histogram_prune() const { return core_->get_enable_histogram_prune(); }
    void set_histogram_prune_threshold(float t) { core_->set_histogram_prune_threshold(t); }
    float get_histogram_prune_threshold() const { return core_->get_histogram_prune_threshold(); }

    // ---------- 级联探测早停 ----------
    void set_enable_cascade_early_exit(bool v) { core_->set_enable_cascade_early_exit(v); }
    bool get_enable_cascade_early_exit() const { return core_->get_enable_cascade_early_exit(); }
    void set_cascade_early_exit_alpha(float alpha) { core_->set_cascade_early_exit_alpha(alpha); }
    float get_cascade_early_exit_alpha() const { return core_->get_cascade_early_exit_alpha(); }

    // ---------- 自适应双路由 ----------
    void set_enable_adaptive_routing(bool v) { core_->set_enable_adaptive_routing(v); }
    bool get_enable_adaptive_routing() const { return core_->get_enable_adaptive_routing(); }
    void set_adaptive_routing_high_threshold(float t) { core_->set_adaptive_routing_high_threshold(t); }
    float get_adaptive_routing_high_threshold() const { return core_->get_adaptive_routing_high_threshold(); }

    void set_attr_lambda(float v) { core_->set_attr_lambda(v); }
    float get_attr_lambda() const { return core_->get_attr_lambda(); }
    void set_scalar_span_lambda(float v) { core_->set_scalar_span_lambda(v); }
    float get_scalar_span_lambda() const { return core_->get_scalar_span_lambda(); }
    void set_enable_coarse_first_training(bool v) { core_->set_enable_coarse_first_training(v); }
    bool get_enable_coarse_first_training() const { return core_->get_enable_coarse_first_training(); }
    void set_coarse_first_mode(int v) { core_->set_coarse_first_mode(v); }
    int get_coarse_first_mode() const { return core_->get_coarse_first_mode(); }
    void set_kmeans_seed(uint32_t v) { core_->set_kmeans_seed(v); }
    uint32_t get_kmeans_seed() const { return core_->get_kmeans_seed(); }
    void set_use_kmeanspp(bool v) { core_->set_use_kmeanspp(v); }
    bool get_use_kmeanspp() const { return core_->get_use_kmeanspp(); }

    void set_enable_maintenance(bool v) { core_->set_enable_maintenance(v); }
    bool get_enable_maintenance() const { return core_->get_enable_maintenance(); }

    void set_enable_graph_routing(bool v) { core_->set_enable_graph_routing(v); }
    bool get_enable_graph_routing() const { return core_->get_enable_graph_routing(); }
    void set_graph_degree(int k) { core_->set_graph_degree(k); }
    int get_graph_degree() const { return core_->get_graph_degree(); }
    void set_seed_L(int l) { core_->set_seed_L(l); }
    int get_seed_L() const { return core_->get_seed_L(); }
    void set_beam_width(int w) { core_->set_beam_width(w); }
    int get_beam_width() const { return core_->get_beam_width(); }
    void set_early_stop_patience(int p) { core_->set_early_stop_patience(p); }
    int get_early_stop_patience() const { return core_->get_early_stop_patience(); }
    void set_early_stop_relax(float r) { core_->set_early_stop_relax(r); }
    float get_early_stop_relax() const { return core_->get_early_stop_relax(); }
    void set_graph_repair_budget(int v) { core_->set_graph_repair_budget(v); }
    int get_graph_repair_budget() const { return core_->get_graph_repair_budget(); }
    void set_enable_filter_aware_routing(bool v) { core_->set_enable_filter_aware_routing(v); }
    bool get_enable_filter_aware_routing() const { return core_->get_enable_filter_aware_routing(); }
    void set_filter_selectivity_epsilon(float v) { core_->set_filter_selectivity_epsilon(v); }
    float get_filter_selectivity_epsilon() const { return core_->get_filter_selectivity_epsilon(); }
    void set_enable_adaptive_fanout(bool v) { core_->set_enable_adaptive_fanout(v); }
    bool get_enable_adaptive_fanout() const { return core_->get_enable_adaptive_fanout(); }
    void set_filter_fanout_temperature_ratio(float v) { core_->set_filter_fanout_temperature_ratio(v); }
    float get_filter_fanout_temperature_ratio() const { return core_->get_filter_fanout_temperature_ratio(); }

    void rebuild_graph_routing_structures() { core_->rebuild_graph_routing_structures(); }
    void save_graph_routing_cache(const std::string& key) { core_->save_graph_routing_cache(key); }
    bool load_graph_routing_cache(const std::string& key) { return core_->load_graph_routing_cache(key); }
    void clear_graph_routing_cache() { core_->clear_graph_routing_cache(); }
    size_t graph_routing_cache_size() const { return core_->graph_routing_cache_size(); }

    void flush_all_lsm_segments() { core_->flush_all_lsm_segments(); }

    py::array_t<float> get_fine_centroids() const {
        auto cents = core_->get_fine_centroids();
        if (cents.empty()) {
            // pybind11 在 GCC4.8 下对 {0,0} 形状的构造容易产生重载歧义；
            // 这里用 buffer_info 显式构造一个 (0,0) 的二维数组。
            py::buffer_info info(
                nullptr,
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                std::vector<py::ssize_t>{0, 0},
                std::vector<py::ssize_t>{0, 0});
            return py::array_t<float>(info);
        }
        const size_t n = cents.size();
        const size_t dim = cents[0].size();
        py::array_t<float> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(dim)});
        auto buf = out.request();
        float* ptr = static_cast<float*>(buf.ptr);
        for (size_t i = 0; i < n; ++i) {
            const auto& row = cents[i];
            for (size_t d = 0; d < dim; ++d) {
                ptr[i * dim + d] = (d < row.size() ? row[d] : 0.0f);
            }
        }
        return out;
    }

    int find_fine_cluster_for_vec_id(int vec_id) const {
        return core_->find_fine_cluster_for_vec_id(vec_id);
    }

    void set_enable_pq_compression(bool v) { core_->set_enable_pq_compression(v); }
    bool get_enable_pq_compression() const { return core_->get_enable_pq_compression(); }
    void set_pq_m(int m) { core_->set_pq_m(m); }
    int get_pq_m() const { return core_->get_pq_m(); }
    void set_pq_ksub(int k) { core_->set_pq_ksub(k); }
    int get_pq_ksub() const { return core_->get_pq_ksub(); }
    void set_pq_release_floats(bool v) { core_->set_pq_release_floats(v); }
    bool get_pq_release_floats() const { return core_->get_pq_release_floats(); }
    bool is_pq_trained() const { return core_->is_pq_trained(); }

    void train_pq_from_samples(py::array_t<float> vectors) {
        auto buf = vectors.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("train_pq_from_samples: 需要 2 维数组 (n_vectors, dimension)");
        }
        const float* data = static_cast<const float*>(buf.ptr);
        size_t n = static_cast<size_t>(buf.shape[0]);
        int dim = static_cast<int>(buf.shape[1]);
        core_->train_pq_from_samples(data, n, dim);
    }

private:
    std::unique_ptr<HierarchicalAdaIVFCore> core_;
};

// 由 CMake 为不同 target 传入 HIER_PYBIND_MODULE_NAME（v1: hierarchical_ada_ivf_cpp，v2: hierarchical_ada_ivf_cpp_v2）
#ifndef HIER_PYBIND_MODULE_NAME
#define HIER_PYBIND_MODULE_NAME hierarchical_ada_ivf_cpp_v2
#endif
PYBIND11_MODULE(HIER_PYBIND_MODULE_NAME, m) {
    m.doc() = "C++ 实现的层次化 Ada-IVF （fine/coarse 两级 IVF + 范围过滤）";

    py::class_<PyFineIndexView>(m, "FineIndexView")
        .def("batch_search", &PyFineIndexView::batch_search,
             py::arg("queries"),
             py::arg("k"),
             py::arg("max_num_distances"),
             py::arg("range_min"),
             py::arg("range_max"),
             "细层直搜：复用 AdaIVFCore::batch_search，返回 indices_list（int32 numpy，按距离升序）")
        .def("batch_search_ranges", &PyFineIndexView::batch_search_ranges,
             py::arg("queries"),
             py::arg("ranges"),
             py::arg("k"),
             py::arg("max_num_distances"),
             "细层直搜：每条 query 独立 range，复用 AdaIVFCore::batch_search_ranges")
        .def("batch_search_with_stats", &PyFineIndexView::batch_search_with_stats,
             py::arg("queries"),
             py::arg("k"),
             py::arg("max_num_distances"),
             py::arg("range_min"),
             py::arg("range_max"),
             "细层直搜：返回 (indices_list, stats_dict)")
        .def("batch_search_ranges_with_stats", &PyFineIndexView::batch_search_ranges_with_stats,
             py::arg("queries"),
             py::arg("ranges"),
             py::arg("k"),
             py::arg("max_num_distances"),
             "细层直搜：每条 query 独立 range，返回 (indices_list, stats_dict)")
        .def_property("n_probe", &PyFineIndexView::n_probe, &PyFineIndexView::set_n_probe,
                      "fine 层 n_probe")
        .def_property("enable_scalar_range_prune", &PyFineIndexView::get_enable_scalar_range_prune,
                      &PyFineIndexView::set_enable_scalar_range_prune,
                      "fine 层簇级 scalar min/max 粗剪枝");

    py::class_<PyHierarchicalAdaIVFIndex>(m, "HierarchicalAdaIVFIndex")
        .def(py::init<int, int, int, size_t, float, float, int>(),
             py::arg("n_fine_clusters"),
             py::arg("n_coarse_clusters"),
             py::arg("n_probe"),
             py::arg("max_cluster_size"),
             py::arg("recluster_threshold"),
             py::arg("centroid_update_threshold"),
             py::arg("n_probe_f") = -1)  // 细层n_probe，-1表示不限制

        .def("train", &PyHierarchicalAdaIVFIndex::train,
             py::arg("vectors"),
             py::arg("scalars") = py::none(),
             "使用训练向量构建 fine / coarse 质心")

        .def("add", &PyHierarchicalAdaIVFIndex::add,
             py::arg("vectors"),
             py::arg("ids") = py::array_t<int>(),
             py::arg("auto_recluster") = true,
             py::arg("scalars") = py::array_t<float>(),
             "插入基础向量（带可选 ID 与标量）")

        .def("batch_search", &PyHierarchicalAdaIVFIndex::batch_search,
             py::arg("queries"),
             py::arg("k"),
             py::arg("base_n_probe"),
             py::arg("base_search_k"),
             py::arg("range_min") = py::none(),
             py::arg("range_max") = py::none(),
             py::arg("use_mid_filtering") = false,
             "批量查询：返回 indices_list（int32 numpy，按距离升序）")

        .def("batch_search_ranges", &PyHierarchicalAdaIVFIndex::batch_search_ranges,
             py::arg("queries"),
             py::arg("ranges"),
             py::arg("k"),
             py::arg("base_n_probe"),
             py::arg("base_search_k"),
             py::arg("use_mid_filtering") = false,
             "批量查询（每条 query 独立 range）：与 fine 层 batch_search_ranges 口径一致")
        .def("batch_search_with_stats", &PyHierarchicalAdaIVFIndex::batch_search_with_stats,
             py::arg("queries"),
             py::arg("k"),
             py::arg("base_n_probe"),
             py::arg("base_search_k"),
             py::arg("range_min") = py::none(),
             py::arg("range_max") = py::none(),
             py::arg("use_mid_filtering") = false,
             "批量查询：返回 (indices_list, stats_dict)")
        .def("batch_search_ranges_with_stats", &PyHierarchicalAdaIVFIndex::batch_search_ranges_with_stats,
             py::arg("queries"),
             py::arg("ranges"),
             py::arg("k"),
             py::arg("base_n_probe"),
             py::arg("base_search_k"),
             py::arg("use_mid_filtering") = false,
             "批量查询（每条 query 独立 range）：返回 (indices_list, stats_dict)")

        .def_property_readonly("fine_index", &PyHierarchicalAdaIVFIndex::fine_index,
                               "细层 AdaIVFCore 搜索视图（batch_search / n_probe，与单层同逻辑）")

        .def_property_readonly("n_vectors", &PyHierarchicalAdaIVFIndex::n_vectors,
                               "当前已插入的向量数量（按 add 次数统计）")
        
        .def_property("n_probe_f", &PyHierarchicalAdaIVFIndex::get_n_probe_f, 
                      &PyHierarchicalAdaIVFIndex::set_n_probe_f,
                      "细层n_probe（可动态修改，用于静态基准测试）")

        .def("get_cluster_size_stats", &PyHierarchicalAdaIVFIndex::get_cluster_size_stats,
             "获取fine层真实聚簇大小统计（min/max/avg）")
        .def("get_all_cluster_sizes", &PyHierarchicalAdaIVFIndex::get_all_cluster_sizes,
             "获取fine层每个簇的真实大小，用于分布统计")

        .def_property_readonly("global_rebuild_count", &PyHierarchicalAdaIVFIndex::global_rebuild_count,
                               "coarse层全局重建累计次数")
        .def_property_readonly("coarse_rebuild_count", &PyHierarchicalAdaIVFIndex::coarse_rebuild_count,
                               "coarse 层重建累计次数")
        .def_property_readonly("coarse_soft_sync_count", &PyHierarchicalAdaIVFIndex::coarse_soft_sync_count,
                               "coarse 层软同步累计次数")
        .def_property_readonly("coarse_soft_moved_fines_last", &PyHierarchicalAdaIVFIndex::coarse_soft_moved_fines_last,
                               "最近一次 coarse 软同步迁移 fine 数")
        .def_property_readonly("coarse_soft_moved_fines_total", &PyHierarchicalAdaIVFIndex::coarse_soft_moved_fines_total,
                               "coarse 软同步累计迁移 fine 数")
        .def_property_readonly("coarse_soft_moved_ratio_last", &PyHierarchicalAdaIVFIndex::coarse_soft_moved_ratio_last,
                               "最近一次 coarse 软同步迁移 fine 比例")
        .def_property_readonly("coarse_soft_rel_shift_last", &PyHierarchicalAdaIVFIndex::coarse_soft_rel_shift_last,
                               "最近一次 coarse 软同步相对质心漂移")
        .def_property_readonly("coarse_soft_avg_radius_last", &PyHierarchicalAdaIVFIndex::coarse_soft_avg_radius_last,
                               "最近一次 coarse 软同步平均半径")
        .def_property_readonly("adaptive_fanout_query_count", &PyHierarchicalAdaIVFIndex::adaptive_fanout_query_count,
                               "adaptive fanout 累计触发 query 数")
        .def_property_readonly("adaptive_fanout_extra_coarse_total", &PyHierarchicalAdaIVFIndex::adaptive_fanout_extra_coarse_total,
                               "adaptive fanout 累计额外 coarse 数")
        .def_property_readonly("adaptive_fanout_extra_target_total", &PyHierarchicalAdaIVFIndex::adaptive_fanout_extra_target_total,
                               "adaptive fanout 累计额外 fine 目标数")
        .def_property_readonly("fine_refresh_count", &PyHierarchicalAdaIVFIndex::fine_refresh_count,
                               "fine 质心刷新/边界修复累计次数")
        .def_property_readonly("local_recluster_count", &PyHierarchicalAdaIVFIndex::local_recluster_count,
                               "fine 局部重聚类累计次数")
        .def_property_readonly("migrated_vector_count", &PyHierarchicalAdaIVFIndex::migrated_vector_count,
                               "受控边界迁移累计向量数")
        .def_property_readonly("maintenance_G_before", &PyHierarchicalAdaIVFIndex::maintenance_G_before,
                               "最近一次 fine 维护前 G")
        .def_property_readonly("maintenance_G_after", &PyHierarchicalAdaIVFIndex::maintenance_G_after,
                               "最近一次 fine 维护后 G")
        .def_property_readonly("last_graph_rebuild_time_s", &PyHierarchicalAdaIVFIndex::last_graph_rebuild_time_s,
                               "最近一次 add 内实际图/coarse 重建耗时（秒）")

        .def("set_maintenance_profile", &PyHierarchicalAdaIVFIndex::set_maintenance_profile,
             py::arg("profile"),
             "维护档位: throughput | balanced | large_recall（调整 G_c 阈值与检查间隔）")
        .def("set_coarse_check_every_insertions", &PyHierarchicalAdaIVFIndex::set_coarse_check_every_insertions,
             py::arg("n_insertions"),
             "累计插入多少向量后做一次 coarse 层 G_c 检查")
        .def("force_coarse_rebuild", &PyHierarchicalAdaIVFIndex::force_coarse_rebuild,
             "强制 coarse 全局重建（重训 coarse 映射与 fine 质心图，不重插向量）")
        
        .def("get_imbalance_metrics", &PyHierarchicalAdaIVFIndex::get_imbalance_metrics,
             "获取失衡指标（sigma_c, epsilon, epsilon_prime, G_c）")

        .def_property("enable_scalar_filter_lsm", &PyHierarchicalAdaIVFIndex::get_enable_scalar_filter_lsm,
                      &PyHierarchicalAdaIVFIndex::set_enable_scalar_filter_lsm,
                      "Module A：IVF 桶内标量有序段+无序段（LSM 风格），默认 False")
        .def_property("enable_scalar_range_prune", &PyHierarchicalAdaIVFIndex::get_enable_scalar_range_prune,
                      &PyHierarchicalAdaIVFIndex::set_enable_scalar_range_prune,
                      "簇级 scalar min/max 粗剪枝，默认 False")
        .def_property("lsm_merge_threshold", &PyHierarchicalAdaIVFIndex::get_lsm_merge_threshold,
                      &PyHierarchicalAdaIVFIndex::set_lsm_merge_threshold,
                      "Module A：无序段达该条数触发并入有序段，默认 100")
        .def_property("attr_lambda", &PyHierarchicalAdaIVFIndex::get_attr_lambda,
                      &PyHierarchicalAdaIVFIndex::set_attr_lambda,
                      "deprecated/no-op：训练固定为纯向量 KMeans")
        .def_property("scalar_span_lambda", &PyHierarchicalAdaIVFIndex::get_scalar_span_lambda,
                      &PyHierarchicalAdaIVFIndex::set_scalar_span_lambda,
                      "deprecated/no-op：scalar 不进入训练")
        .def_property("enable_coarse_first_training", &PyHierarchicalAdaIVFIndex::get_enable_coarse_first_training,
                      &PyHierarchicalAdaIVFIndex::set_enable_coarse_first_training,
                      "deprecated/no-op：保留兼容")
        .def_property("coarse_first_mode", &PyHierarchicalAdaIVFIndex::get_coarse_first_mode,
                      &PyHierarchicalAdaIVFIndex::set_coarse_first_mode,
                      "deprecated/no-op：保留兼容")
        .def_property("kmeans_seed", &PyHierarchicalAdaIVFIndex::get_kmeans_seed,
                      &PyHierarchicalAdaIVFIndex::set_kmeans_seed,
                      "K-means++ 固定随机种子，默认 42")
        .def_property("use_kmeanspp", &PyHierarchicalAdaIVFIndex::get_use_kmeanspp,
                      &PyHierarchicalAdaIVFIndex::set_use_kmeanspp,
                      "是否使用 K-means++ 初始化，默认 True")
        .def_property("enable_maintenance", &PyHierarchicalAdaIVFIndex::get_enable_maintenance,
                      &PyHierarchicalAdaIVFIndex::set_enable_maintenance,
                      "簇维护总开关：fine 重聚类 / coarse 重建 / 插入期图同步，默认 False（纯 IVF）")
        .def_property("enable_graph_routing", &PyHierarchicalAdaIVFIndex::get_enable_graph_routing,
                      &PyHierarchicalAdaIVFIndex::set_enable_graph_routing,
                      "Module B（新）：细层中心图搜索路由开关。True=图搜索；False=旧 coarse->fine 映射")
        .def_property("graph_degree", &PyHierarchicalAdaIVFIndex::get_graph_degree,
                      &PyHierarchicalAdaIVFIndex::set_graph_degree,
                      "fine 质心 kNN 图出度 k（默认 8，建图/重建时生效）")
        .def_property("seed_L", &PyHierarchicalAdaIVFIndex::get_seed_L,
                      &PyHierarchicalAdaIVFIndex::set_seed_L,
                      "每个 coarse 的 seed fine 数（默认 4）")
        .def_property("beam_width", &PyHierarchicalAdaIVFIndex::get_beam_width,
                      &PyHierarchicalAdaIVFIndex::set_beam_width,
                      "图 beam 每步最大扩展邻居数（默认 8）")
        .def_property("early_stop_patience", &PyHierarchicalAdaIVFIndex::get_early_stop_patience,
                      &PyHierarchicalAdaIVFIndex::set_early_stop_patience,
                      "图 beam early-stop 耐心（默认 8）")
        .def_property("early_stop_relax", &PyHierarchicalAdaIVFIndex::get_early_stop_relax,
                      &PyHierarchicalAdaIVFIndex::set_early_stop_relax,
                      "图 beam early-stop 相对松弛系数（默认 1.05）")
        .def_property("graph_repair_budget", &PyHierarchicalAdaIVFIndex::get_graph_repair_budget,
                      &PyHierarchicalAdaIVFIndex::set_graph_repair_budget,
                      "残差图修复最大扩展预算（默认 8；设为 0 时自动计算）")
        .def_property("enable_filter_aware_routing", &PyHierarchicalAdaIVFIndex::get_enable_filter_aware_routing,
                      &PyHierarchicalAdaIVFIndex::set_enable_filter_aware_routing,
                      "range 查询时启用 filter-aware fine routing，默认 True")
        .def_property("enable_adaptive_fanout", &PyHierarchicalAdaIVFIndex::get_enable_adaptive_fanout,
                      &PyHierarchicalAdaIVFIndex::set_enable_adaptive_fanout,
                      "启用 adaptive fanout 预算分配，默认 False")
        .def_property("filter_selectivity_epsilon", &PyHierarchicalAdaIVFIndex::get_filter_selectivity_epsilon,
                      &PyHierarchicalAdaIVFIndex::set_filter_selectivity_epsilon,
                      "filter-aware route score 的 selectivity bias 强度，默认 0.25")
        .def_property("filter_fanout_temperature_ratio", &PyHierarchicalAdaIVFIndex::get_filter_fanout_temperature_ratio,
                      &PyHierarchicalAdaIVFIndex::set_filter_fanout_temperature_ratio,
                      "adaptive fanout 中 coarse 距离温度比例，默认 0.15")
        .def("rebuild_graph_routing_structures", &PyHierarchicalAdaIVFIndex::rebuild_graph_routing_structures,
             "按当前 graph_degree/seed_L 在同一质心上重建 seeds + fine 质心图")
        .def("save_graph_routing_cache", &PyHierarchicalAdaIVFIndex::save_graph_routing_cache,
             py::arg("key"), "缓存当前图路由结构（fine 图 + coarse seeds）")
        .def("load_graph_routing_cache", &PyHierarchicalAdaIVFIndex::load_graph_routing_cache,
             py::arg("key"), "恢复已缓存的图路由结构；不存在返回 False")
        .def("clear_graph_routing_cache", &PyHierarchicalAdaIVFIndex::clear_graph_routing_cache,
             "清空图路由结构缓存")
        .def("graph_routing_cache_size", &PyHierarchicalAdaIVFIndex::graph_routing_cache_size,
             "当前已缓存的图路由结构数量")
        .def("flush_all_lsm_segments", &PyHierarchicalAdaIVFIndex::flush_all_lsm_segments,
             "将所有 fine 桶的无序段并入有序段（导出/保存前建议调用）")
        .def("get_fine_centroids", &PyHierarchicalAdaIVFIndex::get_fine_centroids,
             "导出 fine 层 IVF 质心矩阵 (n_fine_clusters, dim)，用于诊断")
        .def("find_fine_cluster_for_vec_id", &PyHierarchicalAdaIVFIndex::find_fine_cluster_for_vec_id,
             py::arg("vec_id"),
             "诊断：vec_id 当前所在 fine IVF 簇；不存在返回 -1（内部 id 与按序插入下标一致）")
        .def_property("enable_pq_compression", &PyHierarchicalAdaIVFIndex::get_enable_pq_compression,
                      &PyHierarchicalAdaIVFIndex::set_enable_pq_compression,
                      "Module PQ（fine 层）：PQ 压缩向量存储，默认 False")
        .def_property("pq_m", &PyHierarchicalAdaIVFIndex::get_pq_m, &PyHierarchicalAdaIVFIndex::set_pq_m,
                      "PQ 段数 m，默认 8")
        .def_property("pq_ksub", &PyHierarchicalAdaIVFIndex::get_pq_ksub, &PyHierarchicalAdaIVFIndex::set_pq_ksub,
                      "每段 ksub，默认 256")
        .def_property("pq_release_floats", &PyHierarchicalAdaIVFIndex::get_pq_release_floats,
                      &PyHierarchicalAdaIVFIndex::set_pq_release_floats,
                      "PQ 后是否释放原始 float 向量（默认 True：省内存但召回可能下降；False 可用全精度距离精排）")
        .def_property_readonly("is_pq_trained", &PyHierarchicalAdaIVFIndex::is_pq_trained,
                               "fine 层 PQ 码本是否已训练")
        .def("train_pq_from_samples", &PyHierarchicalAdaIVFIndex::train_pq_from_samples,
             py::arg("vectors"),
             "在 fine 层上用样本向量训练/刷新 PQ 码本")

        // ---------- 删除支持 ----------
        .def("remove", &PyHierarchicalAdaIVFIndex::remove,
             py::arg("ids"),
             "将 ids（内部 vec_id，int32 numpy）标记为已删除（查询时跳过）")
        .def("compact_deleted_vectors", &PyHierarchicalAdaIVFIndex::compact_deleted_vectors,
             "从 IVF 列表中物理移除已标记删除向量（降内存；大量删除后调用）")
        .def("compact_deleted_vectors_step", &PyHierarchicalAdaIVFIndex::compact_deleted_vectors_step,
             py::arg("max_clusters") = 64,
             "渐进式物理清除 tombstone：最多扫描 max_clusters 个 IVF 桶，返回本次移除数量")
        .def_property("keep_list_packed_payload",
                      &PyHierarchicalAdaIVFIndex::get_keep_list_packed_payload,
                      &PyHierarchicalAdaIVFIndex::set_keep_list_packed_payload,
                      "是否在 IVF list 内保留向量副本用于 packed SIMD 扫描；删除 workload 可关闭以降内存")
        .def_property_readonly("n_deleted", &PyHierarchicalAdaIVFIndex::get_n_deleted,
                               "当前 tombstone 计数")
        .def_property_readonly("n_live_vectors", &PyHierarchicalAdaIVFIndex::get_n_live_vectors,
                               "当前活跃向量数（n_vectors - n_deleted）")

        // ---------- 直方图剪枝 ----------
        .def_property("enable_histogram_prune",
                      &PyHierarchicalAdaIVFIndex::get_enable_histogram_prune,
                      &PyHierarchicalAdaIVFIndex::set_enable_histogram_prune,
                      "启用 16-bucket 标量直方图剪枝（在 LSM merge 后自动重建直方图），默认 False")
        .def_property("histogram_prune_threshold",
                      &PyHierarchicalAdaIVFIndex::get_histogram_prune_threshold,
                      &PyHierarchicalAdaIVFIndex::set_histogram_prune_threshold,
                      "直方图剪枝阈值：估计选择率 < threshold 时跳过该簇，默认 0.005")

        // ---------- 级联早停 ----------
        .def_property("enable_cascade_early_exit",
                      &PyHierarchicalAdaIVFIndex::get_enable_cascade_early_exit,
                      &PyHierarchicalAdaIVFIndex::set_enable_cascade_early_exit,
                      "启用级联探测早停：heap 满且 worst_dist < alpha * next_centroid_dist 时停止，默认 False")
        .def_property("cascade_early_exit_alpha",
                      &PyHierarchicalAdaIVFIndex::get_cascade_early_exit_alpha,
                      &PyHierarchicalAdaIVFIndex::set_cascade_early_exit_alpha,
                      "级联早停松弛因子 alpha，默认 1.0")

        // ---------- 自适应双路由 ----------
        .def_property("enable_adaptive_routing",
                      &PyHierarchicalAdaIVFIndex::get_enable_adaptive_routing,
                      &PyHierarchicalAdaIVFIndex::set_enable_adaptive_routing,
                      "启用选择率自适应路由：宽范围查询跳过图路由走快速平面路由，默认 False")
        .def_property("adaptive_routing_high_threshold",
                      &PyHierarchicalAdaIVFIndex::get_adaptive_routing_high_threshold,
                      &PyHierarchicalAdaIVFIndex::set_adaptive_routing_high_threshold,
                      "自适应路由选择率阈值：> threshold 时认为是宽范围走快路径，默认 0.15");
}
