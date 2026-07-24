#include "../include/ada_ivf_core.h"
#include "../include/simd_utils.h"
#include "../include/ivf_topk_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <random>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cstring>
#include <climits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#include <climits>
#ifdef _OPENMP
#include <omp.h>
#endif
#if USE_AVX
#include <immintrin.h>
#endif
#if ADA_IVF_DEBUG
#define ADA_IVF_MAINT_DEBUG(stmt) do { stmt; } while (0)
#else
#define ADA_IVF_MAINT_DEBUG(stmt) do {} while (0)
#endif

namespace {

// OpenMP 线程数：ADA_IVF_OMP_THREADS > ADA_IVF_TRAIN_OMP_THREADS > OMP_NUM_THREADS > 默认 32
#ifdef _OPENMP
constexpr int kAdaIvfDefaultOmpThreads = 32;

int resolve_ada_ivf_omp_threads() {
    const char* ada_env = std::getenv("ADA_IVF_OMP_THREADS");
    if (ada_env && *ada_env) {
        int n = std::atoi(ada_env);
        if (n > 0) return n;
    }
    const char* train_env = std::getenv("ADA_IVF_TRAIN_OMP_THREADS");
    if (train_env && *train_env) {
        int n = std::atoi(train_env);
        if (n > 0) return n;
    }
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env && *omp_env) {
        int n = std::atoi(omp_env);
        if (n > 0) return n;
    }
    return kAdaIvfDefaultOmpThreads;
}

// 训练专用：ADA_IVF_TRAIN_OMP_THREADS 优先，便于 insert 设 ADA_IVF_OMP_THREADS=1 时 train 仍多线程
int resolve_ada_ivf_train_omp_threads() {
    const char* train_env = std::getenv("ADA_IVF_TRAIN_OMP_THREADS");
    if (train_env && *train_env) {
        int n = std::atoi(train_env);
        if (n > 0) return n;
    }
    return resolve_ada_ivf_omp_threads();
}

void apply_ada_ivf_omp_threads(const char* tag) {
    const int n = resolve_ada_ivf_omp_threads();
    omp_set_num_threads(n);
    if (tag && *tag) {
        std::cerr << "[OpenMP] " << tag << " threads=" << n << std::endl;
    }
}

void apply_ada_ivf_train_omp_threads(const char* tag) {
    const int n = resolve_ada_ivf_train_omp_threads();
    omp_set_num_threads(n);
    if (tag && *tag) {
        std::cerr << "[OpenMP] " << tag << " threads=" << n << std::endl;
    }
}

void log_kmeans_train_omp_threads() {
    apply_ada_ivf_train_omp_threads("K-means训练");
}
#endif

inline bool ivf_profile_search_enabled() {
    const char* env = std::getenv("IVF_PROFILE_SEARCH");
    if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
        return true;
    }
    env = std::getenv("HIER_ADA_IVF_PROFILE_SEARCH");
    return env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
}

inline int ivf_profile_search_max_queries() {
    const char* env = std::getenv("IVF_PROFILE_SEARCH_MAX");
    if (!env || !env[0]) {
        env = std::getenv("HIER_ADA_IVF_PROFILE_SEARCH_MAX");
    }
    if (!env || !env[0]) {
        return INT_MAX;
    }
    const int n = std::atoi(env);
    if (n <= 0) {
        return INT_MAX;
    }
    return n;
}

void ivf_profile_log_append_line(const std::string& line) {
    const char* path = std::getenv("IVF_PROFILE_LOG");
    if (!path || !path[0]) {
        return;
    }
    std::ofstream ofs(path, std::ios::out | std::ios::app);
    if (ofs) {
        ofs << line << '\n';
    }
}

void dump_single_search_profile(
    int qidx,
    int n_probe,
    double ms_fine_score,
    double ms_fine_select,
    double ms_search_clusters,
    int candidate_count,
    int result_count) {
    const double ms_candidate = ms_fine_score + ms_fine_select;
    const double ms_total = ms_candidate + ms_search_clusters;
    std::ostringstream oss;
    oss << "[search_single耗时] q=" << qidx
        << " mode=single graph_routing=-1"
        << " total=" << ms_total << "ms"
        << ", candidate_total=" << ms_candidate << "ms"
        << ", coarse_score=0ms"
        << ", coarse_select=0ms"
        << ", route_union=0ms"
        << ", route_B_off=0ms"
        << ", route_B_on=0ms"
        << ", b_on_allow_coarse=0ms"
        << ", b_on_seed_build=0ms"
        << ", b_on_seed_fallback=0ms"
        << ", b_on_pq_init=0ms"
        << ", b_on_loop_total=0ms"
        << ", b_on_heap_update=0ms"
        << ", b_on_expand_gate=0ms"
        << ", b_on_neighbor_iter=0ms"
        << ", b_on_alignment=0ms"
        << ", b_on_push=0ms"
        << ", b_on_finalize=0ms"
        << ", b_on_finalize_sort=0ms"
        << ", fine_cluster_dist=" << ms_fine_score << "ms"
        << ", fine_cluster_sort=" << ms_fine_select << "ms(lazy=0)"
        << ", sort_candidates=0ms"
        << ", search_clusters=" << ms_search_clusters << "ms"
        << ", post_sqrt=0ms"
        << ", b_on_loop_iters=0"
        << ", b_on_neighbor_checks=0"
        << ", b_on_pushes=0"
        << ", nc=0"
        << ", nf_target=" << n_probe
        << ", candidate_count=" << candidate_count
        << ", result_count=" << result_count;
    const std::string line = oss.str();
    std::cerr << line << std::endl;
    ivf_profile_log_append_line(line);
}

std::atomic<int>& ivf_profile_query_counter() {
    static std::atomic<int> counter{0};
    return counter;
}

inline void ivf_profile_search_maybe_reset() {
    const char* env = std::getenv("IVF_PROFILE_SEARCH_RESET");
    if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
        ivf_profile_query_counter().store(0);
    }
}

inline int ivf_profile_search_claim_query_index() {
    return ivf_profile_query_counter().fetch_add(1);
}

inline double ivf_ms_between(
    const std::chrono::high_resolution_clock::time_point& t0,
    const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void dump_profile0608_line(
    const char* mode,
    int qidx,
    int nc,
    int nf,
    double coarse_select_ms,
    double fine_select_ms,
    double search_clusters_ms,
    double total_ms,
    int candidate_count) {
    const double coarse_pct =
        (total_ms > 0.0) ? (100.0 * coarse_select_ms / total_ms) : 0.0;
    const double fine_pct =
        (total_ms > 0.0) ? (100.0 * fine_select_ms / total_ms) : 0.0;
    const double search_pct =
        (total_ms > 0.0) ? (100.0 * search_clusters_ms / total_ms) : 0.0;
    std::cerr << "[profile0608] mode=" << mode
              << " q=" << qidx
              << " nc=" << nc
              << " nf=" << nf
              << " coarse_select_ms=" << coarse_select_ms
              << " fine_select_ms=" << fine_select_ms
              << " search_clusters_ms=" << search_clusters_ms
              << " total_ms=" << total_ms
              << " coarse_pct=" << coarse_pct
              << " fine_pct=" << fine_pct
              << " search_pct=" << search_pct
              << " candidate_count=" << candidate_count
              << std::endl;
}

// 扫描期 max-heap：保留距离最小的 capacity 个候选；dist >= heap.top 时剪枝
struct MaxHeapAnnCollector {
    explicit MaxHeapAnnCollector(int capacity) : capacity_(std::max(0, capacity)) {
        if (capacity_ > 0) {
            heap_.reserve(static_cast<size_t>(capacity_) + 1);
        }
    }

    bool is_prunable(float dist_sq) const {
        if (capacity_ <= 0 || static_cast<int>(heap_.size()) < capacity_) {
            return false;
        }
        return dist_sq >= heap_.front().first;
    }

    void try_push(float dist_sq, int vec_id) {
        if (capacity_ <= 0) {
            return;
        }
        if (static_cast<int>(heap_.size()) < capacity_) {
            heap_.emplace_back(dist_sq, vec_id);
            std::push_heap(heap_.begin(), heap_.end(), cmp_max_by_dist);
            return;
        }
        if (dist_sq >= heap_.front().first) {
            return;
        }
        std::pop_heap(heap_.begin(), heap_.end(), cmp_max_by_dist);
        heap_.pop_back();
        heap_.emplace_back(dist_sq, vec_id);
        std::push_heap(heap_.begin(), heap_.end(), cmp_max_by_dist);
    }

    int size() const { return static_cast<int>(heap_.size()); }
    bool is_full() const { return capacity_ > 0 && static_cast<int>(heap_.size()) >= capacity_; }

    std::vector<std::pair<float, int>> extract_top_k(int k) const {
        std::vector<std::pair<float, int>> out = heap_;
        auto by_dist = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        };
        const int result_size = std::min(k, static_cast<int>(out.size()));
        if (result_size <= 0) {
            return {};
        }
        if (static_cast<int>(out.size()) > result_size) {
            std::partial_sort(out.begin(), out.begin() + result_size, out.end(), by_dist);
            out.resize(static_cast<size_t>(result_size));
        } else {
            std::sort(out.begin(), out.end(), by_dist);
        }
        return out;
    }

private:
    static bool cmp_max_by_dist(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    }

    int capacity_;
    std::vector<std::pair<float, int>> heap_;
};

void gather_main_range_ids(const RangeAwareList& list, float range_min, float range_max,
                           std::vector<uint32_t>& out) {
    const auto pr = list.binary_search_range(range_min, range_max);
    out.insert(out.end(), list.main_ids.begin() + static_cast<std::ptrdiff_t>(pr.first),
               list.main_ids.begin() + static_cast<std::ptrdiff_t>(pr.second));
}

void gather_delta_range_ids_simd(const RangeAwareList& list, float range_min, float range_max,
                                 std::vector<uint32_t>& out) {
    const size_t n = list.delta_ids.size();
    if (n == 0) {
        return;
    }
    out.reserve(out.size() + n);
#if USE_AVX
    const __m128 vmin = _mm_set1_ps(range_min);
    const __m128 vmax = _mm_set1_ps(range_max);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 s = _mm_loadu_ps(list.delta_scalars.data() + i);
        const __m128 ge = _mm_cmpge_ps(s, vmin);
        const __m128 le = _mm_cmple_ps(s, vmax);
        const int mask = _mm_movemask_ps(_mm_and_ps(ge, le));
        for (int j = 0; j < 4; ++j) {
            if (mask & (1 << j)) {
                out.push_back(list.delta_ids[i + static_cast<size_t>(j)]);
            }
        }
    }
    for (; i < n; ++i) {
        const float s = list.delta_scalars[i];
        if (s >= range_min && s <= range_max) {
            out.push_back(list.delta_ids[i]);
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        const float s = list.delta_scalars[i];
        if (s >= range_min && s <= range_max) {
            out.push_back(list.delta_ids[i]);
        }
    }
#endif
}

void gather_cluster_all_ids(const RangeAwareList& list, std::vector<uint32_t>& out) {
    out.insert(out.end(), list.main_ids.begin(), list.main_ids.end());
    out.insert(out.end(), list.delta_ids.begin(), list.delta_ids.end());
}

void scan_delta_rows_direct(const RangeAwareList& list,
                            bool use_range_filter,
                            float range_min,
                            float range_max,
                            const float* query,
                            int dim,
                            float query_norm_sq,
                            const std::vector<float>& vectors_flat,
                            const std::vector<float>& vector_norms,
                            size_t n_vectors,
                            MaxHeapAnnCollector& collector,
                            int& dist_computed,
                            int max_num_distances,
                            const std::unordered_set<uint32_t>* tombstones,
                            size_t* scalar_rows_scanned,
                            size_t* scalar_blocks_skipped,
                            size_t* scalar_block_rows_skipped,
                            size_t* filtered_candidates) {
    const size_t n = list.delta_ids.size();
    if (n == 0 || dim <= 0) {
        return;
    }
    const size_t dim_sz = static_cast<size_t>(dim);
    const bool delta_packed =
        list.dimension_ > 0 && list.delta_vecs.size() == n * dim_sz &&
        list.delta_norms_sq.size() == n;
    const bool can_use_flat = !vectors_flat.empty();

    for (size_t i = 0; i < n && dist_computed < max_num_distances; ++i) {
        if (use_range_filter) {
            const size_t block = i / RangeAwareList::kDeltaBlockSize;
            if ((i % RangeAwareList::kDeltaBlockSize) == 0 &&
                !list.delta_block_may_overlap(block, range_min, range_max)) {
                const size_t next = std::min(n, (block + 1) * RangeAwareList::kDeltaBlockSize);
                if (scalar_rows_scanned != nullptr) *scalar_rows_scanned += (next - i);
                if (scalar_blocks_skipped != nullptr) ++(*scalar_blocks_skipped);
                if (scalar_block_rows_skipped != nullptr) *scalar_block_rows_skipped += (next - i);
                i = next - 1;
                continue;
            }
            if (scalar_rows_scanned != nullptr) ++(*scalar_rows_scanned);
            const float s = list.delta_scalars[i];
            if (s < range_min || s > range_max) {
                continue;
            }
        }

        const uint32_t vec_id = list.delta_ids[i];
        if (filtered_candidates != nullptr) ++(*filtered_candidates);
        if (tombstones != nullptr && tombstones->count(vec_id) != 0) continue;
        if (vec_id >= static_cast<uint32_t>(n_vectors)) {
            continue;
        }

        const float* x = nullptr;
        float norm_half = 0.0f;
        if (delta_packed) {
            x = list.delta_vecs.data() + i * dim_sz;
            norm_half = list.delta_norms_sq[i] * 0.5f;
        } else if (can_use_flat &&
                   vectors_flat.size() >= (static_cast<size_t>(vec_id) + 1) * dim_sz &&
                   vec_id < vector_norms.size()) {
            x = vectors_flat.data() + static_cast<size_t>(vec_id) * dim_sz;
            norm_half = vector_norms[vec_id];
        }
        if (x == nullptr) {
            continue;
        }

        const float dist_score = -L2SIMD4ExtAVX(
            const_cast<float*>(query), const_cast<float*>(x), norm_half, dim);
        float l2_dist_sq = query_norm_sq + 2.0f * dist_score;
        l2_dist_sq = std::max(0.0f, l2_dist_sq);
        ++dist_computed;
        if (!collector.is_prunable(l2_dist_sq)) {
            collector.try_push(l2_dist_sq, static_cast<int>(vec_id));
        }
    }
}

constexpr int kMainBatchL2MinRows = 4;

inline void push_l2_results_to_collector(MaxHeapAnnCollector& collector,
                                         int& dist_computed,
                                         int max_num_distances,
                                         const uint32_t* ids,
                                         const float* dist_sq,
                                         int n) {
    for (int i = 0; i < n; ++i) {
        if (dist_computed >= max_num_distances) {
            return;
        }
        ++dist_computed;
        const float dsq = dist_sq[static_cast<size_t>(i)];
        if (collector.is_prunable(dsq)) {
            continue;
        }
        collector.try_push(dsq, static_cast<int>(ids[static_cast<size_t>(i)]));
    }
}

inline void scan_packed_rows_ipsimd(const float* query,
                                    int dim,
                                    float query_norm_sq,
                                    const float* rows,
                                    const float* norms_sq,
                                    const uint32_t* ids,
                                    int n_rows,
                                    MaxHeapAnnCollector& collector,
                                    int& dist_computed,
                                    int max_num_distances) {
    if (n_rows <= 0 || rows == nullptr || norms_sq == nullptr || ids == nullptr) {
        return;
    }
    n_rows = std::min(n_rows, max_num_distances - dist_computed);
    if (n_rows <= 0) return;
    thread_local std::vector<float> dist_buf_tls;
    dist_buf_tls.resize(static_cast<size_t>(n_rows));
    batch_l2_sq_q_rows(query, dim, rows, norms_sq, n_rows, query_norm_sq, dist_buf_tls.data());
    push_l2_results_to_collector(collector, dist_computed, max_num_distances, ids,
                                 dist_buf_tls.data(), n_rows);
}

std::vector<int> dedupe_cluster_ids_preserve_order(const std::vector<int>& cluster_ids) {
    std::vector<int> unique_ids;
    unique_ids.reserve(cluster_ids.size());
    std::unordered_set<int> seen;
    seen.reserve(cluster_ids.size() * 2);
    for (int cid : cluster_ids) {
        if (seen.insert(cid).second) {
            unique_ids.push_back(cid);
        }
    }
    return unique_ids;
}

}  // namespace

namespace {

inline float kmeans_l2_distance_sq_simd(int dimension, const float* vec1, const float* vec2) {
    float ip = IPSIMD4ExtAVX(const_cast<float*>(vec1), const_cast<float*>(vec2), dimension);
    float norm1_sq = 0.0f;
    float norm2_sq = 0.0f;
    for (int d = 0; d < dimension; ++d) {
        norm1_sq += vec1[d] * vec1[d];
        norm2_sq += vec2[d] * vec2[d];
    }
    return norm1_sq + norm2_sq - 2.0f * ip;
}

inline bool kmeans_use_attr(const float* scalars, float attr_lambda) {
    return scalars != nullptr && attr_lambda > 0.0f;
}

float kmeans_attr_term(float scalar,
                       float scalar_centroid,
                       float attr_lambda,
                       float scalar_scale) {
    if (attr_lambda <= 0.0f) {
        return 0.0f;
    }
    const float scale = std::max(1e-6f, scalar_scale);
    const float diff = (scalar - scalar_centroid) / scale;
    return attr_lambda * diff * diff;
}

float kmeans_span_expansion_penalty(float scalar,
                                    float cluster_min,
                                    float cluster_max,
                                    bool bounds_valid,
                                    float scalar_span_lambda,
                                    float scalar_scale) {
    if (!bounds_valid || scalar_span_lambda <= 0.0f) {
        return 0.0f;
    }
    const float new_min = std::min(cluster_min, scalar);
    const float new_max = std::max(cluster_max, scalar);
    const float predicted_span = std::max(0.0f, new_max - new_min);
    const float scale = std::max(1e-6f, scalar_scale);
    const float norm = predicted_span / scale;
    return scalar_span_lambda * norm * norm;
}

void kmeans_compute_scalar_norm(const float* scalars,
                                size_t n_vectors,
                                float& scalar_mean,
                                float& scalar_scale) {
    scalar_mean = 0.0f;
    scalar_scale = 1.0f;
    if (scalars == nullptr || n_vectors == 0) {
        return;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n_vectors; ++i) {
        sum += static_cast<double>(scalars[i]);
    }
    scalar_mean = static_cast<float>(sum / static_cast<double>(n_vectors));
    double var = 0.0;
    for (size_t i = 0; i < n_vectors; ++i) {
        const double diff = static_cast<double>(scalars[i]) - static_cast<double>(scalar_mean);
        var += diff * diff;
    }
    scalar_scale = static_cast<float>(std::sqrt(var / static_cast<double>(n_vectors)));
    if (!std::isfinite(scalar_scale) || scalar_scale < 1e-6f) {
        scalar_scale = 1.0f;
    }
}

float kmeans_combined_distance_sq(int dimension,
                                  const float* vec,
                                  const float* centroid,
                                  float scalar,
                                  float scalar_centroid,
                                  float attr_lambda,
                                  float scalar_scale) {
    return kmeans_l2_distance_sq_simd(dimension, vec, centroid) +
           kmeans_attr_term(scalar, scalar_centroid, attr_lambda, scalar_scale);
}

void kmeans_initialize_centroids(std::vector<std::vector<float>>& centroids,
                                 std::vector<float>& scalar_centroids,
                                 const float* data,
                                 const float* scalars,
                                 size_t n_vectors,
                                 int dimension,
                                 int n_clusters,
                                 bool use_kmeanspp,
                                 uint32_t seed,
                                 float attr_lambda,
                                 float scalar_scale,
                                 const float* sample_weights = nullptr) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<size_t> first_dist(0, n_vectors - 1);
    const int init_clusters = std::min(n_clusters, static_cast<int>(n_vectors));
    scalar_centroids.assign(static_cast<size_t>(n_clusters), 0.0f);

    if (!use_kmeanspp || init_clusters <= 1) {
        std::vector<size_t> indices(n_vectors);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);
        for (int i = 0; i < init_clusters; ++i) {
            const size_t src = indices[static_cast<size_t>(i)];
            centroids[static_cast<size_t>(i)].resize(static_cast<size_t>(dimension));
            std::memcpy(centroids[static_cast<size_t>(i)].data(), data + src * static_cast<size_t>(dimension),
                        static_cast<size_t>(dimension) * sizeof(float));
            scalar_centroids[static_cast<size_t>(i)] = scalars ? scalars[src] : 0.0f;
        }
        return;
    }

    std::vector<float> min_dists(n_vectors, std::numeric_limits<float>::max());
    size_t first = first_dist(g);
    if (sample_weights) {
        double total_weight = 0.0;
        for (size_t i = 0; i < n_vectors; ++i) {
            total_weight += static_cast<double>(std::max(0.0f, sample_weights[i]));
        }
        if (total_weight > 0.0) {
            std::uniform_real_distribution<double> pick_first(0.0, total_weight);
            const double threshold = pick_first(g);
            double cumulative = 0.0;
            for (size_t i = 0; i < n_vectors; ++i) {
                cumulative += static_cast<double>(std::max(0.0f, sample_weights[i]));
                if (cumulative >= threshold) { first = i; break; }
            }
        }
    }
    centroids[0].resize(static_cast<size_t>(dimension));
    std::memcpy(centroids[0].data(), data + first * static_cast<size_t>(dimension),
                static_cast<size_t>(dimension) * sizeof(float));
    scalar_centroids[0] = scalars ? scalars[first] : 0.0f;

    for (int c = 1; c < init_clusters; ++c) {
        double total = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : total)
#endif
        for (size_t i = 0; i < n_vectors; ++i) {
            const float* vec = data + i * static_cast<size_t>(dimension);
            const float dist = kmeans_combined_distance_sq(
                dimension, vec, centroids[static_cast<size_t>(c - 1)].data(),
                scalars ? scalars[i] : 0.0f, scalar_centroids[static_cast<size_t>(c - 1)],
                attr_lambda, scalar_scale);
            if (dist < min_dists[i]) {
                min_dists[i] = dist;
            }
            const double weight = sample_weights
                ? static_cast<double>(std::max(0.0f, sample_weights[i])) : 1.0;
            total += weight * static_cast<double>(std::max(0.0f, min_dists[i]));
        }
        size_t chosen = static_cast<size_t>(c) % n_vectors;
        if (total > 0.0 && std::isfinite(total)) {
            std::uniform_real_distribution<double> pick_dist(0.0, total);
            double threshold = pick_dist(g);
            double cumulative = 0.0;
            for (size_t i = 0; i < n_vectors; ++i) {
                const double weight = sample_weights
                    ? static_cast<double>(std::max(0.0f, sample_weights[i])) : 1.0;
                cumulative += weight * static_cast<double>(std::max(0.0f, min_dists[i]));
                if (cumulative >= threshold) {
                    chosen = i;
                    break;
                }
            }
        }
        centroids[static_cast<size_t>(c)].resize(static_cast<size_t>(dimension));
        std::memcpy(centroids[static_cast<size_t>(c)].data(), data + chosen * static_cast<size_t>(dimension),
                    static_cast<size_t>(dimension) * sizeof(float));
        scalar_centroids[static_cast<size_t>(c)] = scalars ? scalars[chosen] : 0.0f;
    }
}

// K-means 主循环：并行 assignment + 串行累加 + 并行质心更新（无 omp critical）
int simple_kmeans_run_iterations(
    std::vector<std::vector<float>>& centroids,
    std::vector<float>& scalar_centroids,
    const float* data,
    const float* scalars,
    size_t n_vectors,
    int dimension,
    int n_clusters,
    int max_iterations,
    float attr_lambda,
    float scalar_span_lambda,
    float scalar_scale,
    const float* sample_weights = nullptr,
    std::vector<float>* learned_scalar_mins = nullptr,
    std::vector<float>* learned_scalar_maxs = nullptr) {
    std::vector<int> assignments(n_vectors, 0);
    std::vector<std::vector<float>> new_centroids(static_cast<size_t>(n_clusters));
    std::vector<double> new_scalar_sums(static_cast<size_t>(n_clusters), 0.0);
    std::vector<float> scalar_mins(static_cast<size_t>(n_clusters), 0.0f);
    std::vector<float> scalar_maxs(static_cast<size_t>(n_clusters), 0.0f);
    std::vector<uint8_t> scalar_bounds_valid(static_cast<size_t>(n_clusters), 0u);
    std::vector<float> new_scalar_mins(static_cast<size_t>(n_clusters), 0.0f);
    std::vector<float> new_scalar_maxs(static_cast<size_t>(n_clusters), 0.0f);
    std::vector<uint8_t> new_scalar_bounds_valid(static_cast<size_t>(n_clusters), 0u);
    std::vector<double> counts(static_cast<size_t>(n_clusters), 0.0);
    for (int j = 0; j < n_clusters; ++j) {
        new_centroids[static_cast<size_t>(j)].assign(static_cast<size_t>(dimension), 0.0f);
    }
    if (scalar_centroids.size() < static_cast<size_t>(n_clusters)) {
        scalar_centroids.assign(static_cast<size_t>(n_clusters), 0.0f);
    }
    const bool use_attr = kmeans_use_attr(scalars, attr_lambda);
    const bool use_span = scalars != nullptr && scalar_span_lambda > 0.0f;
    const bool use_scalar = scalars != nullptr && (use_attr || use_span);

    int actual_iterations = max_iterations;
    for (int iter = 0; iter < max_iterations; ++iter) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < n_vectors; ++i) {
            const float* vec = data + i * static_cast<size_t>(dimension);
            float min_dist_sq = std::numeric_limits<float>::max();
            int nearest = 0;
            for (int j = 0; j < n_clusters; ++j) {
                float dist_sq = kmeans_combined_distance_sq(
                    dimension, vec, centroids[static_cast<size_t>(j)].data(),
                    use_attr ? scalars[i] : 0.0f,
                    scalar_centroids[static_cast<size_t>(j)],
                    use_attr ? attr_lambda : 0.0f,
                    scalar_scale);
                if (use_span) {
                    const size_t ju = static_cast<size_t>(j);
                    dist_sq += kmeans_span_expansion_penalty(
                        scalars[i], scalar_mins[ju], scalar_maxs[ju],
                        scalar_bounds_valid[ju] != 0u, scalar_span_lambda, scalar_scale);
                }
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    nearest = j;
                }
            }
            assignments[i] = nearest;
        }

        for (int j = 0; j < n_clusters; ++j) {
            std::fill(
                new_centroids[static_cast<size_t>(j)].begin(),
                new_centroids[static_cast<size_t>(j)].end(),
                0.0f);
            counts[static_cast<size_t>(j)] = 0.0;
            new_scalar_sums[static_cast<size_t>(j)] = 0.0;
            new_scalar_bounds_valid[static_cast<size_t>(j)] = 0u;
            new_scalar_mins[static_cast<size_t>(j)] = 0.0f;
            new_scalar_maxs[static_cast<size_t>(j)] = 0.0f;
        }

        for (size_t i = 0; i < n_vectors; ++i) {
            const int nearest = assignments[i];
            const float* vec = data + i * static_cast<size_t>(dimension);
            const double weight = sample_weights
                ? static_cast<double>(std::max(0.0f, sample_weights[i]))
                : 1.0;
            if (weight <= 0.0) continue;
            for (int d = 0; d < dimension; ++d) {
                new_centroids[static_cast<size_t>(nearest)][static_cast<size_t>(d)] +=
                    static_cast<float>(weight * static_cast<double>(vec[d]));
            }
            if (use_scalar) {
                const size_t nu = static_cast<size_t>(nearest);
                new_scalar_sums[nu] += weight * static_cast<double>(scalars[i]);
                if (!new_scalar_bounds_valid[nu]) {
                    new_scalar_mins[nu] = scalars[i];
                    new_scalar_maxs[nu] = scalars[i];
                    new_scalar_bounds_valid[nu] = 1u;
                } else {
                    new_scalar_mins[nu] = std::min(new_scalar_mins[nu], scalars[i]);
                    new_scalar_maxs[nu] = std::max(new_scalar_maxs[nu], scalars[i]);
                }
            }
            counts[static_cast<size_t>(nearest)] += weight;
        }

        bool converged = true;
        float max_centroid_change = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max : max_centroid_change) reduction(&& : converged)
#endif
        for (int j = 0; j < n_clusters; ++j) {
            if (counts[static_cast<size_t>(j)] <= 0.0) {
                continue;
            }
            const size_t ju = static_cast<size_t>(j);
            const double cnt = counts[ju];
            std::vector<float> old_centroid = centroids[ju];
            centroids[ju].resize(static_cast<size_t>(dimension));
            for (int d = 0; d < dimension; ++d) {
                centroids[ju][static_cast<size_t>(d)] =
                    static_cast<float>(static_cast<double>(new_centroids[ju][static_cast<size_t>(d)]) / cnt);
            }
            if (use_scalar) {
                scalar_centroids[ju] = static_cast<float>(new_scalar_sums[ju] / cnt);
                if (new_scalar_bounds_valid[ju]) {
                    scalar_mins[ju] = new_scalar_mins[ju];
                    scalar_maxs[ju] = new_scalar_maxs[ju];
                    scalar_bounds_valid[ju] = 1u;
                }
            }

            float centroid_change = 0.0f;
            for (int d = 0; d < dimension; ++d) {
                const float d_diff = centroids[ju][static_cast<size_t>(d)] - old_centroid[static_cast<size_t>(d)];
                centroid_change += d_diff * d_diff;
            }
            centroid_change = std::sqrt(centroid_change);
            if (centroid_change > max_centroid_change) {
                max_centroid_change = centroid_change;
            }
            const float diff = centroid_change * centroid_change / static_cast<float>(dimension);
            if (diff > 1e-8f) {
                converged = false;
            }
        }

        if (converged) {
            actual_iterations = iter + 1;
            std::cerr << "[K-means早停] 迭代 " << actual_iterations << "/" << max_iterations
                      << ", 最大质心变化: " << max_centroid_change
                      << (max_centroid_change < 5e-6f ? " < 5e-06" : "") << std::endl;
            break;
        }
    }
    if (learned_scalar_mins != nullptr) {
        *learned_scalar_mins = scalar_mins;
    }
    if (learned_scalar_maxs != nullptr) {
        *learned_scalar_maxs = scalar_maxs;
    }
    return actual_iterations;
}

}  // namespace

// ============================================================================
// SimpleKMeans实现
// ============================================================================

void SimpleKMeans::train(const float* data, size_t n_vectors, int dimension, int n_clusters,
                         const float* scalars) {
    dimension_ = dimension;
    centroids_.clear();
    centroids_.resize(n_clusters);
    scalar_centroids_.assign(static_cast<size_t>(n_clusters), 0.0f);
    
    if (n_vectors == 0 || n_clusters == 0) return;
    
    kmeans_compute_scalar_norm(scalars, n_vectors, scalar_mean_, scalar_scale_);
    kmeans_initialize_centroids(
        centroids_, scalar_centroids_, data, scalars, n_vectors, dimension, n_clusters,
        use_kmeanspp_, random_seed_, attr_lambda_, scalar_scale_);
    
    // K-means迭代（根据数据量和聚类数动态调整迭代次数）
    // 高标准：参考FAISS，大数据量或大聚类数需要更多迭代才能充分收敛
    int max_iterations = 20;  // 默认：小数据量 20 次迭代
    if (n_vectors >= 500000 || n_clusters >= 500) {
        max_iterations = 40;   // 大数据量或大聚类数：40 次迭代（最大）
    } else if (n_vectors >= 200000 || n_clusters >= 300) {
        max_iterations = 35;   // 中等数据量：35 次迭代
    } else if (n_vectors >= 100000 || n_clusters >= 200) {
        max_iterations = 30;   // 较大数据量：30 次迭代
    } else if (n_vectors < 10000 && n_clusters < 100) {
        // coarse 层（fine 质心数量少）：20 次迭代
        max_iterations = 20;
    } else {
        max_iterations = 20;   // 小数据量：20 次迭代
    }
    
    // 调试日志：输出训练参数
    std::cerr << "[K-means训练] 开始训练: n_vectors=" << n_vectors 
              << ", n_clusters=" << n_clusters 
              << ", max_iterations=" << max_iterations
              << ", init=" << (use_kmeanspp_ ? "kmeans++" : "shuffle")
              << ", seed=" << random_seed_
              << ", attr_lambda=" << attr_lambda_
              << ", scalar_span_lambda=" << scalar_span_lambda_ << std::endl;
#ifdef _OPENMP
    log_kmeans_train_omp_threads();
#else
    std::cerr << "[K-means训练] OpenMP 未编译（setup.py 需 -fopenmp），使用单线程" << std::endl;
#endif

    const int actual_iterations = simple_kmeans_run_iterations(
        centroids_, scalar_centroids_, data, scalars, n_vectors, dimension,
        n_clusters, max_iterations, attr_lambda_, scalar_span_lambda_, scalar_scale_, nullptr,
        &scalar_mins_, &scalar_maxs_);

    if (actual_iterations >= max_iterations) {
        std::cerr << "[K-means训练] 训练完成: 完成全部 " << max_iterations << " 次迭代（未早停）" << std::endl;
    } else {
        std::cerr << "[K-means训练] 训练完成: 实际迭代次数=" << actual_iterations << std::endl;
    }
}

// 训练聚类中心（带最大迭代次数参数，用于优化coarse层重建）
void SimpleKMeans::train_with_max_iter(const float* data, size_t n_vectors, int dimension,
                                       int n_clusters, int max_iterations, const float* scalars) {
    dimension_ = dimension;
    centroids_.clear();
    centroids_.resize(n_clusters);
    scalar_centroids_.assign(static_cast<size_t>(n_clusters), 0.0f);
    
    if (n_vectors == 0 || n_clusters == 0) return;
    
    kmeans_compute_scalar_norm(scalars, n_vectors, scalar_mean_, scalar_scale_);
    kmeans_initialize_centroids(
        centroids_, scalar_centroids_, data, scalars, n_vectors, dimension, n_clusters,
        use_kmeanspp_, random_seed_, attr_lambda_, scalar_scale_);
    
#ifdef _OPENMP
    log_kmeans_train_omp_threads();
#endif

    (void)simple_kmeans_run_iterations(
        centroids_, scalar_centroids_, data, scalars, n_vectors, dimension,
        n_clusters, max_iterations, attr_lambda_, scalar_span_lambda_, scalar_scale_, nullptr,
        &scalar_mins_, &scalar_maxs_);
}

void SimpleKMeans::train_weighted_with_max_iter(
    const float* data, const float* weights, size_t n_vectors, int dimension,
    int n_clusters, int max_iterations) {
    dimension_ = dimension;
    centroids_.assign(static_cast<size_t>(n_clusters), {});
    scalar_centroids_.assign(static_cast<size_t>(n_clusters), 0.0f);
    if (n_vectors == 0 || n_clusters == 0) return;
    scalar_mean_ = 0.0f;
    scalar_scale_ = 1.0f;
    kmeans_initialize_centroids(
        centroids_, scalar_centroids_, data, nullptr, n_vectors, dimension, n_clusters,
        use_kmeanspp_, random_seed_, 0.0f, 1.0f, weights);
    (void)simple_kmeans_run_iterations(
        centroids_, scalar_centroids_, data, nullptr, n_vectors, dimension,
        n_clusters, max_iterations, 0.0f, 0.0f, 1.0f, weights);
}

uint32_t SimpleKMeans::assign(const float* vector, float scalar) const {
    if (centroids_.empty() || dimension_ <= 0) {
        return 0;
    }
    
    // 使用SIMD优化距离计算
    float min_dist = std::numeric_limits<float>::max();
    uint32_t nearest = 0;
    
    for (size_t j = 0; j < centroids_.size(); ++j) {
        if (centroids_[j].empty() || centroids_[j].size() != static_cast<size_t>(dimension_)) {
            continue;
        }
        
        // 使用SIMD优化的内积计算
        float ip = IPSIMD4ExtAVX(const_cast<float*>(vector), 
                                 const_cast<float*>(centroids_[j].data()), 
                                 dimension_);
        
        // 优化：使用SIMD计算范数平方
        float norm_v_sq = IPSIMD4ExtAVX(const_cast<float*>(vector), 
                                        const_cast<float*>(vector), 
                                        dimension_);
        float norm_c_sq = IPSIMD4ExtAVX(const_cast<float*>(centroids_[j].data()), 
                                        const_cast<float*>(centroids_[j].data()), 
                                        dimension_);
        
        // 计算L2距离平方：||v - c||^2 = ||v||^2 + ||c||^2 - 2*IP
        float dist_sq = norm_v_sq + norm_c_sq - 2.0f * ip;
        if (attr_lambda_ > 0.0f && j < scalar_centroids_.size()) {
            dist_sq += kmeans_attr_term(scalar, scalar_centroids_[j], attr_lambda_, scalar_scale_);
        }
        if (scalar_span_lambda_ > 0.0f && j < scalar_mins_.size() && j < scalar_maxs_.size()) {
            dist_sq += kmeans_span_expansion_penalty(
                scalar, scalar_mins_[j], scalar_maxs_[j], true,
                scalar_span_lambda_, scalar_scale_);
        }
        
        if (dist_sq < min_dist) {
            min_dist = dist_sq;
            nearest = static_cast<uint32_t>(j);
        }
    }
    
    return nearest;
}

// ============================================================================
// AdaIVFCore实现
// ============================================================================

AdaIVFCore::AdaIVFCore(int n_clusters, int n_probe, size_t max_cluster_size,
                      float recluster_threshold)
    : n_clusters_(n_clusters), n_probe_(n_probe), max_cluster_size_(max_cluster_size),
      recluster_threshold_(recluster_threshold),
      dimension_(0), n_vectors_(0), is_trained_(false) {
    ivf_lists_.resize(n_clusters_);
    cluster_stats_.resize(n_clusters_);
    list_locks_.resize(n_clusters_);
    
    for (int i = 0; i < n_clusters_; ++i) {
        cluster_stats_[i] = std::unique_ptr<ClusterStats>(new ClusterStats(i));
        list_locks_[i] = std::unique_ptr<AdaIvfListMutex>(new AdaIvfListMutex());
    }
    
    kmeans_ = std::unique_ptr<SimpleKMeans>(new SimpleKMeans());
    kmeans_->set_attr_lambda(attr_lambda_);
    kmeans_->set_scalar_span_lambda(scalar_span_lambda_);
    kmeans_->set_random_seed(kmeans_seed_);
    kmeans_->set_use_kmeanspp(use_kmeanspp_);
}

AdaIVFCore::~AdaIVFCore() = default;

// =============================================================================
// Module A：无序段达阈值则归并到有序段（持簇锁时调用）
// =============================================================================
void AdaIVFCore::lsm_trigger_merge_if_needed(RangeAwareList& list) {
    if (!enable_scalar_filter_lsm_) {
        return;
    }
    // 固定 256 大于 1M/4096-list 的平均桶大小（约 244），会让多数桶
    // 永久停留在需要全量线性过滤的 delta。以全局平均桶大小约束阈值。
    const size_t avg_list_size = std::max<size_t>(
        1, (n_vectors_ + static_cast<size_t>(n_clusters_) - 1) /
               static_cast<size_t>(n_clusters_));
    const size_t average_aware_threshold = std::max<size_t>(32, avg_list_size / 4);
    const size_t main_aware_threshold = list.main_size() == 0
        ? average_aware_threshold
        : std::max<size_t>(32, std::min<size_t>(4096, list.main_size() / 20));
    // 用户配置是阈值上限；显式设置更小值仍然生效。
    const size_t threshold = std::max<size_t>(
        1, std::min(lsm_merge_threshold_,
                    std::max(average_aware_threshold, main_aware_threshold)));
    if (list.delta_size() >= threshold) {
        list.merge_unordered_into_main(keep_list_packed_payload_);
    }
}

void AdaIVFCore::flush_all_lsm_segments() {
    for (int i = 0; i < n_clusters_; ++i) {
        std::lock_guard<AdaIvfListMutex> lock(*list_locks_[i]);
        ivf_lists_[i].merge_unordered_into_main(keep_list_packed_payload_);
    }
}

int AdaIVFCore::find_cluster_for_vec_id(uint32_t vec_id) const {
    if (!is_trained_ || vec_id >= n_vectors_) {
        return -1;
    }
    for (int c = 0; c < n_clusters_; ++c) {
        std::lock_guard<AdaIvfListMutex> lock(*list_locks_[c]);
        const RangeAwareList& list = ivf_lists_[c];
        for (uint32_t id : list.main_ids) {
            if (id == vec_id) {
                return c;
            }
        }
        for (uint32_t id : list.delta_ids) {
            if (id == vec_id) {
                return c;
            }
        }
    }
    return -1;
}

bool AdaIVFCore::vector_uses_pq(uint32_t vec_id) const {
    return enable_pq_compression_ && pq_.is_trained() && vec_id < pq_codes_.size() &&
           pq_codes_[vec_id].size() == static_cast<size_t>(pq_.m());
}

void AdaIVFCore::decode_vector_for_id(uint32_t vec_id, std::vector<float>& buf) const {
    if (vec_id >= n_vectors_) {
        throw std::runtime_error("decode_vector_for_id: invalid vec_id");
    }
    buf.resize(static_cast<size_t>(dimension_));
    if (vector_uses_pq(vec_id)) {
        pq_.decode(pq_codes_[vec_id].data(), buf.data());
    } else if (!vectors_flat_.empty() &&
               vectors_flat_.size() >= (static_cast<size_t>(vec_id) + 1) * static_cast<size_t>(dimension_)) {
        const float* src = vectors_flat_.data() + static_cast<size_t>(vec_id) * static_cast<size_t>(dimension_);
        std::memcpy(buf.data(), src, static_cast<size_t>(dimension_) * sizeof(float));
    } else {
        throw std::runtime_error("decode_vector_for_id: no float or PQ data");
    }
}

void AdaIVFCore::train_pq_from_samples(const float* data, size_t n_vectors, int dimension) {
    if (!is_trained_) {
        throw std::runtime_error("train_pq_from_samples: index not trained");
    }
    if (dimension != dimension_) {
        throw std::runtime_error("train_pq_from_samples: dimension mismatch");
    }
    enable_pq_compression_ = true;
    pq_.set_params(pq_m_, pq_ksub_);
    pq_.train(data, n_vectors, dimension_);
}

void AdaIVFCore::gather_cluster_candidate_ids(int cluster_id, bool use_range_filter,
                                              float range_min, float range_max,
                                              std::vector<uint32_t>& out) const {
    if (cluster_id < 0 || cluster_id >= n_clusters_) {
        return;
    }
    if (cluster_stats_[cluster_id]->is_deleted) {
        return;
    }
    const RangeAwareList& list = ivf_lists_[cluster_id];
    if (use_range_filter) {
        gather_main_range_ids(list, range_min, range_max, out);
        gather_delta_range_ids_simd(list, range_min, range_max, out);
    } else {
        gather_cluster_all_ids(list, out);
    }
}

std::vector<std::pair<float, int>> AdaIVFCore::scan_cluster_list_for_topk(
    const float* query, int k, int max_num_distances,
    const std::vector<int>& cluster_ids,
    float range_min, float range_max,
    SearchStats* stats) const {
    if (!is_trained_ || n_vectors_ == 0 || cluster_ids.empty() || max_num_distances <= 0) {
        return {};
    }
    if (vectors_flat_.empty()) {
        return {};
    }

    const std::vector<int> clusters = dedupe_cluster_ids_preserve_order(cluster_ids);
    if (clusters.empty()) {
        return {};
    }

    float query_norm_sq = 0.0f;
    for (int d = 0; d < dimension_; ++d) {
        query_norm_sq += query[d] * query[d];
    }

    const bool use_range_filter = (std::isfinite(range_min) && std::isfinite(range_max) &&
                                   range_min > -1e9f && range_max < 1e9f);

    // Adaptive GT-like per-cluster scan budget. Candidate generation already
    // covers the true-GT fine clusters well; the bottleneck is spending distance
    // budget earlier on clusters that are both geometrically near and have
    // reliable range hit mass. We therefore pre-allocate a query-local budget per
    // candidate and use prefix caps during scanning, so unused budget from early
    // clusters naturally carries forward.
    std::vector<int> cluster_budgets(clusters.size(), max_num_distances);
    bool use_targetness_budget = use_range_filter && clusters.size() > 1;
    if (use_targetness_budget) {
        const int sample_n = std::min<int>(64, static_cast<int>(clusters.size()));
        const int stride = std::max<int>(1, static_cast<int>(clusters.size()) / std::max(1, sample_n));
        int sampled = 0;
        int overlap_count = 0;
        int positive_count = 0;
        for (size_t pos = 0; pos < clusters.size() && sampled < sample_n; pos += static_cast<size_t>(stride)) {
            const int cid = clusters[pos];
            if (cid < 0 || cid >= n_clusters_ || cluster_stats_[cid]->is_deleted) continue;
            const bool overlap = cluster_may_match_range(cid, range_min, range_max);
            const size_t hit = overlap ? estimate_cluster_hit_count_from_synopsis(cid, range_min, range_max) : 0;
            overlap_count += overlap ? 1 : 0;
            positive_count += hit > 0 ? 1 : 0;
            ++sampled;
        }
        if (sampled > 0) {
            const float overlap_ratio = static_cast<float>(overlap_count) / static_cast<float>(sampled);
            const float positive_ratio = static_cast<float>(positive_count) / static_cast<float>(sampled);
            if (overlap_ratio > 0.85f || positive_ratio > 0.65f) {
                use_targetness_budget = false;
            }
        }
    }
    if (use_targetness_budget) {
        const float avg_budget = static_cast<float>(max_num_distances) /
            static_cast<float>(std::max<size_t>(1, clusters.size()));
        constexpr float kGeoTau = 48.0f;
        constexpr float kReliabilityC = 32.0f;
        constexpr float kHitFloorScale = 3.0f;
        constexpr float kHitFloorBias = 4.0f;
        constexpr float kHitCapScale = 2.0f;
        constexpr float kHitCapBias = 16.0f;
        constexpr float kAvgCapScale = 3.0f;

        std::vector<float> base_budget(clusters.size(), 0.0f);
        std::vector<float> cap_budget(clusters.size(), 0.0f);
        std::vector<float> targetness(clusters.size(), 0.0f);
        int base_sum = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int cid = clusters[i];
            if (cid < 0 || cid >= n_clusters_ || cluster_stats_[cid]->is_deleted) {
                cluster_budgets[i] = 0;
                continue;
            }
            const float cnt = static_cast<float>(std::max<size_t>(1, cluster_stats_[cid]->size));
            const float hit = static_cast<float>(estimate_cluster_hit_count_from_synopsis(
                cid, range_min, range_max));
            const float rank = static_cast<float>(i);
            const float geo = std::exp(-rank / kGeoTau);
            const float purity = std::min(1.0f, hit / cnt);
            const float mass = std::log1p(hit);
            const float reliability = cnt / (cnt + kReliabilityC);
            const float gain = geo * mass * std::sqrt(std::max(0.0f, purity)) * reliability;

            const float geo_floor = avg_budget * geo;
            const float hit_floor = kHitFloorScale * std::sqrt(std::max(0.0f, hit)) + kHitFloorBias;
            const float raw_base = std::max(8.0f, std::max(geo_floor, hit_floor));
            const float raw_cap = std::max(raw_base, std::max(kHitCapScale * hit + kHitCapBias, avg_budget * kAvgCapScale));
            base_budget[i] = std::min(cnt, raw_base);
            cap_budget[i] = std::min(cnt, raw_cap);
            targetness[i] = gain;
            const int bi = std::max(0, static_cast<int>(std::ceil(base_budget[i])));
            cluster_budgets[i] = bi;
            base_sum += bi;
        }

        if (base_sum > max_num_distances) {
            float scale = static_cast<float>(max_num_distances) / static_cast<float>(std::max(1, base_sum));
            int scaled_sum = 0;
            for (size_t i = 0; i < clusters.size(); ++i) {
                int b = static_cast<int>(std::floor(static_cast<float>(cluster_budgets[i]) * scale));
                if (cluster_budgets[i] > 0) b = std::max(1, b);
                cluster_budgets[i] = b;
                scaled_sum += b;
            }
            for (size_t i = 0; scaled_sum > max_num_distances && i < clusters.size(); ++i) {
                if (cluster_budgets[i] > 1) {
                    cluster_budgets[i] -= 1;
                    --scaled_sum;
                }
            }
        } else {
            const int extra_pool = max_num_distances - base_sum;
            float target_sum = 0.0f;
            for (float t : targetness) target_sum += t;
            if (extra_pool > 0 && target_sum > 0.0f) {
                std::vector<std::pair<float, size_t>> fractional;
                fractional.reserve(clusters.size());
                int extra_used = 0;
                for (size_t i = 0; i < clusters.size(); ++i) {
                    const int cap_i = std::max(0, static_cast<int>(std::ceil(cap_budget[i])));
                    const int room = std::max(0, cap_i - cluster_budgets[i]);
                    if (room <= 0 || targetness[i] <= 0.0f) continue;
                    const float exact = static_cast<float>(extra_pool) * targetness[i] / target_sum;
                    int add = std::min(room, static_cast<int>(std::floor(exact)));
                    cluster_budgets[i] += add;
                    extra_used += add;
                    fractional.emplace_back(exact - static_cast<float>(add), i);
                }
                int remaining_extra = extra_pool - extra_used;
                std::sort(fractional.begin(), fractional.end(),
                          [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                              return a.first > b.first;
                          });
                while (remaining_extra > 0 && !fractional.empty()) {
                    bool progressed = false;
                    for (const auto& item : fractional) {
                        if (remaining_extra <= 0) break;
                        const size_t i = item.second;
                        const int cap_i = std::max(0, static_cast<int>(std::ceil(cap_budget[i])));
                        if (cluster_budgets[i] < cap_i) {
                            cluster_budgets[i] += 1;
                            --remaining_extra;
                            progressed = true;
                        }
                    }
                    if (!progressed) break;
                }
            }
        }
    }

    // 按 cluster_ids 保序逐簇：main 段 co-locate batch IPSIMD；delta 段单趟过滤并直接算距离。
    // max_num_distances 是计算预算，不是结果堆容量。
    MaxHeapAnnCollector collector(k);

    const size_t dim_sz = static_cast<size_t>(dimension_);
    int dist_computed = 0;
    const bool can_use_flat = !vectors_flat_.empty();
    int budget_prefix_cap = 0;

    for (size_t ci = 0; ci < clusters.size(); ++ci) {
        const int cluster_id = clusters[ci];
        if (dist_computed >= max_num_distances) {
            break;
        }
        if (cluster_id < 0 || cluster_id >= n_clusters_) {
            continue;
        }
        if (cluster_stats_[cluster_id]->is_deleted) {
            continue;
        }

        // Prefix cap: a cluster receives its precomputed budget, while any
        // distance budget unused by earlier clusters remains available to later
        // ones because the cap is cumulative rather than dist_computed + share.
        int cluster_cap = max_num_distances;
        if (use_targetness_budget) {
            budget_prefix_cap += (ci < cluster_budgets.size()) ? cluster_budgets[ci] : 0;
            cluster_cap = std::min(max_num_distances, std::max(dist_computed, budget_prefix_cap));
        }

        {
            AdaIvfSharedLock<AdaIvfListMutex> list_lock(*list_locks_[cluster_id]);
            const RangeAwareList& list = ivf_lists_[cluster_id];
            const size_t list_total = list.total_size();
            if (list_total == 0) {
                continue;
            }
            if (use_range_filter && enable_scalar_range_prune_ &&
                !list.scalar_may_overlap(range_min, range_max)) {
                if (stats) {
                    stats->num_pruned_clusters += 1;
                }
                continue;
            }
            // histogram 只能作为软估算信号，不能在簇扫描阶段硬剪枝；
            // 硬剪枝只允许使用 min/max overlap，避免估算误差造成 recall 下降。
            (void)histogram_prune_threshold_;
            if (stats) {
                stats->num_visited_clusters += 1;
                stats->list_vectors_considered += list_total;
                stats->num_scanned_vectors += list_total;
            }

            if (use_range_filter) {
                if (list.main_is_packed()) {
                    const RangeAwareList::MainRangeView view =
                        list.main_range_view(range_min, range_max);
                    if (stats) stats->num_filtered_candidates += view.count;
                    if (view.count >= static_cast<size_t>(kMainBatchL2MinRows) &&
                        view.vecs != nullptr && view.norms_sq != nullptr && view.ids != nullptr && tombstone_ids_.empty()) {
                        scan_packed_rows_ipsimd(
                            query, dimension_, query_norm_sq, view.vecs, view.norms_sq,
                            view.ids, static_cast<int>(view.count), collector, dist_computed,
                            cluster_cap);
                    } else if (view.count > 0 && view.ids != nullptr) {
                        for (size_t ii = 0; ii < view.count; ++ii) {
                            if (dist_computed >= cluster_cap) {
                                break;
                            }
                            const uint32_t vec_id = view.ids[ii];
                            if (!tombstone_ids_.empty() && tombstone_ids_.count(vec_id) != 0) continue;
                            const float* x = view.vecs != nullptr
                                                 ? (view.vecs + ii * dim_sz)
                                                 : nullptr;
                            if (x == nullptr) {
                                if (!can_use_flat || vec_id >= static_cast<uint32_t>(n_vectors_) ||
                                    vectors_flat_.size() < (static_cast<size_t>(vec_id) + 1) * dim_sz) {
                                    continue;
                                }
                                x = vectors_flat_.data() + static_cast<size_t>(vec_id) * dim_sz;
                            }
                            float norm_half = (view.norms_sq != nullptr)
                                                  ? (view.norms_sq[ii] * 0.5f)
                                                  : vector_norms_[vec_id];
                            const float dist_score = -L2SIMD4ExtAVX(
                                const_cast<float*>(query), const_cast<float*>(x), norm_half, dimension_);
                            float l2_dist_sq = query_norm_sq + 2.0f * dist_score;
                            l2_dist_sq = std::max(0.0f, l2_dist_sq);
                            ++dist_computed;
                            if (!collector.is_prunable(l2_dist_sq)) {
                                collector.try_push(l2_dist_sq, static_cast<int>(vec_id));
                            }
                        }
                    }
                } else if (can_use_flat) {
                    std::vector<uint32_t> main_ids;
                    gather_main_range_ids(list, range_min, range_max, main_ids);
                    if (stats) stats->num_filtered_candidates += main_ids.size();
                    for (uint32_t vec_id : main_ids) {
                        if (dist_computed >= cluster_cap) {
                            break;
                        }
                        if (vec_id >= static_cast<uint32_t>(n_vectors_)) {
                            continue;
                        }
                        if (!tombstone_ids_.empty() && tombstone_ids_.count(vec_id) != 0) continue;
                        if (vectors_flat_.size() < (static_cast<size_t>(vec_id) + 1) * dim_sz) {
                            continue;
                        }
                        const float* x = vectors_flat_.data() + static_cast<size_t>(vec_id) * dim_sz;
                        const float dist_score = -L2SIMD4ExtAVX(
                            const_cast<float*>(query), const_cast<float*>(x),
                            vector_norms_[vec_id], dimension_);
                        float l2_dist_sq = query_norm_sq + 2.0f * dist_score;
                        l2_dist_sq = std::max(0.0f, l2_dist_sq);
                        ++dist_computed;
                        if (!collector.is_prunable(l2_dist_sq)) {
                            collector.try_push(l2_dist_sq, static_cast<int>(vec_id));
                        }
                    }
                }
            } else {
                if (list.main_is_packed() &&
                    list.main_size() >= static_cast<size_t>(kMainBatchL2MinRows) &&
                    tombstone_ids_.empty()) {
                    if (stats) stats->num_filtered_candidates += list.main_size();
                    scan_packed_rows_ipsimd(
                        query, dimension_, query_norm_sq, list.main_vecs.data(),
                        list.main_norms_sq.data(), list.main_ids.data(),
                        static_cast<int>(list.main_size()), collector, dist_computed,
                        cluster_cap);
                } else if (can_use_flat && !list.main_ids.empty()) {
                    if (stats) stats->num_filtered_candidates += list.main_size();
                    for (uint32_t vec_id : list.main_ids) {
                        if (dist_computed >= cluster_cap) {
                            break;
                        }
                        if (vec_id >= static_cast<uint32_t>(n_vectors_)) {
                            continue;
                        }
                        if (!tombstone_ids_.empty() && tombstone_ids_.count(vec_id) != 0) continue;
                        if (vectors_flat_.size() < (static_cast<size_t>(vec_id) + 1) * dim_sz) {
                            continue;
                        }
                        const float* x = vectors_flat_.data() + static_cast<size_t>(vec_id) * dim_sz;
                        const float dist_score = -L2SIMD4ExtAVX(
                            const_cast<float*>(query), const_cast<float*>(x),
                            vector_norms_[vec_id], dimension_);
                        float l2_dist_sq = query_norm_sq + 2.0f * dist_score;
                        l2_dist_sq = std::max(0.0f, l2_dist_sq);
                        ++dist_computed;
                        if (!collector.is_prunable(l2_dist_sq)) {
                            collector.try_push(l2_dist_sq, static_cast<int>(vec_id));
                        }
                    }
                }
            }

            scan_delta_rows_direct(
                list, use_range_filter, range_min, range_max, query, dimension_, query_norm_sq,
                vectors_flat_, vector_norms_, n_vectors_, collector, dist_computed,
                cluster_cap,
                tombstone_ids_.empty() ? nullptr : &tombstone_ids_,
                stats ? &stats->scalar_rows_scanned : nullptr,
                stats ? &stats->scalar_blocks_skipped : nullptr,
                stats ? &stats->scalar_block_rows_skipped : nullptr,
                stats ? &stats->num_filtered_candidates : nullptr);
        }
    }

    if (collector.size() == 0) {
        if (stats) {
            stats->num_dist_computations += static_cast<size_t>(dist_computed);
        }
        return {};
    }
    if (stats) {
        stats->num_dist_computations += static_cast<size_t>(dist_computed);
    }
    // tombstone 已在距离计算和入堆前过滤。
    return collector.extract_top_k(k);
}

void AdaIVFCore::train(const float* vectors, size_t n_vectors, int dimension, const float* scalars) {
    (void)scalars;
#ifdef _OPENMP
    apply_ada_ivf_train_omp_threads("train");
#endif
    // 训练入口：K-means 得到 n_clusters 质心；训练向量不写入 IVF 列表（仅学 centroids）
    // Module PQ 可选：train 末尾拟合码本
    dimension_ = dimension;
    n_vectors_ = 0;
    is_trained_ = false;
    
    // 初始化向量存储
    vectors_flat_.clear();
    vector_scalars_.clear();
    vector_norms_.clear();
    pq_codes_.clear();
    tombstone_ids_.clear();
    compacted_deleted_count_ = 0;
    compact_cursor_cluster_ = 0;
    
    // 预分配容量
    reserve_capacity(n_vectors);
    
    // 初始化聚类中心（使用K-means训练）
    centroids_.clear();
    centroids_.resize(n_clusters_);
    
    kmeans_->set_attr_lambda(attr_lambda_);
    kmeans_->set_scalar_span_lambda(scalar_span_lambda_);
    kmeans_->set_random_seed(kmeans_seed_);
    kmeans_->set_use_kmeanspp(use_kmeanspp_);

    // 使用固定 seed 的 k-means++ 训练；attr_lambda>0 且 scalars 非空时启用属性感知距离。
    kmeans_->train(vectors, n_vectors, dimension, n_clusters_, scalars);
    centroids_ = kmeans_->get_centroids();
    
    // 保存初始质心（用于计算漂移 f_d）
    initial_centroids_ = centroids_;
    rebuild_centroid_norms_sq();
    reset_incremental_centroid_state();
    cluster_scalar_mins_.assign(static_cast<size_t>(n_clusters_), 0.0f);
    cluster_scalar_maxs_.assign(static_cast<size_t>(n_clusters_), 0.0f);
    cluster_scalar_bounds_valid_.assign(static_cast<size_t>(n_clusters_), 0u);
    cluster_scalar_synopses_.assign(static_cast<size_t>(n_clusters_), ScalarSynopsis{});
    
    // 初始化IVF lists（空列表，训练向量不插入索引）
    ivf_lists_.clear();
    ivf_lists_.resize(n_clusters_);
    float hist_domain_min = 0.0f;
    float hist_domain_max = 1.0f;
    if (scalars != nullptr && n_vectors > 0) {
        hist_domain_min = scalars[0];
        hist_domain_max = scalars[0];
        for (size_t i = 1; i < n_vectors; ++i) {
            hist_domain_min = std::min(hist_domain_min, scalars[i]);
            hist_domain_max = std::max(hist_domain_max, scalars[i]);
        }
        if (!(hist_domain_min < hist_domain_max)) {
            hist_domain_min -= 0.5f;
            hist_domain_max += 0.5f;
        }
    }
    scalar_synopsis_domain_min_ = hist_domain_min;
    scalar_synopsis_domain_max_ = hist_domain_max;
    scalar_synopsis_domain_valid_ = hist_domain_min < hist_domain_max;
    for (auto& list : ivf_lists_) {
        list.set_dimension(dimension_);
        list.set_histogram_domain(hist_domain_min, hist_domain_max);
    }
    
    // 预分配容量
    size_t avg_list_size = (n_vectors / n_clusters_) + (n_vectors / n_clusters_ / 5);
    for (int i = 0; i < n_clusters_; ++i) {
        ivf_lists_[i].reserve_main(avg_list_size);
    }
    
    // Module PQ：用训练集在内存中拟合码本（不写入 IVF）
    if (enable_pq_compression_) {
        pq_.set_params(pq_m_, pq_ksub_);
        pq_.train(vectors, n_vectors, dimension_);
    } else {
        pq_.clear();
    }
    
    // 训练仅用于训练聚簇中心，不插入训练向量
    n_vectors_ = 0;
    is_trained_ = true;
}

void AdaIVFCore::train_with_centroids(
    const float* vectors,
    size_t n_vectors,
    int dimension,
    const float* scalars,
    const std::vector<std::vector<float>>& centroids,
    const std::vector<float>& scalar_centroids,
    const std::vector<float>& scalar_mins,
    const std::vector<float>& scalar_maxs) {
#ifdef _OPENMP
    apply_ada_ivf_train_omp_threads("train_with_centroids");
#endif
    if (vectors == nullptr || n_vectors == 0) {
        throw std::runtime_error("empty training vectors");
    }
    if (centroids.empty()) {
        throw std::runtime_error("external centroids are empty");
    }
    if (static_cast<int>(centroids.size()) != n_clusters_) {
        throw std::runtime_error("external centroid count must equal n_clusters");
    }
    for (const auto& c : centroids) {
        if (static_cast<int>(c.size()) != dimension) {
            throw std::runtime_error("external centroid dimension mismatch");
        }
    }

    dimension_ = dimension;
    n_vectors_ = 0;
    is_trained_ = false;
    vectors_flat_.clear();
    vector_scalars_.clear();
    vector_norms_.clear();
    pq_codes_.clear();
    tombstone_ids_.clear();
    compacted_deleted_count_ = 0;
    compact_cursor_cluster_ = 0;
    reserve_capacity(n_vectors);

    centroids_ = centroids;
    initial_centroids_ = centroids_;
    rebuild_centroid_norms_sq();
    reset_incremental_centroid_state();

    kmeans_->set_centroids(centroids_);
    kmeans_->set_attr_lambda(attr_lambda_);
    kmeans_->set_scalar_span_lambda(scalar_span_lambda_);
    kmeans_->set_random_seed(kmeans_seed_);
    kmeans_->set_use_kmeanspp(use_kmeanspp_);
    float scalar_mean = 0.0f;
    float scalar_scale = 1.0f;
    kmeans_compute_scalar_norm(scalars, n_vectors, scalar_mean, scalar_scale);
    kmeans_->set_scalar_normalization(scalar_mean, scalar_scale);
    if (scalar_centroids.size() == centroids_.size() &&
        scalar_mins.size() == centroids_.size() &&
        scalar_maxs.size() == centroids_.size()) {
        kmeans_->set_scalar_metadata(scalar_centroids, scalar_mins, scalar_maxs);
    }

    cluster_scalar_mins_.assign(static_cast<size_t>(n_clusters_), 0.0f);
    cluster_scalar_maxs_.assign(static_cast<size_t>(n_clusters_), 0.0f);
    cluster_scalar_bounds_valid_.assign(static_cast<size_t>(n_clusters_), 0u);
    cluster_scalar_synopses_.assign(static_cast<size_t>(n_clusters_), ScalarSynopsis{});
    ivf_lists_.clear();
    ivf_lists_.resize(n_clusters_);

    float hist_domain_min = 0.0f;
    float hist_domain_max = 1.0f;
    if (scalars != nullptr && n_vectors > 0) {
        hist_domain_min = scalars[0];
        hist_domain_max = scalars[0];
        for (size_t i = 1; i < n_vectors; ++i) {
            hist_domain_min = std::min(hist_domain_min, scalars[i]);
            hist_domain_max = std::max(hist_domain_max, scalars[i]);
        }
        if (!(hist_domain_min < hist_domain_max)) {
            hist_domain_min -= 0.5f;
            hist_domain_max += 0.5f;
        }
    }
    scalar_synopsis_domain_min_ = hist_domain_min;
    scalar_synopsis_domain_max_ = hist_domain_max;
    scalar_synopsis_domain_valid_ = hist_domain_min < hist_domain_max;
    for (auto& list : ivf_lists_) {
        list.set_dimension(dimension_);
        list.set_histogram_domain(hist_domain_min, hist_domain_max);
    }

    const size_t avg_list_size = (n_vectors / static_cast<size_t>(n_clusters_)) +
        (n_vectors / static_cast<size_t>(n_clusters_) / 5);
    for (int i = 0; i < n_clusters_; ++i) {
        ivf_lists_[i].reserve_main(avg_list_size);
        if (cluster_stats_[i]) {
            cluster_stats_[i]->size = 0;
            cluster_stats_[i]->insert_count_since_recluster = 0;
            cluster_stats_[i]->quality_score = 1.0f;
            cluster_stats_[i]->temperature = 1.0f;
            cluster_stats_[i]->is_deleted = false;
        }
    }

    if (enable_pq_compression_) {
        pq_.set_params(pq_m_, pq_ksub_);
        pq_.train(vectors, n_vectors, dimension_);
    } else {
        pq_.clear();
    }

    n_vectors_ = 0;
    is_trained_ = true;
}

void AdaIVFCore::add(const float* vectors, size_t n_vectors, const int* ids,
                    bool auto_recluster, const float* scalars) {
    // 插入入口：逐条 assign→写入 RangeAwareList→(维护)更新质心/局部重聚类/全局重建
    if (!is_trained_) {
        throw std::runtime_error("索引未训练，无法插入向量");
    }
#ifdef _OPENMP
    apply_ada_ivf_omp_threads("add");
#endif
    
    // 性能分析：记录各阶段耗时
    auto total_start = std::chrono::high_resolution_clock::now();
    double time_reserve = 0, time_vector_copy = 0, time_norm = 0, time_assign = 0;
    double time_binary_search = 0, time_insert = 0, time_lock = 0;
    double time_centroid_update = 0, time_recluster = 0;
    
    // 扩展向量存储
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t new_size = vector_scalars_.size() + n_vectors;
    reserve_capacity(new_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    time_reserve = std::chrono::duration<double>(t1 - t0).count();
    
    std::set<int> clusters_exceeding_limit;
    const bool track_maintenance = enable_maintenance_;

    // 并行预分配簇 id（assign 只读质心，可 OpenMP；写入 IVF 仍串行保序）
    std::vector<uint32_t> batch_cluster_ids(n_vectors);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_vectors >= 256)
#endif
    for (size_t i = 0; i < n_vectors; ++i) {
        const float scalar_val = (scalars != nullptr) ? scalars[i] : 0.0f;
        uint32_t cluster_id = kmeans_->assign(
            vectors + i * static_cast<size_t>(dimension_), scalar_val);
        if (cluster_id >= static_cast<uint32_t>(n_clusters_)) {
            cluster_id = static_cast<uint32_t>(n_clusters_ - 1);
        }
        batch_cluster_ids[i] = cluster_id;
    }

    // 直接插入方式（旧代码方式，小数据量时性能更好）
    for (size_t i = 0; i < n_vectors; ++i) {
        // 使用 vector_scalars_.size() 作为新向量的 ID，确保连续且唯一
        uint32_t vec_id = static_cast<uint32_t>(vector_scalars_.size());
        const float* vec_ptr = vectors + i * dimension_;
        
        // 存储向量
        auto t2 = std::chrono::high_resolution_clock::now();
        const bool keep_floats = (!enable_pq_compression_) || (!pq_release_floats_);
        const bool keep_list_payload = keep_floats && keep_list_packed_payload_;
        if (keep_floats) {
            vectors_flat_.insert(vectors_flat_.end(), vec_ptr, vec_ptr + dimension_);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        time_vector_copy += std::chrono::duration<double>(t3 - t2).count();
        
        // 计算并存储范数（使用简单循环，与旧代码一致）
        auto t4 = std::chrono::high_resolution_clock::now();
        float norm_sq_full = 0.0f;
        if (keep_floats) {
            for (int d = 0; d < dimension_; ++d) {
                norm_sq_full += vec_ptr[d] * vec_ptr[d];
            }
            vector_norms_.push_back(norm_sq_full / 2.0f);
        } else {
            // 不保留 float 时，norm 仅为占位（全精度距离路径不会使用）
            vector_norms_.push_back(0.0f);
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        time_norm += std::chrono::duration<double>(t5 - t4).count();
        
        // 存储标量值
        float scalar_val = (scalars != nullptr) ? scalars[i] : 0.0f;
        vector_scalars_.push_back(scalar_val);
        
        // 分配到最近的聚类（已在上方并行完成）
        auto t6 = std::chrono::high_resolution_clock::now();
        uint32_t cluster_id = batch_cluster_ids[i];
        add_vector_to_cluster_sum(static_cast<int>(cluster_id), vec_ptr);
        add_scalar_to_cluster_sum(static_cast<int>(cluster_id), scalar_val);
        observe_cluster_scalar_bounds(static_cast<int>(cluster_id), scalar_val);
        observe_cluster_scalar_synopsis(static_cast<int>(cluster_id), scalar_val);
        auto t7 = std::chrono::high_resolution_clock::now();
        time_assign += std::chrono::duration<double>(t7 - t6).count();
        
        // 写入 IVF 列表：Module A 走无序段+归并；否则维持原「main 内二分插入」
        auto t8 = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<AdaIvfListMutex> lock(*list_locks_[cluster_id]);
            auto t9 = std::chrono::high_resolution_clock::now();
            time_lock += std::chrono::duration<double>(t9 - t8).count();
            
            RangeAwareList& list = ivf_lists_[cluster_id];
            list.set_dimension(dimension_);

            if (enable_scalar_filter_lsm_) {
                // Module A：函数1 add_into_unorder — 先入无序段，再 trigger/merge
                auto t10 = std::chrono::high_resolution_clock::now();
                if (keep_list_payload) {
                    list.add_into_unordered(vec_id, scalar_val, vec_ptr, norm_sq_full, dimension_);
                } else {
                    list.add_into_unordered(vec_id, scalar_val);
                }
                lsm_trigger_merge_if_needed(list);
                auto t11 = std::chrono::high_resolution_clock::now();
                time_binary_search += std::chrono::duration<double>(t11 - t10).count();
            } else {
                auto t10 = std::chrono::high_resolution_clock::now();
                auto it = std::lower_bound(list.main_scalars.begin(), list.main_scalars.end(), scalar_val);
                size_t pos = std::distance(list.main_scalars.begin(), it);
                auto t11 = std::chrono::high_resolution_clock::now();
                time_binary_search += std::chrono::duration<double>(t11 - t10).count();
                auto t12 = std::chrono::high_resolution_clock::now();
                if (keep_list_payload) {
                    list.insert_main_at(pos, vec_id, scalar_val, vec_ptr, norm_sq_full);
                } else {
                    list.insert_main_metadata_at(pos, vec_id, scalar_val);
                }
                auto t13 = std::chrono::high_resolution_clock::now();
                time_insert += std::chrono::duration<double>(t13 - t12).count();
            }
            
            cluster_stats_[cluster_id]->size++;
            if (track_maintenance) {
                cluster_stats_[cluster_id]->insert_count_since_recluster++;
                if (list.total_size() > get_dynamic_max_cluster_size()) {
                    mark_hot_cluster(static_cast<int>(cluster_id));
                }
            }
        }

        // Module PQ：编码为 m 字节/向量；可选释放原始 float
        if (enable_pq_compression_) {
            if (!pq_.is_trained()) {
                throw std::runtime_error(
                    "enable_pq_compression 为 true 但 PQ 未训练：请 train() 前打开开关并训练，或调用 train_pq_from_samples");
            }
            if (pq_codes_.size() != static_cast<size_t>(vec_id)) {
                throw std::runtime_error("pq_codes_ 与向量 id 不同步");
            }
            std::vector<uint8_t> code;
            pq_.encode(vec_ptr, code);
            pq_codes_.push_back(std::move(code));
        }
    }

    // 维护逻辑会用 n_vectors_ 判断 vec_id 是否有效；先刷新，避免刚插入的向量在重聚类中被跳过。
    n_vectors_ = vector_scalars_.size();
    
    // 定期更新聚类中心（Baseline 模式：关闭 Ada 维护时跳过）
    if (enable_maintenance_) {
        inserted_since_fine_refresh_ += n_vectors;
        insert_count_since_centroid_update_ += n_vectors;
        if (insert_count_since_centroid_update_ >= config_.centroid_update_interval) {
            auto t14 = std::chrono::high_resolution_clock::now();
            refresh_dirty_centroids_from_sums();
            auto t15 = std::chrono::high_resolution_clock::now();
            time_centroid_update = std::chrono::duration<double>(t15 - t14).count();
            insert_count_since_centroid_update_ = 0;
        }
    }
    
    // 批量重聚类检查（维护关闭时整段跳过，不计时、不算 G/ε）
    if (enable_maintenance_) {
        auto t16 = std::chrono::high_resolution_clock::now();

        ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] add: 开始重聚类检查，n_vectors=" << n_vectors << ", clusters_exceeding_limit.size()=" << clusters_exceeding_limit.size() << std::endl);

        process_hot_clusters();

        if (auto_recluster) {
            ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] add: 检查常规重聚类..." << std::endl);
            auto identify_start = std::chrono::high_resolution_clock::now();
            std::vector<MaintenanceCandidate> to_recluster = identify_clusters_to_recluster();
            auto identify_end = std::chrono::high_resolution_clock::now();
            double identify_time = std::chrono::duration<double>(identify_end - identify_start).count();
            ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] add: 识别出 " << to_recluster.size() << " 个需要重聚类的聚类，识别耗时: " << identify_time << " 秒" << std::endl);

            for (const MaintenanceCandidate& candidate : to_recluster) {
                const int cid = candidate.cluster_id;
                if (clusters_exceeding_limit.find(cid) == clusters_exceeding_limit.end()) {
                    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] add: 重聚类聚类 " << cid << " (大小: " << ivf_lists_[cid].total_size() << ")..." << std::endl);
                    auto recluster_start = std::chrono::high_resolution_clock::now();
                    recluster_cluster(cid);
                    local_recluster_count_++;
                    auto recluster_end = std::chrono::high_resolution_clock::now();
                    double recluster_time = std::chrono::duration<double>(recluster_end - recluster_start).count();
                    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] add: 聚类 " << cid << " 重聚类完成，耗时: " << recluster_time << " 秒" << std::endl);
                }
            }

            ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] add: 检查 fine 质心刷新..." << std::endl);
            check_and_refresh_fine_centroids();
            ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] add: fine 质心刷新检查完成" << std::endl);
        }

        auto t17 = std::chrono::high_resolution_clock::now();
        time_recluster = std::chrono::duration<double>(t17 - t16).count();
    }
    
    // 输出性能分析（使用原子变量保证线程安全）
    static std::atomic<size_t> total_inserted{0};
    size_t current_total = total_inserted.fetch_add(n_vectors, std::memory_order_relaxed) + n_vectors;
    if (current_total % config_.perf_analysis_interval == 0 || n_vectors >= config_.perf_analysis_interval) {
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        
        ADA_IVF_DEBUG_OUT("\n[性能分析] 插入 " << n_vectors << " 个向量 (累计: " << current_total << "):\n");
        ADA_IVF_DEBUG_OUT("  总时间: " << std::fixed << std::setprecision(3) << total_time << "s\n");
        ADA_IVF_DEBUG_OUT("  内存预留: " << time_reserve * 1000 << "ms (" << (time_reserve/total_time*100) << "%)\n");
        ADA_IVF_DEBUG_OUT("  向量拷贝: " << time_vector_copy * 1000 << "ms (" << (time_vector_copy/total_time*100) << "%)\n");
        ADA_IVF_DEBUG_OUT("  范数计算: " << time_norm * 1000 << "ms (" << (time_norm/total_time*100) << "%)\n");
        ADA_IVF_DEBUG_OUT("  聚类分配: " << time_assign * 1000 << "ms (" << (time_assign/total_time*100) << "%)\n");
        ADA_IVF_DEBUG_OUT("  锁等待: " << time_lock * 1000 << "ms (" << (time_lock/total_time*100) << "%)\n");
        ADA_IVF_DEBUG_OUT("  二分查找: " << time_binary_search * 1000 << "ms (" << (time_binary_search/total_time*100) << "%)\n");
        ADA_IVF_DEBUG_OUT("  向量插入: " << time_insert * 1000 << "ms (" << (time_insert/total_time*100) << "%)\n");
        if (time_centroid_update > 0) {
            ADA_IVF_DEBUG_OUT("  中心更新: " << time_centroid_update * 1000 << "ms (" << (time_centroid_update/total_time*100) << "%)\n");
        }
        if (time_recluster > 0) {
            ADA_IVF_DEBUG_OUT("  重聚类: " << time_recluster * 1000 << "ms (" << (time_recluster/total_time*100) << "%)\n");
        }
        ADA_IVF_DEBUG_OUT(std::endl);
    }
}

std::vector<std::pair<float, int>> AdaIVFCore::search(
    const float* query, int k, int max_num_distances,
    float range_min, float range_max) const {
    if (!is_trained_ || n_vectors_ == 0) {
        return {};
    }

    const bool do_profile = ivf_profile_search_enabled();
    const int profile_qidx = do_profile ? ivf_profile_search_claim_query_index() : -1;
    const bool emit_profile =
        do_profile && profile_qidx < ivf_profile_search_max_queries();
    auto t_search_begin = std::chrono::high_resolution_clock::now();

    // 调试信息：静态计数器，只输出前10个查询的详细信息
    static int debug_query_count = 0;
    bool enable_debug = (debug_query_count++ < 10);
    
    auto t_fine_score_begin = std::chrono::high_resolution_clock::now();
    std::vector<float> scores;
    compute_cluster_scores(query, scores);
    auto t_fine_score_end = std::chrono::high_resolution_clock::now();
    
    auto t_fine_select_begin = std::chrono::high_resolution_clock::now();
    std::vector<int> selected_clusters;
    select_top_k_clusters(scores, n_probe_, selected_clusters);
    auto t_fine_select_end = std::chrono::high_resolution_clock::now();
    
    if (selected_clusters.empty()) {
        if (enable_debug) {
            ADA_IVF_DEBUG_OUT("[搜索调试] 查询#" << debug_query_count 
                      << ": 没有选中的聚类" << std::endl);
        }
        if (emit_profile) {
            const double ms_score = ivf_ms_between(t_fine_score_begin, t_fine_score_end);
            const double ms_select = ivf_ms_between(t_fine_select_begin, t_fine_select_end);
            dump_single_search_profile(
                profile_qidx, n_probe_, ms_score, ms_select, 0.0, 0, 0);
        }
        return {};
    }
    
    if (enable_debug) {
        ADA_IVF_DEBUG_OUT("[搜索调试] 查询#" << debug_query_count 
                  << ": 选中聚类数=" << selected_clusters.size() 
                  << "/" << n_clusters_ << ", n_probe=" << n_probe_
                  << ", 范围=[" << range_min << ", " << range_max << "]"
                  << ", k=" << k << ", max_num_distances=" << max_num_distances << std::endl);
    }
    
    auto t_search_clusters_begin = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<float, int>> all_candidates =
        scan_cluster_list_for_topk(query, k, max_num_distances, selected_clusters, range_min, range_max);
    auto t_search_clusters_end = std::chrono::high_resolution_clock::now();

    if (enable_debug) {
        ADA_IVF_DEBUG_OUT("[搜索统计] 候选条数=" << all_candidates.size() << std::endl);
    }

    if (all_candidates.empty()) {
        if (enable_debug) {
            ADA_IVF_DEBUG_OUT("[搜索警告] 没有找到任何候选！" << std::endl);
        }
        if (emit_profile) {
            const double ms_score = ivf_ms_between(t_fine_score_begin, t_fine_score_end);
            const double ms_select = ivf_ms_between(t_fine_select_begin, t_fine_select_end);
            const double ms_search =
                ivf_ms_between(t_search_clusters_begin, t_search_clusters_end);
            dump_single_search_profile(
                profile_qidx,
                n_probe_,
                ms_score,
                ms_select,
                ms_search,
                static_cast<int>(selected_clusters.size()),
                0);
        }
        return {};
    }

    if (enable_debug && !all_candidates.empty()) {
        ADA_IVF_DEBUG_OUT("[返回候选] 前5个距离=[" << all_candidates[0].first);
        for (size_t i = 1; i < std::min(5UL, all_candidates.size()); ++i) {
            ADA_IVF_DEBUG_OUT(", " << all_candidates[i].first);
        }
        ADA_IVF_DEBUG_OUT("]" << std::endl);
    }

    if (enable_debug) {
        ADA_IVF_DEBUG_OUT("[返回结果] 返回" << all_candidates.size() << "个结果" << std::endl);
    }

    if (emit_profile) {
        const double ms_score = ivf_ms_between(t_fine_score_begin, t_fine_score_end);
        const double ms_select = ivf_ms_between(t_fine_select_begin, t_fine_select_end);
        const double ms_search =
            ivf_ms_between(t_search_clusters_begin, t_search_clusters_end);
        dump_single_search_profile(
            profile_qidx,
            n_probe_,
            ms_score,
            ms_select,
            ms_search,
            static_cast<int>(selected_clusters.size()),
            static_cast<int>(all_candidates.size()));
    }

    return all_candidates;
}

std::vector<std::pair<float, int>> AdaIVFCore::search_in_clusters(
    const float* query, int k, int max_num_distances,
    const std::vector<int>& cluster_ids,
    float range_min, float range_max) const {
    if (!is_trained_ || n_vectors_ == 0 || cluster_ids.empty()) {
        return {};
    }

    return scan_cluster_list_for_topk(query, k, max_num_distances, cluster_ids, range_min, range_max);
}

std::vector<std::pair<float, int>> AdaIVFCore::search_in_clusters_with_stats(
    const float* query, int k, int max_num_distances,
    const std::vector<int>& cluster_ids,
    float range_min, float range_max,
    SearchStats* stats) const {
    if (!is_trained_ || n_vectors_ == 0 || cluster_ids.empty()) {
        return {};
    }

    return scan_cluster_list_for_topk(
        query, k, max_num_distances, cluster_ids, range_min, range_max, stats);
}

std::vector<std::vector<std::pair<float, int>>> AdaIVFCore::batch_search(
    const float* queries, size_t n_queries, int k, int max_num_distances,
    float range_min, float range_max) const {
    ivf_profile_search_maybe_reset();
    std::vector<std::vector<std::pair<float, int>>> results(n_queries);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_queries > 1)
#endif
    for (size_t i = 0; i < n_queries; ++i) {
        results[i] = search(
            queries + i * static_cast<size_t>(dimension_),
            k,
            max_num_distances,
            range_min,
            range_max);
    }

    return results;
}

std::vector<std::vector<std::pair<float, int>>> AdaIVFCore::batch_search_ranges(
    const float* queries, const float* ranges, size_t n_queries, int k, int max_num_distances) const {
    ivf_profile_search_maybe_reset();
    std::vector<std::vector<std::pair<float, int>>> results(n_queries);
    if (ranges == nullptr) {
        return batch_search(queries, n_queries, k, max_num_distances,
                            std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::max());
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_queries > 1)
#endif
    for (size_t i = 0; i < n_queries; ++i) {
        const float rmin = ranges[i * 2];
        const float rmax = ranges[i * 2 + 1];
        results[i] = search(
            queries + i * static_cast<size_t>(dimension_),
            k,
            max_num_distances,
            rmin,
            rmax);
    }

    return results;
}

AdaIVFCore::BatchSearchResult AdaIVFCore::batch_search_with_stats(
    const float* queries, size_t n_queries, int k, int max_num_distances,
    float range_min, float range_max) const {
    ivf_profile_search_maybe_reset();
    BatchSearchResult out;
    out.results.resize(n_queries);
    out.per_query_stats.resize(n_queries);
    if (!is_trained_ || n_vectors_ == 0) {
        return out;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_queries > 1)
#endif
    for (size_t i = 0; i < n_queries; ++i) {
        std::vector<float> scores;
        compute_cluster_scores(queries + i * static_cast<size_t>(dimension_), scores);
        std::vector<int> selected_clusters;
        select_top_k_clusters(scores, n_probe_, selected_clusters);
        out.results[i] = scan_cluster_list_for_topk(
            queries + i * static_cast<size_t>(dimension_),
            k,
            max_num_distances,
            selected_clusters,
            range_min,
            range_max,
            &out.per_query_stats[i]);
    }
    return out;
}

AdaIVFCore::BatchSearchResult AdaIVFCore::batch_search_ranges_with_stats(
    const float* queries, const float* ranges, size_t n_queries, int k, int max_num_distances) const {
    if (ranges == nullptr) {
        return batch_search_with_stats(
            queries, n_queries, k, max_num_distances,
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::max());
    }
    ivf_profile_search_maybe_reset();
    BatchSearchResult out;
    out.results.resize(n_queries);
    out.per_query_stats.resize(n_queries);
    if (!is_trained_ || n_vectors_ == 0) {
        return out;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_queries > 1)
#endif
    for (size_t i = 0; i < n_queries; ++i) {
        std::vector<float> scores;
        compute_cluster_scores(queries + i * static_cast<size_t>(dimension_), scores);
        std::vector<int> selected_clusters;
        select_top_k_clusters(scores, n_probe_, selected_clusters);
        const float rmin = ranges[i * 2];
        const float rmax = ranges[i * 2 + 1];
        out.results[i] = scan_cluster_list_for_topk(
            queries + i * static_cast<size_t>(dimension_),
            k,
            max_num_distances,
            selected_clusters,
            rmin,
            rmax,
            &out.per_query_stats[i]);
    }
    return out;
}

void AdaIVFCore::invalidate_centroid_neighbor_cache() {
    centroid_neighbor_cache_.clear();
    centroid_neighbor_cache_k_ = 0;
}

std::vector<int> AdaIVFCore::get_cached_centroid_neighbors(int cluster_id, int count) {
    if (cluster_id < 0 || cluster_id >= n_clusters_ || count <= 0 || centroids_[cluster_id].empty()) {
        return {};
    }
    count = std::min(count, std::max(0, n_clusters_ - 1));
    if (count <= 0) {
        return {};
    }
    if (centroid_neighbor_cache_.size() != static_cast<size_t>(n_clusters_) ||
        centroid_neighbor_cache_k_ < count) {
        centroid_neighbor_cache_.assign(static_cast<size_t>(n_clusters_), std::vector<int>());
        centroid_neighbor_cache_k_ = count;
    }
    std::vector<int>& cached = centroid_neighbor_cache_[static_cast<size_t>(cluster_id)];
    if (cached.empty()) {
        std::vector<std::pair<float, int>> centroid_dists;
        centroid_dists.reserve(static_cast<size_t>(std::max(0, n_clusters_ - 1)));
        const float* base_centroid = centroids_[cluster_id].data();
        for (int j = 0; j < n_clusters_; ++j) {
            if (j == cluster_id || cluster_stats_[j]->is_deleted || centroids_[j].empty()) {
                continue;
            }
            centroid_dists.emplace_back(compute_distance_sq(base_centroid, centroids_[j].data()), j);
        }
        const int take = std::min(centroid_neighbor_cache_k_, static_cast<int>(centroid_dists.size()));
        if (take > 0) {
            std::partial_sort(
                centroid_dists.begin(), centroid_dists.begin() + take, centroid_dists.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first < b.first;
                });
            cached.reserve(static_cast<size_t>(take));
            for (int t = 0; t < take; ++t) {
                cached.push_back(centroid_dists[static_cast<size_t>(t)].second);
            }
        }
    }
    if (static_cast<int>(cached.size()) <= count) {
        return cached;
    }
    return std::vector<int>(cached.begin(), cached.begin() + count);
}

void AdaIVFCore::recluster_cluster(int cluster_id) {
    if (!enable_maintenance_) return;
    if (cluster_id < 0 || cluster_id >= n_clusters_) return;
    
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 开始处理聚类 " << cluster_id << std::endl);
    auto recluster_start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<int> clusters_to_lock;
    clusters_to_lock.push_back(cluster_id);
    
    // 先不加锁地检查哪些聚类需要更新（避免嵌套锁）
    {
        std::lock_guard<AdaIvfListMutex> lock(*list_locks_[cluster_id]);
        RangeAwareList& list = ivf_lists_[cluster_id];
        // Module A：重聚类前先把无序段并入有序段，避免丢 delta 内 id
        list.merge_unordered_into_main(keep_list_packed_payload_);
        
        ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 聚类 " << cluster_id << " 大小: " << list.total_size() << std::endl);
        if (list.total_size() < 2) {
            ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 聚类 " << cluster_id << " 大小 < 2，跳过" << std::endl);
            return;
        }
    }
    
    // 重新分配向量（先不加锁计算，避免长时间持锁）。
    // 只记录 touched clusters，避免为 4096+ 个簇反复分配空 vector。
    std::unordered_map<int, std::vector<uint32_t>> new_assignments;
    new_assignments.reserve(static_cast<size_t>(std::min(n_clusters_, 64)));
    
    // 提取该聚类的所有向量（需要短暂持锁）
    std::vector<uint32_t> vector_ids;
    {
        std::lock_guard<AdaIvfListMutex> lock(*list_locks_[cluster_id]);
        RangeAwareList& list = ivf_lists_[cluster_id];
        list.merge_unordered_into_main(keep_list_packed_payload_);
        vector_ids = list.main_ids;
    }
    
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 提取了 " << vector_ids.size() << " 个向量，开始重新分配..." << std::endl);
    
    if (vector_ids.empty()) {
        ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 向量列表为空，跳过" << std::endl);
        return;
    }
    
    // 回退：不再使用sub_centroids映射（避免性能问题）
    // 直接使用全局centroids重新分配向量
    
    // 优化：使用SIMD加速距离计算（L2距离平方）
    auto compute_dist_sq_simd = [this](const float* vec, const float* centroid) -> float {
        // 使用SIMD优化的内积计算
        float ip = IPSIMD4ExtAVX(const_cast<float*>(vec), const_cast<float*>(centroid), dimension_);
        
        // 计算L2距离平方：||v1 - v2||^2 = ||v1||^2 + ||v2||^2 - 2*IP
        // 优化：使用预计算的范数（如果可用）
        float norm_v_sq = IPSIMD4ExtAVX(const_cast<float*>(vec), const_cast<float*>(vec), dimension_);
        float norm_c_sq = IPSIMD4ExtAVX(const_cast<float*>(centroid), const_cast<float*>(centroid), dimension_);
        return norm_v_sq + norm_c_sq - 2.0f * ip;
    };
    
    // 两阶段策略平衡性能和召回率。
    // 阶段1不再使用 cluster_id 的编号邻域；编号通常没有几何意义。
    // 这里先按质心距离找一批真正邻近的候选簇，后续每个向量只在这些候选里快速重分配。
    const int local_search_count = std::min(
        std::max(1, static_cast<int>(n_probe_ * config_.local_search_radius_factor)),
        n_clusters_);
    const int global_search_count = std::min(static_cast<int>(n_probe_), n_clusters_);

    std::vector<int> local_candidate_clusters =
        get_cached_centroid_neighbors(cluster_id, local_search_count);
    
    std::vector<float> dec;
    for (uint32_t vec_id : vector_ids) {
        if (vec_id >= n_vectors_) {
            continue;
        }
        decode_vector_for_id(vec_id, dec);
        const float* vptr = dec.data();
        
        // 先计算到原聚类的距离作为基准
        float base_dist_sq = compute_dist_sq_simd(vptr, centroids_[cluster_id].data());
        float min_dist_sq = base_dist_sq;
        int nearest = cluster_id;
        
        // 阶段1：检查几何上邻近的簇，而不是编号相邻的簇。
        for (int j : local_candidate_clusters) {
            if (j == cluster_id || cluster_stats_[j]->is_deleted || centroids_[j].empty()) {
                continue;
            }
            float dist_sq = compute_dist_sq_simd(vptr, centroids_[j].data());
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest = j;
            }
        }
        
        // 阶段2：如果附近聚类不够好（距离改善<阈值），检查全局采样簇
        // 优化：使用简单的步进采样策略，避免遍历所有聚类
        const float improvement =
            (base_dist_sq > 1e-12f) ? ((base_dist_sq - min_dist_sq) / base_dist_sq) : 0.0f;
        if (nearest == cluster_id || improvement < config_.reassignment_improvement_threshold) {
            // 优化：使用步进采样，每隔一定间隔检查一个聚类，避免遍历所有聚类
            // 这样可以大幅减少计算量，同时保持召回率
            const int step = std::max(1, n_clusters_ / global_search_count);  // 步进间隔
            
            // 从随机起始点开始，步进采样（最多检查global_search_count个聚类）
            int start_offset = (cluster_id * 7 + vec_id) % step;  // 简单的伪随机起始点
            int checked_count = 0;
            for (int j = start_offset; j < n_clusters_ && checked_count < global_search_count; j += step) {
                if (j != cluster_id && !cluster_stats_[j]->is_deleted) {
                    float dist_sq = compute_dist_sq_simd(vptr, centroids_[j].data());
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        nearest = j;
                    }
                    checked_count++;
                }
            }
        }
        
        new_assignments[nearest].push_back(vec_id);
    }
    
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 重新分配完成，统计分配结果..." << std::endl);
    size_t total_reassigned = 0;
    for (const auto& kv : new_assignments) {
        const int j = kv.first;
        total_reassigned += kv.second.size();
        if (j != cluster_id) {
            clusters_to_lock.push_back(j);
        }
    }
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 共重新分配 " << total_reassigned << " 个向量" << std::endl);
    
    std::sort(clusters_to_lock.begin(), clusters_to_lock.end());
    clusters_to_lock.erase(std::unique(clusters_to_lock.begin(), clusters_to_lock.end()), clusters_to_lock.end());
    
#if ADA_IVF_DEBUG
    std::cerr << "[DEBUG] recluster_cluster: 需要锁定 " << clusters_to_lock.size() << " 个聚类: [";
    for (size_t i = 0; i < clusters_to_lock.size(); ++i) {
        std::cerr << clusters_to_lock[i];
        if (i < clusters_to_lock.size() - 1) std::cerr << ", ";
    }
    std::cerr << "]" << std::endl;
#endif
    
    // 注意：由于C++11不支持std::scoped_lock，我们使用手动顺序锁定
    std::vector<std::unique_lock<AdaIvfListMutex>> locks;
    locks.reserve(clusters_to_lock.size());
    
    auto lock_start_time = std::chrono::high_resolution_clock::now();
    for (int cid : clusters_to_lock) {
        locks.emplace_back(*list_locks_[cid]);
    }
    auto lock_end_time = std::chrono::high_resolution_clock::now();
    double lock_time = std::chrono::duration<double>(lock_end_time - lock_start_time).count();
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 获取所有锁耗时: " << lock_time << " 秒" << std::endl);
    
    // 现在所有锁都已获取，可以安全地操作所有聚类
    RangeAwareList& list = ivf_lists_[cluster_id];
    
    // 清空原聚类（main + delta 一并清空）
    list.clear_all();
    
    // 优化：批量插入，而不是逐个插入（大幅提升性能）
    // 会直接导致索引丢数据（召回率骤降、聚类平均大小异常偏小）。
    // 先把属于cluster_id的向量写回本聚类列表（按scalar有序插入）。
    auto source_assignment_it = new_assignments.find(cluster_id);
    if (source_assignment_it != new_assignments.end() && !source_assignment_it->second.empty()) {
        // 优化：先收集所有需要插入的向量和标量值，然后排序，最后批量插入
        const std::vector<uint32_t>& source_assignment = source_assignment_it->second;
        std::vector<std::pair<float, uint32_t>> to_insert;
        to_insert.reserve(source_assignment.size());
        for (uint32_t vec_id : source_assignment) {
            float scalar = (vec_id < vector_scalars_.size()) ? vector_scalars_[vec_id] : 0.0f;
            to_insert.emplace_back(scalar, vec_id);
        }
        
        // 排序
        std::sort(to_insert.begin(), to_insert.end(),
                 [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                     return a.first < b.first;
                 });
        
        // 批量插入（如果列表为空，直接赋值；否则归并）
        if (list.main_scalars.empty()) {
            list.main_scalars.reserve(to_insert.size());
            list.main_ids.reserve(to_insert.size());
            for (const auto& pair : to_insert) {
                list.main_scalars.push_back(pair.first);
                list.main_ids.push_back(pair.second);
            }
        } else {
            // 归并两个已排序的列表
            std::vector<float> merged_scalars;
            std::vector<uint32_t> merged_ids;
            merged_scalars.reserve(list.main_scalars.size() + to_insert.size());
            merged_ids.reserve(list.main_ids.size() + to_insert.size());
            
            size_t i = 0, j = 0;
            while (i < list.main_scalars.size() && j < to_insert.size()) {
                if (list.main_scalars[i] <= to_insert[j].first) {
                    merged_scalars.push_back(list.main_scalars[i]);
                    merged_ids.push_back(list.main_ids[i]);
                    i++;
                } else {
                    merged_scalars.push_back(to_insert[j].first);
                    merged_ids.push_back(to_insert[j].second);
                    j++;
                }
            }
            while (i < list.main_scalars.size()) {
                merged_scalars.push_back(list.main_scalars[i]);
                merged_ids.push_back(list.main_ids[i]);
                i++;
            }
            while (j < to_insert.size()) {
                merged_scalars.push_back(to_insert[j].first);
                merged_ids.push_back(to_insert[j].second);
                j++;
            }
            
            list.main_scalars = std::move(merged_scalars);
            list.main_ids = std::move(merged_ids);
        }
        if (keep_list_packed_payload_ && !vectors_flat_.empty()) {
            list.set_dimension(dimension_);
            list.rebuild_main_packed(dimension_, vectors_flat_.data(), n_vectors_, vector_norms_);
        } else if (!keep_list_packed_payload_) {
            list.release_packed_payload();
        }
    }
    
    // 优化：批量插入到其他聚类，而不是逐个插入
    for (const auto& kv : new_assignments) {
        const int j = kv.first;
        const std::vector<uint32_t>& assigned_ids = kv.second;
        if (j == cluster_id || assigned_ids.empty()) continue;
        
        RangeAwareList& other_list = ivf_lists_[j];
        other_list.set_dimension(dimension_);

        if (enable_scalar_filter_lsm_) {
            for (uint32_t vec_id : assigned_ids) {
                const float scalar = (vec_id < vector_scalars_.size()) ? vector_scalars_[vec_id] : 0.0f;
                const bool has_float =
                    !vectors_flat_.empty() && vec_id < n_vectors_ &&
                    (static_cast<size_t>(vec_id) + 1) * static_cast<size_t>(dimension_) <= vectors_flat_.size();
                if (has_float && keep_list_packed_payload_) {
                    const float* vptr = vectors_flat_.data() + static_cast<size_t>(vec_id) * static_cast<size_t>(dimension_);
                    const float norm_sq_full =
                        (vec_id < vector_norms_.size()) ? (vector_norms_[vec_id] * 2.0f) : -1.0f;
                    other_list.add_into_unordered(vec_id, scalar, vptr, norm_sq_full, dimension_);
                } else {
                    other_list.add_into_unordered(vec_id, scalar);
                }
            }
            lsm_trigger_merge_if_needed(other_list);
            cluster_stats_[j]->size = other_list.total_size();
            continue;
        }
        
        // 收集需要插入的向量和标量值
        std::vector<std::pair<float, uint32_t>> to_insert;
        to_insert.reserve(assigned_ids.size());
        for (uint32_t vec_id : assigned_ids) {
            float scalar = (vec_id < vector_scalars_.size()) ? 
                          vector_scalars_[vec_id] : 0.0f;
            to_insert.emplace_back(scalar, vec_id);
        }
        
        // 排序
        std::sort(to_insert.begin(), to_insert.end(),
                 [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                     return a.first < b.first;
                 });
        
        // 批量插入（归并）
        if (other_list.main_scalars.empty()) {
            other_list.main_scalars.reserve(to_insert.size());
            other_list.main_ids.reserve(to_insert.size());
            for (const auto& pair : to_insert) {
                other_list.main_scalars.push_back(pair.first);
                other_list.main_ids.push_back(pair.second);
            }
        } else {
            // 归并两个已排序的列表
            std::vector<float> merged_scalars;
            std::vector<uint32_t> merged_ids;
            merged_scalars.reserve(other_list.main_scalars.size() + to_insert.size());
            merged_ids.reserve(other_list.main_ids.size() + to_insert.size());
            
            size_t i = 0, k = 0;
            while (i < other_list.main_scalars.size() && k < to_insert.size()) {
                if (other_list.main_scalars[i] <= to_insert[k].first) {
                    merged_scalars.push_back(other_list.main_scalars[i]);
                    merged_ids.push_back(other_list.main_ids[i]);
                    i++;
                } else {
                    merged_scalars.push_back(to_insert[k].first);
                    merged_ids.push_back(to_insert[k].second);
                    k++;
                }
            }
            while (i < other_list.main_scalars.size()) {
                merged_scalars.push_back(other_list.main_scalars[i]);
                merged_ids.push_back(other_list.main_ids[i]);
                i++;
            }
            while (k < to_insert.size()) {
                merged_scalars.push_back(to_insert[k].first);
                merged_ids.push_back(to_insert[k].second);
                k++;
            }
            
            other_list.main_scalars = std::move(merged_scalars);
            other_list.main_ids = std::move(merged_ids);
        }
        if (keep_list_packed_payload_ && !vectors_flat_.empty()) {
            other_list.set_dimension(dimension_);
            other_list.rebuild_main_packed(dimension_, vectors_flat_.data(), n_vectors_, vector_norms_);
        } else if (!keep_list_packed_payload_) {
            other_list.release_packed_payload();
        }

        cluster_stats_[j]->size = other_list.total_size();
    }
    
    // 更新原聚类的统计信息
    cluster_stats_[cluster_id]->size = list.total_size();
    cluster_stats_[cluster_id]->insert_count_since_recluster = 0;
    cluster_stats_[cluster_id]->temperature = 1.0f;  // 重聚类后重置温度
    
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 更新聚类统计信息..." << std::endl);
    
    // 更新所有受影响聚类的增量 sum/count 与质心（包括原聚类和其他接收向量的聚类）。
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 更新受影响聚类的质心..." << std::endl);
    for (const auto& kv : new_assignments) {
        const int j = kv.first;
        if (kv.second.empty()) continue;
        if (j < 0 || j >= n_clusters_ || cluster_stats_[j]->is_deleted) continue;
        rebuild_cluster_sum_from_list(j);
    }
    
    // 重聚类会改写 centroids_；同步到 assign() 使用的 kmeans_，并刷新路由范数缓存。
    kmeans_->set_centroids(centroids_);
    rebuild_centroid_norms_sq();

    // initial_centroids_应该只在train()时设置，之后不再更新
    // 这样可以正确反映质心相对于训练时的漂移
    
    // 锁会在locks析构时自动释放
    auto recluster_end_time = std::chrono::high_resolution_clock::now();
    double recluster_total_time = std::chrono::duration<double>(recluster_end_time - recluster_start_time).count();
    ADA_IVF_MAINT_DEBUG(std::cerr << "[DEBUG] recluster_cluster: 聚类 " << cluster_id << " 重聚类完成，总耗时: " << recluster_total_time << " 秒" << std::endl);
}

std::vector<AdaIVFCore::MaintenanceCandidate> AdaIVFCore::identify_clusters_to_recluster() const {
    if (!enable_maintenance_) {
        return {};
    }
    std::vector<MaintenanceCandidate> candidates;
    const size_t dynamic_limit = get_dynamic_max_cluster_size();
    const size_t insert_trigger = get_insert_trigger_threshold();

    size_t total_size = 0;
    size_t active_clusters = 0;
    for (int i = 0; i < n_clusters_; ++i) {
        if (cluster_stats_[i]->is_deleted) continue;
        total_size += ivf_lists_[i].total_size();
        active_clusters++;
    }
    const float avg_size = active_clusters > 0
        ? static_cast<float>(total_size) / static_cast<float>(active_clusters)
        : 0.0f;

    for (int i = 0; i < n_clusters_; ++i) {
        const auto& stats = cluster_stats_[i];
        if (stats->is_deleted) continue;
        const size_t list_size = ivf_lists_[i].total_size();

        // 过载由 hot queue 做受控迁移；这里不再全簇重聚类。
        if (list_size > dynamic_limit) {
            continue;
        }

        float drift_ratio = 0.0f;
        if (list_size > config_.min_cluster_size_for_quality_check &&
            !initial_centroids_.empty() && i < static_cast<int>(initial_centroids_.size()) &&
            !initial_centroids_[i].empty() && !centroids_[i].empty()) {
            float drift_sq = compute_distance_sq(centroids_[i].data(), initial_centroids_[i].data());
            float initial_norm_sq = 0.0f;
            for (size_t d = 0; d < initial_centroids_[i].size(); ++d) {
                initial_norm_sq += initial_centroids_[i][d] * initial_centroids_[i][d];
            }
            if (initial_norm_sq > 1e-10f) {
                drift_ratio = std::sqrt(drift_sq) / std::sqrt(initial_norm_sq);
            }
        }

        const float insert_pressure = static_cast<float>(stats->insert_count_since_recluster) /
            static_cast<float>(std::max<size_t>(1, insert_trigger));
        const float size_pressure = (avg_size > 1.0f && list_size > static_cast<size_t>(avg_size))
            ? (static_cast<float>(list_size) / avg_size - 1.0f)
            : 0.0f;
        const float drift_pressure = drift_ratio / std::max(1e-6f, recluster_threshold_);

        // 重聚类必须同时满足“有足够新插入/漂移”与“簇规模可靠”；避免小簇被噪声反复维护。
        const float reliability = static_cast<float>(list_size) /
            (static_cast<float>(list_size) + 64.0f);
        const float priority = reliability * (
            0.45f * insert_pressure +
            0.30f * std::min(size_pressure, 2.0f) +
            0.25f * drift_pressure);

        const bool insert_triggered = insert_pressure >= 1.0f && priority >= 0.55f;
        const bool drift_triggered = drift_pressure >= 1.0f && priority >= 0.45f;
        if (insert_triggered || drift_triggered) {
            candidates.push_back({i,
                                  drift_triggered ? MaintenanceReason::CentroidDrift
                                                  : MaintenanceReason::InsertPressure,
                                  priority});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
             [](const MaintenanceCandidate& a, const MaintenanceCandidate& b) {
                 if (a.priority != b.priority) return a.priority > b.priority;
                 return a.cluster_id < b.cluster_id;
             });
    if (candidates.size() > config_.max_recluster_per_batch) {
        candidates.resize(config_.max_recluster_per_batch);
    }
    return candidates;
}


float AdaIVFCore::compute_cluster_quality(int cluster_id) const {
    if (!enable_maintenance_) return 1.0f;
    if (cluster_id < 0 || cluster_id >= n_clusters_) return 1.0f;
    
    const RangeAwareList& list = ivf_lists_[cluster_id];
    if (list.total_size() == 0) {
        return 1.0f;
    }
    
    // 计算平均距离到中心
    const auto& centroid = centroids_[cluster_id];
    if (centroid.empty()) return 1.0f;
    
    float total_dist = 0.0f;
    size_t count = 0;
    std::vector<float> buf;
    auto accum = [&](uint32_t vec_id) {
        if (vec_id >= n_vectors_) {
            return;
        }
        decode_vector_for_id(vec_id, buf);
        float dist = compute_distance_sq(buf.data(), centroid.data());
        total_dist += dist;
        count++;
    };
    for (uint32_t vec_id : list.main_ids) {
        accum(vec_id);
    }
    for (uint32_t vec_id : list.delta_ids) {
        accum(vec_id);
    }
    
    if (count == 0) return 1.0f;
    
    float avg_dist = total_dist / count;
    
    // 归一化质量分数（距离越小，质量越高）
    return 1.0f / (1.0f + avg_dist);
}

void AdaIVFCore::update_all_centroids() {
    if (!enable_maintenance_) return;
    if (!is_trained_ || n_vectors_ == 0) {
        return;
    }

    // 快路径：插入/小步迁移持续维护 cluster_sums_/cluster_counts_，这里仅刷新 dirty 质心。
    if (cluster_sums_.size() == static_cast<size_t>(n_clusters_) &&
        cluster_counts_.size() == static_cast<size_t>(n_clusters_)) {
        refresh_dirty_centroids_from_sums();
        return;
    }

    // 兜底：状态缺失时重建一次 sum/count，再同步质心。
    reset_incremental_centroid_state();
    for (int i = 0; i < n_clusters_; ++i) {
        if (cluster_stats_[i]->is_deleted) {
            continue;
        }
        rebuild_cluster_sum_from_list(i);
    }
    kmeans_->set_centroids(centroids_);
    rebuild_centroid_norms_sq();
}

AdaIVFCore::ClusterSizeStats AdaIVFCore::get_cluster_size_stats() const {
    ClusterSizeStats stats;
    stats.min_size = std::numeric_limits<size_t>::max();
    stats.max_size = 0;
    stats.avg_size = 0.0;
    
    size_t total_size = 0;
    for (int i = 0; i < n_clusters_; ++i) {
        // 跳过已删除的聚类
        if (cluster_stats_[i]->is_deleted) {
            continue;
        }
        
        size_t size = ivf_lists_[i].total_size();
        total_size += size;
        if (size < stats.min_size) stats.min_size = size;
        if (size > stats.max_size) stats.max_size = size;
    }
    
    stats.avg_size = (n_clusters_ > 0) ? (static_cast<double>(total_size) / n_clusters_) : 0.0;
    if (stats.min_size == std::numeric_limits<size_t>::max()) {
        stats.min_size = 0;
    }
    
    return stats;
}

std::vector<size_t> AdaIVFCore::get_all_cluster_sizes() const {
    std::vector<size_t> sizes;
    sizes.reserve(static_cast<size_t>(n_clusters_));
    
    for (int i = 0; i < n_clusters_; ++i) {
        // 跳过已删除的聚类
        if (cluster_stats_[i]->is_deleted) {
            sizes.push_back(0);
        } else {
            sizes.push_back(ivf_lists_[i].total_size());
        }
    }
    
    return sizes;
}

bool AdaIVFCore::cluster_may_match_range(int cluster_id, float range_min, float range_max) const {
    if (cluster_id < 0 || cluster_id >= n_clusters_) {
        return false;
    }
    if (!std::isfinite(range_min) || !std::isfinite(range_max) || range_min > range_max) {
        return true;
    }
    if (cluster_stats_[cluster_id]->is_deleted) {
        return false;
    }
    const size_t idx = static_cast<size_t>(cluster_id);
    if (cluster_scalar_synopses_.size() == static_cast<size_t>(n_clusters_) &&
        cluster_scalar_synopses_[idx].valid) {
        const ScalarSynopsis& syn = cluster_scalar_synopses_[idx];
        return !(range_max < syn.min_value || range_min > syn.max_value);
    }
    if (cluster_scalar_bounds_valid_.size() == static_cast<size_t>(n_clusters_) &&
        cluster_scalar_bounds_valid_[idx]) {
        return !(range_max < cluster_scalar_mins_[idx] || range_min > cluster_scalar_maxs_[idx]);
    }
    AdaIvfSharedLock<AdaIvfListMutex> list_lock(*list_locks_[cluster_id]);
    return ivf_lists_[cluster_id].scalar_may_overlap(range_min, range_max);
}

size_t AdaIVFCore::count_cluster_in_range(int cluster_id, float range_min, float range_max) const {
    if (cluster_id < 0 || cluster_id >= n_clusters_) {
        return 0;
    }
    if (!std::isfinite(range_min) || !std::isfinite(range_max) || range_min > range_max) {
        return 0;
    }
    if (cluster_stats_[cluster_id]->is_deleted) {
        return 0;
    }
    AdaIvfSharedLock<AdaIvfListMutex> list_lock(*list_locks_[cluster_id]);
    return ivf_lists_[cluster_id].count_in_range(range_min, range_max);
}

size_t AdaIVFCore::estimate_cluster_in_range_count(int cluster_id, float range_min, float range_max) const {
    return estimate_cluster_hit_count_from_synopsis(cluster_id, range_min, range_max);
}


float AdaIVFCore::estimate_cluster_selectivity(int cluster_id, float range_min, float range_max) const {
    if (cluster_id < 0 || cluster_id >= n_clusters_) {
        return 0.0f;
    }
    if (!std::isfinite(range_min) || !std::isfinite(range_max) || range_min > range_max) {
        return 1.0f;
    }
    if (cluster_stats_[cluster_id]->is_deleted) {
        return 0.0f;
    }
    if (cluster_scalar_synopses_.size() != static_cast<size_t>(n_clusters_)) {
        return 1.0f;
    }
    const ScalarSynopsis& syn = cluster_scalar_synopses_[static_cast<size_t>(cluster_id)];
    if (!syn.valid || syn.count == 0) {
        return 0.0f;
    }
    const size_t hit = estimate_cluster_hit_count_from_synopsis(cluster_id, range_min, range_max);
    return std::min(1.0f, static_cast<float>(hit) / static_cast<float>(syn.count));
}


float AdaIVFCore::estimate_cluster_scalar_utility(int cluster_id, float range_min, float range_max) const {
    if (cluster_id < 0 || cluster_id >= n_clusters_) {
        return 0.0f;
    }
    if (!std::isfinite(range_min) || !std::isfinite(range_max) || range_min > range_max) {
        return 1.0f;
    }
    if (cluster_stats_[cluster_id]->is_deleted) {
        return 0.0f;
    }
    if (cluster_scalar_synopses_.size() != static_cast<size_t>(n_clusters_)) {
        return 1.0f;
    }
    const ScalarSynopsis& syn = cluster_scalar_synopses_[static_cast<size_t>(cluster_id)];
    if (!syn.valid || syn.count == 0) {
        return 0.0f;
    }
    if (range_max < syn.min_value || range_min > syn.max_value) {
        return 0.0f;
    }

    const size_t hit = estimate_cluster_hit_count_from_synopsis(cluster_id, range_min, range_max);
    const float count_f = static_cast<float>(std::max<uint32_t>(1u, syn.count));
    const float sel = std::min(1.0f, static_cast<float>(hit) / count_f);
    const float mass_term = std::log1p(static_cast<float>(hit)) / std::log1p(count_f);
    const float density_term = std::sqrt(std::max(0.0f, sel));

    const float span = std::max(1e-6f, syn.max_value - syn.min_value);
    const float overlap_lo = std::max(range_min, syn.min_value);
    const float overlap_hi = std::min(range_max, syn.max_value);
    const float overlap_len = std::max(0.0f, overlap_hi - overlap_lo);
    const float range_len = std::max(1e-6f, range_max - range_min);
    const float overlap_ratio = std::min(1.0f, overlap_len / span);
    const float range_cover = std::min(1.0f, overlap_len / range_len);
    const float boundary_term = std::sqrt(std::max(0.0f, overlap_ratio * range_cover));

    float center_gap = 0.0f;
    if (syn.mean_value < range_min) {
        center_gap = range_min - syn.mean_value;
    } else if (syn.mean_value > range_max) {
        center_gap = syn.mean_value - range_max;
    }
    const float global_span = (scalar_synopsis_domain_valid_ && scalar_synopsis_domain_max_ > scalar_synopsis_domain_min_)
        ? (scalar_synopsis_domain_max_ - scalar_synopsis_domain_min_)
        : span;
    const float sigma = std::sqrt(std::max(0.0f, syn.variance_value)) + std::max(1e-6f, global_span * 1e-4f);
    const float z = center_gap / sigma;
    const float mean_affinity = 1.0f / (1.0f + z * z);
    const float concentration = 1.0f / (1.0f + sigma / std::max(1e-6f, global_span));
    const float mean_term = mean_affinity * concentration;

    const float confidence = count_f / (count_f + 32.0f);
    const float utility = confidence * (
        0.45f * mass_term +
        0.25f * density_term +
        0.20f * mean_term +
        0.10f * boundary_term);
    return std::max(0.0f, std::min(1.0f, utility));
}


float AdaIVFCore::compute_global_imbalance_indicator() const {
    if (!enable_maintenance_) {
        return 0.0f;
    }
    if (!is_trained_ || n_vectors_ == 0) {
        return 0.0f;
    }
    
    // 全局失衡指标（fine 层）：用于“全局重建安全网”的触发判断。
    //
    // 仅保留 **簇大小失衡**（cv_size = σ / avg_size），避免在插入路径上额外计算
    // 漂移类指标（ε、ε'）带来的额外开销与误触发风险。
    
    // 1. 计算聚类大小标准差 σ
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(n_clusters_);
    for (int i = 0; i < n_clusters_; ++i) {
        if (!cluster_stats_[i]->is_deleted) {
            // 冷启动由平均簇大小门槛屏蔽；成熟索引中的空簇仍属于真实失衡。
            cluster_sizes.push_back(ivf_lists_[i].total_size());
        }
    }
    
    if (cluster_sizes.empty()) return 0.0f;
    
    float avg_size = 0.0f;
    for (size_t s : cluster_sizes) {
        avg_size += static_cast<float>(s);
    }
    avg_size /= static_cast<float>(cluster_sizes.size());
    
    float variance = 0.0f;
    for (size_t s : cluster_sizes) {
        float diff = static_cast<float>(s) - avg_size;
        variance += diff * diff;
    }
    float sigma = std::sqrt(variance / static_cast<float>(cluster_sizes.size()));  // 聚类大小标准差
    const float cv_size = (avg_size > 1e-6f) ? (sigma / avg_size) : 0.0f;
    return std::min(cv_size, 1.0f);
}

AdaIVFCore::ImbalanceMetrics AdaIVFCore::get_imbalance_metrics() const {
    ImbalanceMetrics metrics;
    metrics.sigma = 0.0f;
    metrics.epsilon = 0.0f;
    metrics.epsilon_prime = 0.0f;
    metrics.G = 0.0f;
    
    if (!is_trained_ || n_vectors_ == 0) {
        return metrics;
    }
    if (!enable_maintenance_) {
        return metrics;
    }
    
    // 1. 计算聚类大小标准差 σ
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(n_clusters_);
    for (int i = 0; i < n_clusters_; ++i) {
        if (!cluster_stats_[i]->is_deleted) {
            cluster_sizes.push_back(ivf_lists_[i].total_size());
        }
    }
    
    if (cluster_sizes.empty()) return metrics;
    
    float avg_size = 0.0f;
    for (size_t s : cluster_sizes) {
        avg_size += static_cast<float>(s);
    }
    avg_size /= static_cast<float>(cluster_sizes.size());
    
    float variance = 0.0f;
    for (size_t s : cluster_sizes) {
        float diff = static_cast<float>(s) - avg_size;
        variance += diff * diff;
    }
    metrics.sigma = std::sqrt(variance / static_cast<float>(cluster_sizes.size()));
    
    // 2. 计算重建误差 ε（质心变化均值）
    size_t valid_clusters = 0;
    for (int i = 0; i < n_clusters_; ++i) {
        if (cluster_stats_[i]->is_deleted || initial_centroids_.empty() || 
            i >= static_cast<int>(initial_centroids_.size()) ||
            initial_centroids_[i].empty() || centroids_[i].empty()) {
            continue;
        }
        
        float drift_sq = compute_distance_sq(centroids_[i].data(), initial_centroids_[i].data());
        metrics.epsilon += std::sqrt(drift_sq);
        valid_clusters++;
    }
    if (valid_clusters > 0) {
        metrics.epsilon /= static_cast<float>(valid_clusters);
    }
    
    // 3. 计算漂移误差 ε'（质心变化标准差）
    if (valid_clusters > 1) {
        float variance_drift = 0.0f;
        for (int i = 0; i < n_clusters_; ++i) {
            if (cluster_stats_[i]->is_deleted || initial_centroids_.empty() || 
                i >= static_cast<int>(initial_centroids_.size()) ||
                initial_centroids_[i].empty() || centroids_[i].empty()) {
                continue;
            }
            
            float drift_sq = compute_distance_sq(centroids_[i].data(), initial_centroids_[i].data());
            float drift = std::sqrt(drift_sq);
            float diff = drift - metrics.epsilon;
            variance_drift += diff * diff;
        }
        metrics.epsilon_prime = std::sqrt(variance_drift / static_cast<float>(valid_clusters));
    }
    
    // 4. 全局失衡指标：G = (σ + ε + ε') / normalization
    // 与 compute_global_imbalance_indicator() 保持一致：归一化到可阈值化尺度
    const float cv_size = (avg_size > 1e-6f) ? (metrics.sigma / avg_size) : 0.0f;
    
    // 计算 initial centroid 的平均范数（与上面 epsilon 的有效簇集合一致）
    float avg_initial_norm = 0.0f;
    if (valid_clusters > 0) {
        for (int i = 0; i < n_clusters_; ++i) {
            if (cluster_stats_[i]->is_deleted || initial_centroids_.empty() ||
                i >= static_cast<int>(initial_centroids_.size()) ||
                initial_centroids_[i].empty() || centroids_[i].empty()) {
                continue;
            }
            float norm_sq = 0.0f;
            for (size_t d = 0; d < initial_centroids_[i].size(); ++d) {
                float v = initial_centroids_[i][d];
                norm_sq += v * v;
            }
            avg_initial_norm += std::sqrt(std::max(0.0f, norm_sq));
        }
        avg_initial_norm /= static_cast<float>(valid_clusters);
    }
    
    const float relative_epsilon = (avg_initial_norm > 1e-6f) ? (metrics.epsilon / avg_initial_norm) : 0.0f;
    const float relative_epsilon_prime = (metrics.epsilon > 1e-6f) ? (metrics.epsilon_prime / metrics.epsilon) : 0.0f;
    
    const float w_size = 0.4f;
    const float w_drift = 0.4f;
    const float w_variance = 0.2f;
    metrics.G = w_size * std::min(cv_size, 1.0f) +
                w_drift * std::min(relative_epsilon, 1.0f) +
                w_variance * std::min(relative_epsilon_prime, 1.0f);
    
    return metrics;
}

void AdaIVFCore::check_and_refresh_fine_centroids() {
    if (!enable_maintenance_) {
        return;
    }
    const size_t avg_size = n_clusters_ > 0 ? n_vectors_ / static_cast<size_t>(n_clusters_) : 0;
    if (avg_size < fine_refresh_min_avg_cluster_size_) return;

    const size_t required_interval = fine_refresh_base_interval_insertions_ *
                                     fine_refresh_backoff_multiplier_;
    if (inserted_since_fine_refresh_ < required_interval) return;

    const float G_before = compute_global_imbalance_indicator();
    last_maintenance_G_before_ = G_before;
    last_maintenance_G_after_ = G_before;
    if (G_before <= config_.tau_G) {
        fine_refresh_backoff_multiplier_ = 1;
        return;
    }

    inserted_since_fine_refresh_ = 0;
    fine_refresh_count_++;
    update_all_centroids();
    const size_t moved = run_global_boundary_repair(/*max_clusters=*/4,
                                                    /*max_vectors_per_cluster=*/256);
    migrated_vector_count_ += moved;
    const float G_after = compute_global_imbalance_indicator();
    last_maintenance_G_after_ = G_after;
    const bool effective = (G_before - G_after) >= fine_refresh_min_G_drop_;

    std::cerr << "[fine centroid refresh] G=" << G_before << " -> " << G_after
              << ", moved=" << moved << ", effective=" << (effective ? 1 : 0)
              << ", next_interval="
              << fine_refresh_base_interval_insertions_ *
                     (effective ? 1 : std::min(fine_refresh_backoff_max_,
                                                fine_refresh_backoff_multiplier_ * 2))
              << std::endl;

    if (effective) {
        fine_refresh_backoff_multiplier_ = 1;
        // 只有指标确实改善才清理局部维护状态。
        for (int i = 0; i < n_clusters_; ++i) {
            if (!cluster_stats_[i]->is_deleted) {
                cluster_stats_[i]->insert_count_since_recluster = 0;
            }
        }
    } else {
        fine_refresh_backoff_multiplier_ = std::min(
            fine_refresh_backoff_max_, fine_refresh_backoff_multiplier_ * 2);
    }
}

void AdaIVFCore::rebuild_centroid_norms_sq() {
    invalidate_centroid_neighbor_cache();
    centroid_norms_sq_.assign(static_cast<size_t>(n_clusters_), 0.0f);
    for (int i = 0; i < n_clusters_; ++i) {
        if (centroids_[i].empty() || static_cast<int>(centroids_[i].size()) != dimension_) {
            continue;
        }
        centroid_norms_sq_[static_cast<size_t>(i)] = IPSIMD4ExtAVX(
            const_cast<float*>(centroids_[i].data()),
            const_cast<float*>(centroids_[i].data()),
            static_cast<size_t>(dimension_));
    }
}

void AdaIVFCore::compute_cluster_scores(const float* query, 
                                        std::vector<float>& scores) const {
    // query 到全部 n_clusters 质心的 L2²；IPSIMD：||q-c||^2 = ||q||^2 + ||c||^2 - 2<q,c>
    scores.clear();
    scores.resize(static_cast<size_t>(n_clusters_));

    const float query_norm_sq = IPSIMD4ExtAVX(
        const_cast<float*>(query),
        const_cast<float*>(query),
        static_cast<size_t>(dimension_));

    for (int i = 0; i < n_clusters_; ++i) {
        if (cluster_stats_[i]->is_deleted || centroids_[i].empty()) {
            scores[static_cast<size_t>(i)] = std::numeric_limits<float>::max();
            continue;
        }

        const float ip = IPSIMD4ExtAVX(
            const_cast<float*>(query),
            const_cast<float*>(centroids_[i].data()),
            static_cast<size_t>(dimension_));
        const float norm_c_sq =
            (static_cast<size_t>(i) < centroid_norms_sq_.size())
                ? centroid_norms_sq_[static_cast<size_t>(i)]
                : IPSIMD4ExtAVX(
                      const_cast<float*>(centroids_[i].data()),
                      const_cast<float*>(centroids_[i].data()),
                      static_cast<size_t>(dimension_));
        scores[static_cast<size_t>(i)] = query_norm_sq + norm_c_sq - 2.0f * ip;
    }
}

void AdaIVFCore::select_top_k_clusters(const std::vector<float>& scores, int k,
                                      std::vector<int>& selected) const {
    ivf_topk::select_smallest_k_indices(scores, n_clusters_, k, selected);
}

void AdaIVFCore::write_cluster_dist_sq_for_ids(
    const float* query,
    const std::vector<int>& cluster_ids,
    std::vector<float>& dist_cache) const {
    if (cluster_ids.empty() || !is_trained_ || dimension_ <= 0) {
        return;
    }
    if (dist_cache.size() < static_cast<size_t>(n_clusters_)) {
        dist_cache.resize(static_cast<size_t>(n_clusters_), 0.0f);
    }

    const float query_norm_sq = IPSIMD4ExtAVX(
        const_cast<float*>(query),
        const_cast<float*>(query),
        static_cast<size_t>(dimension_));

    for (int cluster_id : cluster_ids) {
        if (cluster_id < 0 || cluster_id >= n_clusters_) {
            continue;
        }
        if (cluster_stats_[cluster_id]->is_deleted || centroids_[cluster_id].empty()) {
            dist_cache[static_cast<size_t>(cluster_id)] = std::numeric_limits<float>::max();
            continue;
        }

        const float ip = IPSIMD4ExtAVX(
            const_cast<float*>(query),
            const_cast<float*>(centroids_[cluster_id].data()),
            static_cast<size_t>(dimension_));
        const float norm_c_sq =
            (static_cast<size_t>(cluster_id) < centroid_norms_sq_.size())
                ? centroid_norms_sq_[static_cast<size_t>(cluster_id)]
                : IPSIMD4ExtAVX(
                      const_cast<float*>(centroids_[cluster_id].data()),
                      const_cast<float*>(centroids_[cluster_id].data()),
                      static_cast<size_t>(dimension_));
        dist_cache[static_cast<size_t>(cluster_id)] = query_norm_sq + norm_c_sq - 2.0f * ip;
    }
}

float AdaIVFCore::compute_distance_sq(const float* vec1, const float* vec2) const {
    float dist = 0.0f;
    for (int d = 0; d < dimension_; ++d) {
        float diff = vec1[d] - vec2[d];
        dist += diff * diff;
    }
    return dist;
}

void AdaIVFCore::reset_incremental_centroid_state() {
    cluster_sums_.assign(static_cast<size_t>(n_clusters_),
                         std::vector<float>(static_cast<size_t>(std::max(0, dimension_)), 0.0f));
    cluster_counts_.assign(static_cast<size_t>(n_clusters_), 0);
    cluster_scalar_sums_.assign(static_cast<size_t>(n_clusters_), 0.0);
    cluster_scalar_counts_.assign(static_cast<size_t>(n_clusters_), 0);
    centroid_dirty_.assign(static_cast<size_t>(n_clusters_), 0);
    hot_cluster_mark_.assign(static_cast<size_t>(n_clusters_), 0);
    hot_clusters_.clear();
}

void AdaIVFCore::add_vector_to_cluster_sum(int cluster_id, const float* vec) {
    if (cluster_id < 0 || cluster_id >= n_clusters_ || vec == nullptr || dimension_ <= 0) {
        return;
    }
    const size_t cid = static_cast<size_t>(cluster_id);
    if (cluster_sums_.size() != static_cast<size_t>(n_clusters_) ||
        cluster_sums_[cid].size() != static_cast<size_t>(dimension_)) {
        reset_incremental_centroid_state();
    }
    for (int d = 0; d < dimension_; ++d) {
        cluster_sums_[cid][static_cast<size_t>(d)] += vec[static_cast<size_t>(d)];
    }
    cluster_counts_[cid] += 1;
    centroid_dirty_[cid] = 1;
}

void AdaIVFCore::add_scalar_to_cluster_sum(int cluster_id, float scalar) {
    if (cluster_id < 0 || cluster_id >= n_clusters_) {
        return;
    }
    if (cluster_scalar_sums_.size() != static_cast<size_t>(n_clusters_) ||
        cluster_scalar_counts_.size() != static_cast<size_t>(n_clusters_)) {
        cluster_scalar_sums_.assign(static_cast<size_t>(n_clusters_), 0.0);
        cluster_scalar_counts_.assign(static_cast<size_t>(n_clusters_), 0);
    }
    const size_t cid = static_cast<size_t>(cluster_id);
    cluster_scalar_sums_[cid] += static_cast<double>(scalar);
    cluster_scalar_counts_[cid] += 1;
    if (kmeans_ && cluster_scalar_counts_[cid] > 0) {
        kmeans_->set_scalar_centroid(
            cluster_id,
            static_cast<float>(cluster_scalar_sums_[cid] / static_cast<double>(cluster_scalar_counts_[cid])));
    }
}

void AdaIVFCore::subtract_vector_from_cluster_sum(int cluster_id, const float* vec) {
    if (cluster_id < 0 || cluster_id >= n_clusters_ || vec == nullptr || dimension_ <= 0 ||
        cluster_sums_.size() != static_cast<size_t>(n_clusters_)) {
        return;
    }
    const size_t cid = static_cast<size_t>(cluster_id);
    if (cluster_sums_[cid].size() != static_cast<size_t>(dimension_)) {
        return;
    }
    for (int d = 0; d < dimension_; ++d) {
        cluster_sums_[cid][static_cast<size_t>(d)] -= vec[static_cast<size_t>(d)];
    }
    if (cluster_counts_[cid] > 0) {
        cluster_counts_[cid] -= 1;
    }
    centroid_dirty_[cid] = 1;
}

void AdaIVFCore::observe_cluster_scalar_bounds(int cluster_id, float scalar) {
    if (cluster_id < 0 || cluster_id >= n_clusters_) return;
    const size_t idx = static_cast<size_t>(cluster_id);
    if (cluster_scalar_bounds_valid_.size() != static_cast<size_t>(n_clusters_)) {
        cluster_scalar_mins_.assign(static_cast<size_t>(n_clusters_), 0.0f);
        cluster_scalar_maxs_.assign(static_cast<size_t>(n_clusters_), 0.0f);
        cluster_scalar_bounds_valid_.assign(static_cast<size_t>(n_clusters_), 0u);
    }
    if (!cluster_scalar_bounds_valid_[idx]) {
        cluster_scalar_mins_[idx] = scalar;
        cluster_scalar_maxs_[idx] = scalar;
        cluster_scalar_bounds_valid_[idx] = 1u;
    } else {
        cluster_scalar_mins_[idx] = std::min(cluster_scalar_mins_[idx], scalar);
        cluster_scalar_maxs_[idx] = std::max(cluster_scalar_maxs_[idx], scalar);
    }
}

void AdaIVFCore::observe_cluster_scalar_synopsis(int cluster_id, float scalar) {
    if (cluster_id < 0 || cluster_id >= n_clusters_) return;
    if (cluster_scalar_synopses_.size() != static_cast<size_t>(n_clusters_)) {
        cluster_scalar_synopses_.assign(static_cast<size_t>(n_clusters_), ScalarSynopsis{});
    }
    ScalarSynopsis& syn = cluster_scalar_synopses_[static_cast<size_t>(cluster_id)];
    if (!syn.valid) {
        syn.min_value = scalar;
        syn.max_value = scalar;
        syn.mean_value = scalar;
        syn.variance_value = 0.0f;
        syn.valid = true;
    } else {
        syn.min_value = std::min(syn.min_value, scalar);
        syn.max_value = std::max(syn.max_value, scalar);
    }
    const uint32_t old_count = syn.count;
    ++syn.count;
    if (old_count > 0) {
        const double old_mean = static_cast<double>(syn.mean_value);
        const double old_m2 = static_cast<double>(syn.variance_value) * static_cast<double>(old_count);
        const double delta = static_cast<double>(scalar) - old_mean;
        const double new_mean = old_mean + delta / static_cast<double>(syn.count);
        const double delta2 = static_cast<double>(scalar) - new_mean;
        const double new_m2 = old_m2 + delta * delta2;
        syn.mean_value = static_cast<float>(new_mean);
        syn.variance_value = static_cast<float>(std::max(0.0, new_m2 / static_cast<double>(syn.count)));
    }
    if (!scalar_synopsis_domain_valid_ || !(scalar_synopsis_domain_min_ < scalar_synopsis_domain_max_)) {
        return;
    }
    const float inv = static_cast<float>(ScalarSynopsis::kHistBins) /
                      (scalar_synopsis_domain_max_ - scalar_synopsis_domain_min_);
    int b = static_cast<int>((scalar - scalar_synopsis_domain_min_) * inv);
    if (b < 0) b = 0;
    if (b >= ScalarSynopsis::kHistBins) b = ScalarSynopsis::kHistBins - 1;
    uint16_t& slot = syn.histogram[b];
    if (slot < std::numeric_limits<uint16_t>::max()) {
        ++slot;
    }
}

size_t AdaIVFCore::estimate_cluster_hit_count_from_synopsis(
    int cluster_id, float range_min, float range_max) const {
    if (cluster_id < 0 || cluster_id >= n_clusters_) return 0;
    if (!std::isfinite(range_min) || !std::isfinite(range_max) || range_min > range_max) return 0;
    if (cluster_scalar_synopses_.size() != static_cast<size_t>(n_clusters_)) return 0;
    const ScalarSynopsis& syn = cluster_scalar_synopses_[static_cast<size_t>(cluster_id)];
    if (!syn.valid || syn.count == 0) return 0;
    if (range_max < syn.min_value || range_min > syn.max_value) return 0;
    if (!scalar_synopsis_domain_valid_ || !(scalar_synopsis_domain_min_ < scalar_synopsis_domain_max_)) {
        return syn.count;
    }
    if (range_min <= syn.min_value && range_max >= syn.max_value) {
        return syn.count;
    }

    const float width = (scalar_synopsis_domain_max_ - scalar_synopsis_domain_min_) /
                        static_cast<float>(ScalarSynopsis::kHistBins);
    if (!(width > 0.0f)) return syn.count;
    double hit = 0.0;
    uint32_t hist_total = 0;
    for (int b = 0; b < ScalarSynopsis::kHistBins; ++b) {
        const uint16_t c = syn.histogram[b];
        hist_total += c;
        if (c == 0) continue;
        const float blo = scalar_synopsis_domain_min_ + static_cast<float>(b) * width;
        const float bhi = blo + width;
        const float lo = std::max(range_min, blo);
        const float hi = std::min(range_max, bhi);
        if (hi > lo) {
            hit += static_cast<double>(c) * static_cast<double>((hi - lo) / width);
        }
    }
    if (hist_total == 0) {
        return syn.count;
    }
    const double scale = static_cast<double>(syn.count) / static_cast<double>(hist_total);
    const size_t est = static_cast<size_t>(std::ceil(hit * scale));
    return std::min<size_t>(syn.count, est);
}

void AdaIVFCore::rebuild_cluster_scalar_bounds_from_list(int cluster_id) {
    if (cluster_id < 0 || cluster_id >= n_clusters_) return;
    const size_t idx = static_cast<size_t>(cluster_id);
    if (cluster_scalar_bounds_valid_.size() != static_cast<size_t>(n_clusters_)) {
        cluster_scalar_mins_.assign(static_cast<size_t>(n_clusters_), 0.0f);
        cluster_scalar_maxs_.assign(static_cast<size_t>(n_clusters_), 0.0f);
        cluster_scalar_bounds_valid_.assign(static_cast<size_t>(n_clusters_), 0u);
    }
    cluster_scalar_bounds_valid_[idx] = 0u;
    if (cluster_scalar_synopses_.size() == static_cast<size_t>(n_clusters_)) {
        cluster_scalar_synopses_[idx] = ScalarSynopsis{};
    }
    const RangeAwareList& list = ivf_lists_[cluster_id];
    for (float scalar : list.main_scalars) {
        observe_cluster_scalar_bounds(cluster_id, scalar);
        observe_cluster_scalar_synopsis(cluster_id, scalar);
    }
    for (float scalar : list.delta_scalars) {
        observe_cluster_scalar_bounds(cluster_id, scalar);
        observe_cluster_scalar_synopsis(cluster_id, scalar);
    }
}

void AdaIVFCore::refresh_centroid_from_sum(int cluster_id) {
    if (cluster_id < 0 || cluster_id >= n_clusters_ || dimension_ <= 0 ||
        cluster_sums_.size() != static_cast<size_t>(n_clusters_)) {
        return;
    }
    const size_t cid = static_cast<size_t>(cluster_id);
    if (cluster_counts_[cid] == 0 || cluster_sums_[cid].size() != static_cast<size_t>(dimension_)) {
        return;
    }
    centroids_[cid].resize(static_cast<size_t>(dimension_));
    const float inv = 1.0f / static_cast<float>(cluster_counts_[cid]);
    for (int d = 0; d < dimension_; ++d) {
        centroids_[cid][static_cast<size_t>(d)] = cluster_sums_[cid][static_cast<size_t>(d)] * inv;
    }
    if (kmeans_ && cluster_scalar_counts_.size() == static_cast<size_t>(n_clusters_) &&
        cluster_scalar_counts_[cid] > 0) {
        kmeans_->set_scalar_centroid(
            cluster_id,
            static_cast<float>(cluster_scalar_sums_[cid] / static_cast<double>(cluster_scalar_counts_[cid])));
    }
    centroid_dirty_[cid] = 0;
}

void AdaIVFCore::refresh_dirty_centroids_from_sums() {
    if (cluster_sums_.size() != static_cast<size_t>(n_clusters_)) {
        return;
    }
    bool changed = false;
    for (int cid = 0; cid < n_clusters_; ++cid) {
        if (static_cast<size_t>(cid) < centroid_dirty_.size() && centroid_dirty_[static_cast<size_t>(cid)]) {
            refresh_centroid_from_sum(cid);
            changed = true;
        }
    }
    if (changed) {
        kmeans_->set_centroids(centroids_);
        rebuild_centroid_norms_sq();
    }
}

void AdaIVFCore::rebuild_cluster_sum_from_list(int cluster_id) {
    if (cluster_id < 0 || cluster_id >= n_clusters_ || dimension_ <= 0) {
        return;
    }
    if (cluster_sums_.size() != static_cast<size_t>(n_clusters_)) {
        reset_incremental_centroid_state();
    }
    const size_t cid = static_cast<size_t>(cluster_id);
    std::fill(cluster_sums_[cid].begin(), cluster_sums_[cid].end(), 0.0f);
    cluster_counts_[cid] = 0;
    if (cluster_scalar_sums_.size() != static_cast<size_t>(n_clusters_) ||
        cluster_scalar_counts_.size() != static_cast<size_t>(n_clusters_)) {
        cluster_scalar_sums_.assign(static_cast<size_t>(n_clusters_), 0.0);
        cluster_scalar_counts_.assign(static_cast<size_t>(n_clusters_), 0);
    }
    cluster_scalar_sums_[cid] = 0.0;
    cluster_scalar_counts_[cid] = 0;

    const RangeAwareList& list = ivf_lists_[cluster_id];
    std::vector<float> buf;
    auto accum = [&](uint32_t vec_id) {
        if (vec_id >= n_vectors_) {
            return;
        }
        decode_vector_for_id(vec_id, buf);
        for (int d = 0; d < dimension_; ++d) {
            cluster_sums_[cid][static_cast<size_t>(d)] += buf[static_cast<size_t>(d)];
        }
        if (vec_id < vector_scalars_.size()) {
            cluster_scalar_sums_[cid] += static_cast<double>(vector_scalars_[vec_id]);
            cluster_scalar_counts_[cid] += 1;
        }
        cluster_counts_[cid] += 1;
    };
    for (uint32_t vid : list.main_ids) accum(vid);
    for (uint32_t vid : list.delta_ids) accum(vid);
    centroid_dirty_[cid] = 1;
    refresh_centroid_from_sum(cluster_id);
}

void AdaIVFCore::mark_hot_cluster(int cluster_id) {
    if (cluster_id < 0 || cluster_id >= n_clusters_) {
        return;
    }
    if (hot_cluster_mark_.size() != static_cast<size_t>(n_clusters_)) {
        hot_cluster_mark_.assign(static_cast<size_t>(n_clusters_), 0);
        hot_clusters_.clear();
    }
    const size_t cid = static_cast<size_t>(cluster_id);
    if (!hot_cluster_mark_[cid]) {
        hot_cluster_mark_[cid] = 1;
        hot_clusters_.push_back(cluster_id);
    }
}

void AdaIVFCore::process_hot_clusters() {
    if (!enable_maintenance_ || hot_clusters_.empty()) {
        return;
    }
    const size_t target_size = get_dynamic_max_cluster_size();
    const size_t rounds = std::min(hot_cluster_budget_per_batch_, hot_clusters_.size());
    for (size_t r = 0; r < rounds; ++r) {
        const int cid = hot_clusters_.front();
        hot_clusters_.pop_front();
        if (cid < 0 || cid >= n_clusters_) {
            continue;
        }
        hot_cluster_mark_[static_cast<size_t>(cid)] = 0;
        migrated_vector_count_ += migrate_hot_cluster(cid, hot_vector_migration_budget_, target_size);
        const size_t current_size = ivf_lists_[cid].total_size();
        if (current_size > target_size) {
            mark_hot_cluster(cid);
        }
    }
}

size_t AdaIVFCore::migrate_hot_cluster(int cluster_id, size_t max_vectors_to_move,
                                       size_t target_size) {
    if (cluster_id < 0 || cluster_id >= n_clusters_ || max_vectors_to_move == 0 || dimension_ <= 0 ||
        cluster_stats_[cluster_id]->is_deleted || centroids_[cluster_id].empty()) {
        return 0;
    }
    if (target_size == 0) target_size = get_dynamic_max_cluster_size();

    std::vector<uint32_t> source_ids;
    size_t source_size = 0;
    {
        std::lock_guard<AdaIvfListMutex> lock(*list_locks_[cluster_id]);
        RangeAwareList& source = ivf_lists_[cluster_id];
        source.merge_unordered_into_main(keep_list_packed_payload_);
        source_size = source.total_size();
        if (source_size <= target_size || source.main_ids.empty()) {
            cluster_stats_[cluster_id]->size = source_size;
            return 0;
        }
        // 迁移预算很小时无需复制并扫描整个热桶；确定性等距抽样保持可复现。
        const size_t sample_cap = std::max<size_t>(4096, max_vectors_to_move * 8);
        const size_t sample_count = std::min(sample_cap, source.main_ids.size());
        source_ids.reserve(sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            const size_t pos = (i * source.main_ids.size()) / sample_count;
            source_ids.push_back(source.main_ids[pos]);
        }
    }

    const size_t overflow = source_size - target_size;
    const size_t take = std::min(max_vectors_to_move, std::max<size_t>(overflow, 1));
    if (take == 0) {
        return 0;
    }

    const int candidate_count = std::min(
        std::max(1, static_cast<int>(n_probe_ * config_.local_search_radius_factor)),
        n_clusters_);
    std::vector<int> candidate_clusters = get_cached_centroid_neighbors(cluster_id, candidate_count);
    if (candidate_clusters.empty()) {
        return 0;
    }

    std::vector<float> buf;
    std::vector<std::pair<float, uint32_t>> farthest;
    farthest.reserve(source_ids.size());
    for (uint32_t vid : source_ids) {
        if (vid >= n_vectors_) {
            continue;
        }
        decode_vector_for_id(vid, buf);
        const float dist = compute_distance_sq(buf.data(), centroids_[cluster_id].data());
        farthest.emplace_back(dist, vid);
    }
    if (farthest.empty()) {
        return 0;
    }
    const size_t selected_count = std::min(take, farthest.size());
    std::partial_sort(
        farthest.begin(), farthest.begin() + static_cast<std::ptrdiff_t>(selected_count), farthest.end(),
        [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) { return a.first > b.first; });
    farthest.resize(selected_count);

    std::vector<std::pair<uint32_t, int>> moves;
    moves.reserve(selected_count);
    for (const auto& item : farthest) {
        const uint32_t vid = item.second;
        if (vid >= n_vectors_) {
            continue;
        }
        decode_vector_for_id(vid, buf);
        float best = item.first;
        int best_cluster = cluster_id;
        for (int cand : candidate_clusters) {
            const float dist = compute_distance_sq(buf.data(), centroids_[cand].data());
            if (dist < best) {
                best = dist;
                best_cluster = cand;
            }
        }
        if (best_cluster != cluster_id) {
            moves.emplace_back(vid, best_cluster);
        }
    }
    if (moves.empty()) {
        return 0;
    }

    std::vector<int> clusters_to_lock;
    clusters_to_lock.push_back(cluster_id);
    for (const auto& mv : moves) {
        clusters_to_lock.push_back(mv.second);
    }
    std::sort(clusters_to_lock.begin(), clusters_to_lock.end());
    clusters_to_lock.erase(std::unique(clusters_to_lock.begin(), clusters_to_lock.end()), clusters_to_lock.end());
    std::vector<std::unique_lock<AdaIvfListMutex>> locks;
    locks.reserve(clusters_to_lock.size());
    for (int cid : clusters_to_lock) {
        locks.emplace_back(*list_locks_[cid]);
    }

    std::unordered_map<uint32_t, int> move_target;
    move_target.reserve(moves.size() * 2);
    for (const auto& mv : moves) {
        move_target[mv.first] = mv.second;
    }

    RangeAwareList& source = ivf_lists_[cluster_id];
    source.merge_unordered_into_main(keep_list_packed_payload_);
    std::vector<float> kept_scalars;
    std::vector<uint32_t> kept_ids;
    kept_scalars.reserve(source.main_scalars.size());
    kept_ids.reserve(source.main_ids.size());
    for (size_t i = 0; i < source.main_ids.size(); ++i) {
        const uint32_t vid = source.main_ids[i];
        if (move_target.find(vid) == move_target.end()) {
            kept_ids.push_back(vid);
            kept_scalars.push_back(i < source.main_scalars.size() ? source.main_scalars[i] : 0.0f);
        }
    }
    source.set_main_metadata(std::move(kept_scalars), std::move(kept_ids));
    if (keep_list_packed_payload_ && !vectors_flat_.empty()) {
        source.rebuild_main_packed(dimension_, vectors_flat_.data(), n_vectors_, vector_norms_);
    } else if (!keep_list_packed_payload_) {
        source.release_packed_payload();
    }
    rebuild_cluster_scalar_bounds_from_list(cluster_id);

    for (const auto& mv : moves) {
        const uint32_t vid = mv.first;
        const int dst = mv.second;
        if (dst < 0 || dst >= n_clusters_ || vid >= n_vectors_) {
            continue;
        }
        decode_vector_for_id(vid, buf);
        const float scalar = (vid < vector_scalars_.size()) ? vector_scalars_[vid] : 0.0f;
        subtract_vector_from_cluster_sum(cluster_id, buf.data());
        if (cluster_scalar_sums_.size() == static_cast<size_t>(n_clusters_) &&
            cluster_scalar_counts_.size() == static_cast<size_t>(n_clusters_)) {
            const size_t src_cid = static_cast<size_t>(cluster_id);
            cluster_scalar_sums_[src_cid] -= static_cast<double>(scalar);
            if (cluster_scalar_counts_[src_cid] > 0) {
                cluster_scalar_counts_[src_cid] -= 1;
            }
        }
        add_vector_to_cluster_sum(dst, buf.data());
        add_scalar_to_cluster_sum(dst, scalar);
        observe_cluster_scalar_bounds(dst, scalar);
        observe_cluster_scalar_synopsis(dst, scalar);

        RangeAwareList& dst_list = ivf_lists_[dst];
        dst_list.set_dimension(dimension_);
        const bool has_float =
            !vectors_flat_.empty() && (static_cast<size_t>(vid) + 1) * static_cast<size_t>(dimension_) <= vectors_flat_.size();
        if (enable_scalar_filter_lsm_) {
            if (has_float && keep_list_packed_payload_) {
                const float* vptr = vectors_flat_.data() + static_cast<size_t>(vid) * static_cast<size_t>(dimension_);
                const float norm_sq_full = (vid < vector_norms_.size()) ? (vector_norms_[vid] * 2.0f) : -1.0f;
                dst_list.add_into_unordered(vid, scalar, vptr, norm_sq_full, dimension_);
            } else {
                dst_list.add_into_unordered(vid, scalar);
            }
            lsm_trigger_merge_if_needed(dst_list);
        } else {
            auto it = std::lower_bound(dst_list.main_scalars.begin(), dst_list.main_scalars.end(), scalar);
            const size_t pos = static_cast<size_t>(std::distance(dst_list.main_scalars.begin(), it));
            if (has_float && keep_list_packed_payload_) {
                const float* vptr = vectors_flat_.data() + static_cast<size_t>(vid) * static_cast<size_t>(dimension_);
                const float norm_sq_full = (vid < vector_norms_.size()) ? (vector_norms_[vid] * 2.0f) : -1.0f;
                dst_list.insert_main_at(pos, vid, scalar, vptr, norm_sq_full);
            } else {
                dst_list.insert_main_metadata_at(pos, vid, scalar);
            }
        }
        cluster_stats_[dst]->size = dst_list.total_size();
    }

    cluster_stats_[cluster_id]->size = source.total_size();
    refresh_centroid_from_sum(cluster_id);
    for (const auto& mv : moves) {
        refresh_centroid_from_sum(mv.second);
    }
    kmeans_->set_centroids(centroids_);
    rebuild_centroid_norms_sq();
    return moves.size();
}

size_t AdaIVFCore::run_global_boundary_repair(size_t max_clusters,
                                               size_t max_vectors_per_cluster) {
    if (n_vectors_ == 0 || n_clusters_ <= 0 || max_clusters == 0) return 0;
    std::vector<std::pair<size_t, int>> ranked;
    size_t active = 0;
    size_t total = 0;
    for (int cid = 0; cid < n_clusters_; ++cid) {
        if (cluster_stats_[cid]->is_deleted) continue;
        const size_t size = ivf_lists_[cid].total_size();
        if (size == 0) continue;
        active++;
        total += size;
        ranked.emplace_back(size, cid);
    }
    if (active == 0) return 0;
    const size_t avg = std::max<size_t>(1, total / active);
    const size_t overloaded = std::max<size_t>(avg + 1, (avg * 3) / 2);
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<size_t, int>& a, const std::pair<size_t, int>& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });
    size_t moved = 0;
    size_t repaired = 0;
    for (const auto& entry : ranked) {
        if (repaired >= max_clusters || entry.first <= overloaded) break;
        moved += migrate_hot_cluster(entry.second, max_vectors_per_cluster, overloaded);
        repaired++;
    }
    return moved;
}

void AdaIVFCore::reserve_capacity(size_t estimated_size) {
    const bool keep_floats = (!enable_pq_compression_) || (!pq_release_floats_);
    if (keep_floats) {
        const size_t want = static_cast<size_t>(estimated_size) * static_cast<size_t>(std::max(1, dimension_));
        if (vectors_flat_.capacity() < want) {
            vectors_flat_.reserve(want);
        }
    }
    if (vector_scalars_.capacity() < estimated_size) {
        vector_scalars_.reserve(estimated_size);
        vector_norms_.reserve(estimated_size);
        if (enable_pq_compression_) {
            pq_codes_.reserve(estimated_size);
        }
    }
}

// ===========================================================================
// 删除支持（Tombstone Lazy Delete）
// ===========================================================================

void AdaIVFCore::remove(const int* ids, size_t n) {
    if (!ids || n == 0) return;
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] >= 0 && static_cast<size_t>(ids[i]) < n_vectors_) {
            tombstone_ids_.insert(static_cast<uint32_t>(ids[i]));
        }
    }
}

size_t AdaIVFCore::get_n_deleted() const {
    return tombstone_ids_.size();
}

size_t AdaIVFCore::get_n_live_vectors() const {
    const size_t unavailable = compacted_deleted_count_ + tombstone_ids_.size();
    return (n_vectors_ > unavailable) ? (n_vectors_ - unavailable) : 0;
}

size_t AdaIVFCore::compact_deleted_vectors_step(size_t max_clusters) {
    if (tombstone_ids_.empty() || n_clusters_ <= 0) return 0;

    const size_t budget = std::max<size_t>(1, max_clusters);
    size_t removed_total = 0;
    size_t scanned = 0;
    std::vector<uint32_t> removed_ids;

    while (scanned < budget && !tombstone_ids_.empty()) {
        const int cid = static_cast<int>(compact_cursor_cluster_ % static_cast<size_t>(n_clusters_));
        compact_cursor_cluster_ = (compact_cursor_cluster_ + 1) % static_cast<size_t>(n_clusters_);
        ++scanned;

        if (cluster_stats_[cid]->is_deleted) continue;

        std::lock_guard<AdaIvfListMutex> lock(*list_locks_[cid]);
        RangeAwareList& list = ivf_lists_[cid];

        bool has_deleted = false;
        for (uint32_t vid : list.main_ids) {
            if (tombstone_ids_.count(vid) != 0) { has_deleted = true; break; }
        }
        if (!has_deleted) {
            for (uint32_t vid : list.delta_ids) {
                if (tombstone_ids_.count(vid) != 0) { has_deleted = true; break; }
            }
        }
        if (!has_deleted) {
            if (!keep_list_packed_payload_) list.release_packed_payload();
            continue;
        }

        removed_ids.clear();

        const size_t mn = list.main_ids.size();
        const bool packed = keep_list_packed_payload_ && list.main_is_packed();
        const size_t dim_sz = static_cast<size_t>(dimension_);
        std::vector<uint32_t> new_mids;
        std::vector<float> new_ms;
        std::vector<float> new_mn_sq;
        std::vector<float> new_mv;
        new_mids.reserve(mn);
        new_ms.reserve(mn);
        if (packed) {
            new_mn_sq.reserve(mn);
            new_mv.reserve(mn * dim_sz);
        }
        for (size_t i = 0; i < mn; ++i) {
            const uint32_t vid = list.main_ids[i];
            if (tombstone_ids_.count(vid) != 0) {
                removed_ids.push_back(vid);
                continue;
            }
            new_mids.push_back(vid);
            new_ms.push_back(i < list.main_scalars.size() ? list.main_scalars[i] : 0.0f);
            if (packed) {
                new_mn_sq.push_back(list.main_norms_sq[i]);
                const float* src = list.main_vecs.data() + i * dim_sz;
                new_mv.insert(new_mv.end(), src, src + dim_sz);
            }
        }
        list.main_ids = std::move(new_mids);
        list.main_scalars = std::move(new_ms);
        if (packed) {
            list.main_norms_sq = std::move(new_mn_sq);
            list.main_vecs = std::move(new_mv);
        } else {
            list.main_norms_sq.clear();
            list.main_vecs.clear();
        }

        const size_t dn = list.delta_ids.size();
        const bool dpacked = keep_list_packed_payload_ && list.dimension_ > 0 &&
            list.delta_vecs.size() == dn * dim_sz && list.delta_norms_sq.size() == dn;
        std::vector<uint32_t> new_dids;
        std::vector<float> new_ds;
        std::vector<float> new_dn_sq;
        std::vector<float> new_dv;
        new_dids.reserve(dn);
        new_ds.reserve(dn);
        if (dpacked) {
            new_dn_sq.reserve(dn);
            new_dv.reserve(dn * dim_sz);
        }
        for (size_t i = 0; i < dn; ++i) {
            const uint32_t vid = list.delta_ids[i];
            if (tombstone_ids_.count(vid) != 0) {
                removed_ids.push_back(vid);
                continue;
            }
            new_dids.push_back(vid);
            new_ds.push_back(i < list.delta_scalars.size() ? list.delta_scalars[i] : 0.0f);
            if (dpacked) {
                new_dn_sq.push_back(list.delta_norms_sq[i]);
                const float* src = list.delta_vecs.data() + i * dim_sz;
                new_dv.insert(new_dv.end(), src, src + dim_sz);
            }
        }
        list.delta_ids = std::move(new_dids);
        list.delta_scalars = std::move(new_ds);
        if (dpacked) {
            list.delta_norms_sq = std::move(new_dn_sq);
            list.delta_vecs = std::move(new_dv);
        } else {
            list.delta_norms_sq.clear();
            list.delta_vecs.clear();
        }
        if (!keep_list_packed_payload_) list.release_packed_payload();
        list.rebuild_delta_stats_from_delta();
        list.rebuild_histogram_from_all();

        for (uint32_t vid : removed_ids) {
            tombstone_ids_.erase(vid);
        }
        removed_total += removed_ids.size();
        compacted_deleted_count_ += removed_ids.size();
        cluster_stats_[cid]->size = list.total_size();
        rebuild_cluster_scalar_bounds_from_list(cid);
    }

    return removed_total;
}

void AdaIVFCore::compact_deleted_vectors() {
    if (tombstone_ids_.empty()) return;
    const size_t budget = static_cast<size_t>(std::max(1, n_clusters_));
    size_t last_remaining = tombstone_ids_.size();
    while (!tombstone_ids_.empty()) {
        compact_deleted_vectors_step(budget);
        if (tombstone_ids_.size() == last_remaining) {
            tombstone_ids_.clear();
            break;
        }
        last_remaining = tombstone_ids_.size();
    }
}
