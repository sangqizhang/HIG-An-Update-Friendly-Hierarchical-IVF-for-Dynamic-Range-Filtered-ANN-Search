#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <deque>
#include <unordered_set>
#if __cplusplus >= 201402L
#include <shared_mutex>
#endif
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include "lsm_ivf_list.h"
#include "product_quantizer.h"

#if __cplusplus >= 201402L
using AdaIvfListMutex = std::shared_mutex;
template<typename M>
using AdaIvfSharedLock = std::shared_lock<M>;
#else
using AdaIvfListMutex = std::mutex;
template<typename M>
class AdaIvfSharedLock {
public:
    explicit AdaIvfSharedLock(M& m) : guard_(m) {}
private:
    std::lock_guard<M> guard_;
};
#endif

// 可见性宏：确保AdaIVFCore类可以被层次化模块访问
// 当使用-fvisibility=hidden编译时，需要显式标记符号为可见
#ifdef __GNUC__
    #define ADA_IVF_EXPORT __attribute__((visibility("default")))
#else
    #define ADA_IVF_EXPORT
#endif

// 调试宏：控制调试输出（发布版可禁用）
#ifndef ADA_IVF_DEBUG
#endif

#if ADA_IVF_DEBUG
#define ADA_IVF_DEBUG_OUT(x) std::cerr << x
#else
#define ADA_IVF_DEBUG_OUT(x) ((void)0)
#endif

/**
 * Ada-IVF核心实现（根据论文：Incremental IVF Index Maintenance for Streaming Vector Search）
 * 
 * 核心设计：
 * 1. IVF结构：每个聚类维护一个有序的向量列表（按scalar值排序）
 * 2. 训练阶段：K-means聚类，训练向量不插入索引（仅用于训练聚类中心）
 * 3. 增量插入：分配到最近聚类，直接插入有序列表（支持基于范围的混合检查）
 * 4. 搜索：选择top-n_probe聚类，范围过滤，距离计算（可选 PQ / ADC 近似距离）
 * 5. 自适应维护：对质量下降的聚类进行局部重聚类
 */

// 聚类统计信息
struct ClusterStats {
    int cluster_id;
    size_t size;
    size_t insert_count_since_recluster;
    float quality_score;
    float temperature;  // 温度：用于识别"热区"（频繁查询的聚类），初始值为1.0
    bool is_deleted;    // 标记：是否已被合并（用于merge机制）
    
    ClusterStats(int id) 
        : cluster_id(id), size(0), insert_count_since_recluster(0), quality_score(1.0f), temperature(1.0f), is_deleted(false) {}
};

// 简单的K-means聚类器
class SimpleKMeans {
private:
    std::vector<std::vector<float>> centroids_;
    std::vector<float> scalar_centroids_;
    int dimension_;
    float attr_lambda_{0.0f};
    float scalar_span_lambda_{0.0f};
    float scalar_mean_{0.0f};
    float scalar_scale_{1.0f};
    uint32_t random_seed_{42};
    bool use_kmeanspp_{true};
    std::vector<float> scalar_mins_;
    std::vector<float> scalar_maxs_;
    
public:
    SimpleKMeans() : dimension_(0) {}
    
    // 训练聚类中心
    void train(const float* data, size_t n_vectors, int dimension, int n_clusters,
               const float* scalars = nullptr);
    
    // 训练聚类中心（带最大迭代次数参数，用于优化coarse层重建）
    void train_with_max_iter(const float* data, size_t n_vectors, int dimension, int n_clusters,
                             int max_iterations, const float* scalars = nullptr);

    // 使用样本权重训练；coarse 层用 fine 簇实际向量数作为权重。
    void train_weighted_with_max_iter(const float* data, const float* weights,
                                      size_t n_vectors, int dimension, int n_clusters,
                                      int max_iterations);
    
    // 分配向量到最近的聚类
    uint32_t assign(const float* vector, float scalar = 0.0f) const;
    
    // 获取聚类中心
    const std::vector<std::vector<float>>& get_centroids() const { return centroids_; }
    const std::vector<float>& get_scalar_centroids() const { return scalar_centroids_; }
    const std::vector<float>& get_scalar_mins() const { return scalar_mins_; }
    const std::vector<float>& get_scalar_maxs() const { return scalar_maxs_; }
    
    // 设置聚类中心（用于更新）
    void set_centroids(const std::vector<std::vector<float>>& new_centroids) {
        centroids_ = new_centroids;
        if (!centroids_.empty()) {
            dimension_ = static_cast<int>(centroids_[0].size());
        }
    }

    void set_scalar_centroid(int cluster_id, float scalar_centroid) {
        if (cluster_id < 0) return;
        if (scalar_centroids_.size() < centroids_.size()) {
            scalar_centroids_.assign(centroids_.size(), 0.0f);
        }
        if (static_cast<size_t>(cluster_id) < scalar_centroids_.size()) {
            scalar_centroids_[static_cast<size_t>(cluster_id)] = scalar_centroid;
        }
    }

    void set_scalar_metadata(const std::vector<float>& scalar_centroids,
                             const std::vector<float>& scalar_mins,
                             const std::vector<float>& scalar_maxs) {
        scalar_centroids_ = scalar_centroids;
        scalar_mins_ = scalar_mins;
        scalar_maxs_ = scalar_maxs;
    }

    void set_scalar_normalization(float mean, float scale) {
        scalar_mean_ = mean;
        scalar_scale_ = (scale > 1e-6f) ? scale : 1.0f;
    }

    void set_attr_lambda(float lambda) { attr_lambda_ = lambda > 0.0f ? lambda : 0.0f; }
    float get_attr_lambda() const { return attr_lambda_; }
    void set_scalar_span_lambda(float lambda) { scalar_span_lambda_ = lambda > 0.0f ? lambda : 0.0f; }
    float get_scalar_span_lambda() const { return scalar_span_lambda_; }
    void set_random_seed(uint32_t seed) { random_seed_ = seed; }
    uint32_t get_random_seed() const { return random_seed_; }
    void set_use_kmeanspp(bool enabled) { use_kmeanspp_ = enabled; }
    bool get_use_kmeanspp() const { return use_kmeanspp_; }
    
    int get_dimension() const { return dimension_; }
};

// Ada-IVF配置参数结构体（提取硬编码参数）
struct AdaIVFConfig {
    float temperature_eta = 0.05f;
    float temperature_nu = 0.01f;
    float temperature_max = 100.0f;     // 温度上限：防止热区温度无限增长
    
    // 原：float alpha = 0.01f;  // ❌ 太小，温度影响极弱
    float alpha = 1.0f;                 // ✅ 温度缩放因子，使 f_T = T（即 α=1）
    float beta = 0.5f;                   // 失衡-漂移权重
    
    // ========== 容量层级（明确关系） ==========
    // 三层结构：合并阈值 < 目标容量 < 硬上限
    // 使用比例而不是硬编码，避免与max_cluster_size冲突
    float target_size_ratio = 0.5f;      // 目标容量 = max_cluster_size * 0.5
    float min_size_ratio = 0.25f;       // 合并阈值 = max_cluster_size * 0.25
    
    // 动态计算（不在配置中硬编码）：
    // tau_s = max_cluster_size * target_size_ratio
    // l_min = max_cluster_size * min_size_ratio
    
    // ========== 重聚类触发（放宽） ==========
    // 原：float insert_count_threshold_ratio = 0.5f;  // 太激进
    float insert_count_threshold_ratio = 0.9f;  // 更保守：更接近满容量才触发，提升插入吞吐
    
    // 原：int max_recluster_per_batch = 5;  // 太小
    size_t max_recluster_per_batch = 2;    // 更少的局部重聚类：显著降低插入阶段维护开销
    
    // ========== 全局安全网 ==========
    float tau_G = 0.95f;             // 全局重建阈值：更保守，减少触发频率
    
    // ========== 动态聚簇数（已禁用，保留配置项以备将来使用） ==========
    // int max_clusters = 16384;          // 最大聚类数量
    // float max_load_threshold = 0.8f;    // 最大负载阈值
    // int min_free_slots = 100;           // 最小空闲槽位数
    // bool allow_full_rebuild = true;     // 是否允许全量重建
    
    // ========== 性能调优 ==========
    size_t centroid_update_interval = 50000;   // 降低全量质心更新频率（插入更快）
    size_t perf_analysis_interval = 50000;     // 降低性能分析频率（插入更快）
    
    // ========== 新增：自适应参数（替代硬编码） ==========
    float early_termination_factor = 3.0f;      // 提前终止倍数（原k*3）
    float local_search_radius_factor = 2.0f;
    float global_search_count_factor = 2.0f;
    float reassignment_improvement_threshold = 0.3f;
    size_t min_cluster_size_for_quality_check = 100;  // 质量检查最小聚类大小（原硬编码100）
};

/**
 * Ada-IVF索引核心类
 * 使用ADA_IVF_EXPORT确保符号可见，以便层次化模块可以链接
 */
class ADA_IVF_EXPORT AdaIVFCore {
public:
    AdaIVFCore(int n_clusters, int n_probe, size_t max_cluster_size,
               float recluster_threshold);
    ~AdaIVFCore();
    
    // 训练索引（训练向量不插入索引，仅用于训练聚类中心）
    void train(const float* vectors, size_t n_vectors, int dimension, const float* scalars = nullptr);

    // 使用外部质心初始化索引（coarse-first / local refine 训练用），训练向量不插入索引。
    void train_with_centroids(const float* vectors, size_t n_vectors, int dimension,
                              const float* scalars,
                              const std::vector<std::vector<float>>& centroids,
                              const std::vector<float>& scalar_centroids = std::vector<float>(),
                              const std::vector<float>& scalar_mins = std::vector<float>(),
                              const std::vector<float>& scalar_maxs = std::vector<float>());
    
    // 增量插入（支持基于范围的混合检查）
    void add(const float* vectors, size_t n_vectors, const int* ids,
             bool auto_recluster, const float* scalars);
    
    // 搜索（支持范围过滤）
    std::vector<std::pair<float, int>> search(
        const float* query, int k, int max_num_distances,
        float range_min, float range_max) const;
    
    // 在指定的聚类列表中搜索（用于层次化搜索）
    // cluster_ids 应按质心距离升序；内部保序去重 + 预算内 vector 收集 + partial_sort(k)
    std::vector<std::pair<float, int>> search_in_clusters(
        const float* query, int k, int max_num_distances,
        const std::vector<int>& cluster_ids,
        float range_min, float range_max) const;

    /// 仅对 cluster_ids 写入 dist_cache[fid]=L2²(query, centroids_[fid])（IPSIMD，与 compute_cluster_scores 同口径）
    void write_cluster_dist_sq_for_ids(
        const float* query,
        const std::vector<int>& cluster_ids,
        std::vector<float>& dist_cache) const;
    
    // 批量查询：整批共享同一个 range。
    std::vector<std::vector<std::pair<float, int>>> batch_search(
        const float* queries, size_t n_queries, int k, int max_num_distances,
        float range_min, float range_max) const;

    // 批量查询：每条 query 使用自己的 [range_min, range_max]，保持 C++ OpenMP 批量并行。
    std::vector<std::vector<std::pair<float, int>>> batch_search_ranges(
        const float* queries, const float* ranges, size_t n_queries, int k, int max_num_distances) const;

    struct SearchStats {
        size_t num_dist_computations{0};
        // 兼容旧输出：等同 list_vectors_considered，不代表实际逐行扫描。
        size_t num_scanned_vectors{0};
        size_t list_vectors_considered{0};
        size_t scalar_rows_scanned{0};
        size_t scalar_blocks_skipped{0};
        size_t scalar_block_rows_skipped{0};
        size_t num_filtered_candidates{0};
        size_t num_visited_clusters{0};
        size_t num_pruned_clusters{0};
        size_t num_candidate_clusters{0};
        size_t graph_nodes_visited{0};
        size_t graph_scalar_positive_nodes{0};
        size_t graph_scalar_negative_nodes{0};
        size_t graph_fallback_expansions{0};

        void add(const SearchStats& other) {
            num_dist_computations += other.num_dist_computations;
            num_scanned_vectors += other.num_scanned_vectors;
            list_vectors_considered += other.list_vectors_considered;
            scalar_rows_scanned += other.scalar_rows_scanned;
            scalar_blocks_skipped += other.scalar_blocks_skipped;
            scalar_block_rows_skipped += other.scalar_block_rows_skipped;
            num_filtered_candidates += other.num_filtered_candidates;
            num_visited_clusters += other.num_visited_clusters;
            num_pruned_clusters += other.num_pruned_clusters;
            num_candidate_clusters += other.num_candidate_clusters;
            graph_nodes_visited += other.graph_nodes_visited;
            graph_scalar_positive_nodes += other.graph_scalar_positive_nodes;
            graph_scalar_negative_nodes += other.graph_scalar_negative_nodes;
            graph_fallback_expansions += other.graph_fallback_expansions;
        }
    };

    struct BatchSearchResult {
        std::vector<std::vector<std::pair<float, int>>> results;
        std::vector<SearchStats> per_query_stats;
    };

    struct ScalarSynopsis {
        static constexpr int kHistBins = 32;
        uint32_t count{0};
        float min_value{0.0f};
        float max_value{0.0f};
        float mean_value{0.0f};
        float variance_value{0.0f};
        uint16_t histogram[kHistBins]{};
        bool valid{false};
    };

    BatchSearchResult batch_search_with_stats(
        const float* queries, size_t n_queries, int k, int max_num_distances,
        float range_min, float range_max) const;

    BatchSearchResult batch_search_ranges_with_stats(
        const float* queries, const float* ranges, size_t n_queries, int k, int max_num_distances) const;

    std::vector<std::pair<float, int>> search_in_clusters_with_stats(
        const float* query, int k, int max_num_distances,
        const std::vector<int>& cluster_ids,
        float range_min, float range_max,
        SearchStats* stats) const;
    
    // 局部重聚类
    void recluster_cluster(int cluster_id);

    enum class MaintenanceReason { InsertPressure, CentroidDrift };
    struct MaintenanceCandidate {
        int cluster_id;
        MaintenanceReason reason;
        float priority;
    };
    std::vector<MaintenanceCandidate> identify_clusters_to_recluster() const;
    
    // 已移除：合并和分裂机制已禁用
    // std::vector<int> identify_clusters_to_merge() const;
    // void merge_cluster(int small_cluster_id);
    // void split_cluster(int cluster_id);
    // bool has_free_slot() const;
    // int count_free_slots() const;
    
    // 计算聚类质量
    float compute_cluster_quality(int cluster_id) const;
    
    // 更新所有聚类中心（基于当前聚类中的所有向量）
    void update_all_centroids();
    
    // 计算全局失衡指标（用于触发全局重建）
    float compute_global_imbalance_indicator() const;
    
    // 检查并执行 fine 质心刷新/受控边界修复；不是全量重建。
    void check_and_refresh_fine_centroids();
    
    size_t get_target_size() const { 
        return static_cast<size_t>(get_dynamic_max_cluster_size() * config_.target_size_ratio); 
    }
    
    size_t get_min_size() const { 
        return static_cast<size_t>(get_dynamic_max_cluster_size() * config_.min_size_ratio); 
    }
    
    size_t get_insert_trigger_threshold() const {
        return static_cast<size_t>(get_dynamic_max_cluster_size() * config_.insert_count_threshold_ratio);
    }
    
    // 根据当前向量总数和聚类数，动态计算目标聚类大小
    // 目标：每个聚类平均大小的3-5倍（推荐4倍）
    // 限制在合理范围（最小500，最大10000）
    size_t get_dynamic_max_cluster_size() const {
        if (n_vectors_ == 0 || n_clusters_ == 0) {
            return max_cluster_size_;  // 初始值
        }
        
        // 初期：严格限制，防止早期失衡
        if (n_vectors_ < 100000) {
            return 500;
        }
        
        // 计算平均聚类大小
        size_t avg_size = n_vectors_ / n_clusters_;
        
        // 动态调整：随着数据增长放宽限制
        size_t dynamic_target;
        if (avg_size < 100) {
            dynamic_target = 500;           // 极小数据：严格限制
        } else if (avg_size < 500) {
            dynamic_target = avg_size * 5;  // 小数据：5倍平均
        } else if (avg_size < 2000) {
            dynamic_target = avg_size * 4;  // 中数据：4倍平均（推荐）
        } else {
            dynamic_target = std::min(avg_size * 3, static_cast<size_t>(10000));  // 大数据：3倍，上限10K
        }
        
        // 限制在合理范围
        return std::max(static_cast<size_t>(500), std::min(dynamic_target, static_cast<size_t>(10000)));
    }
    
    // 获取统计信息
    size_t get_n_vectors() const { return n_vectors_; }
    int get_dimension() const { return dimension_; }
    int get_n_clusters() const { return n_clusters_; }
    int get_n_probe() const { return n_probe_; }
    bool is_trained() const { return is_trained_; }
    
    // 设置n_probe（用于静态基准测试中复用索引）
    void set_n_probe(int n_probe) { 
        if (n_probe <= 0 || n_probe > n_clusters_) {
            throw std::invalid_argument("n_probe must be between 1 and n_clusters");
        }
        n_probe_ = n_probe; 
    }
    
    // 获取质心（const 引用，避免 batch 路由等热路径按值拷贝）
    const std::vector<std::vector<float>>& get_centroids() const { return centroids_; }
    
    // 获取聚类大小统计
    struct ClusterSizeStats {
        size_t min_size;
        size_t max_size;
        double avg_size;
    };
    ClusterSizeStats get_cluster_size_stats() const;
    
    // 获取所有聚类的实际大小（用于层次化方法计算sigma_c）
    std::vector<size_t> get_all_cluster_sizes() const;
    
    // 轻量范围统计：用于层次化 filter-aware routing，不计算向量距离。
    bool cluster_may_match_range(int cluster_id, float range_min, float range_max) const;
    size_t count_cluster_in_range(int cluster_id, float range_min, float range_max) const;
    size_t estimate_cluster_in_range_count(int cluster_id, float range_min, float range_max) const;
    float estimate_cluster_selectivity(int cluster_id, float range_min, float range_max) const;
    float estimate_cluster_scalar_utility(int cluster_id, float range_min, float range_max) const;

    // 获取失衡指标（用于Python接口）
    struct ImbalanceMetrics {
        float sigma;          // 分区大小标准差
        float epsilon;        // 重建误差（质心变化均值）
        float epsilon_prime;  // 漂移误差（质心变化标准差）
        float G;              // 全局失衡指标
    };
    ImbalanceMetrics get_imbalance_metrics() const;

    size_t get_fine_refresh_count() const { return fine_refresh_count_; }
    size_t get_local_recluster_count() const { return local_recluster_count_; }
    size_t get_migrated_vector_count() const { return migrated_vector_count_; }
    float get_last_maintenance_G_before() const { return last_maintenance_G_before_; }
    float get_last_maintenance_G_after() const { return last_maintenance_G_after_; }

    // =========================================================================
    // Baseline 模式开关：用于构造“纯 IVF”基线（不启用 Ada 的自适应维护机制）
    // =========================================================================
    // enable_maintenance=false 时：
    // - add() 不触发重聚类（包括超过容量的强制重聚类）
    // - 不触发质心周期更新
    // - 不触发全局重建安全网
    void set_enable_maintenance(bool enabled) { enable_maintenance_ = enabled; }
    bool get_enable_maintenance() const { return enable_maintenance_; }

    // =========================================================================
    // Module A 开关（标量 LSM）
    // =========================================================================
    /// Module A：每个 IVF 桶内 main（有序）+ delta（无序），delta 达阈值后归并到 main
    void set_enable_scalar_filter_lsm(bool enabled) { enable_scalar_filter_lsm_ = enabled; }
    bool get_enable_scalar_filter_lsm() const { return enable_scalar_filter_lsm_; }

    /// 簇级 scalar min/max 粗剪枝：range 与 list 标量跨度不相交时跳过整个 list。
    void set_enable_scalar_range_prune(bool enabled) { enable_scalar_range_prune_ = enabled; }
    bool get_enable_scalar_range_prune() const { return enable_scalar_range_prune_; }

    /// 属性感知聚类：D = L2^2 + attr_lambda * normalized_attr_distance^2。
    void set_attr_lambda(float lambda) {
        attr_lambda_ = lambda > 0.0f ? lambda : 0.0f;
        if (kmeans_) kmeans_->set_attr_lambda(attr_lambda_);
    }
    float get_attr_lambda() const { return attr_lambda_; }
    void set_scalar_span_lambda(float lambda) {
        scalar_span_lambda_ = lambda > 0.0f ? lambda : 0.0f;
        if (kmeans_) kmeans_->set_scalar_span_lambda(scalar_span_lambda_);
    }
    float get_scalar_span_lambda() const { return scalar_span_lambda_; }
    void set_kmeans_seed(uint32_t seed) {
        kmeans_seed_ = seed;
        if (kmeans_) kmeans_->set_random_seed(seed);
    }
    uint32_t get_kmeans_seed() const { return kmeans_seed_; }
    void set_use_kmeanspp(bool enabled) {
        use_kmeanspp_ = enabled;
        if (kmeans_) kmeans_->set_use_kmeanspp(enabled);
    }
    bool get_use_kmeanspp() const { return use_kmeanspp_; }

    /// delta 归并阈值上限 t（默认 256；实际阈值结合平均桶大小和 main_size 自适应缩小）
    void set_lsm_merge_threshold(size_t t) { lsm_merge_threshold_ = (t == 0 ? 1 : t); }
    size_t get_lsm_merge_threshold() const { return lsm_merge_threshold_; }

    /// 将所有簇的 delta 并入 main（建议在「保存快照 / 导出」前调用）
    void flush_all_lsm_segments();

    /// 诊断：在 IVF 桶（main + delta）中查找 vec_id 所在的簇；不存在返回 -1
    /// vec_id 为内部连续 id（与按序 add 的存储下标一致）
    int find_cluster_for_vec_id(uint32_t vec_id) const;

    // =========================================================================
    // 删除支持（Tombstone Lazy Delete）
    // =========================================================================
    /// 将 ids（内部 vec_id）标记为已删除（查询时跳过）。
    /// ids 为内部连续 vec_id（与插入顺序对应的 0..n_vectors_-1）。
    void remove(const int* ids, size_t n);

    /// 返回当前已删除向量数量
    size_t get_n_deleted() const;

    /// 返回当前活跃向量数量
    size_t get_n_live_vectors() const;

    /// 从所有 IVF list 物理移除 tombstone 向量（降内存；删除大量向量后调用）
    void compact_deleted_vectors();
    size_t compact_deleted_vectors_step(size_t max_clusters);

    void set_keep_list_packed_payload(bool keep) { keep_list_packed_payload_ = keep; }
    bool get_keep_list_packed_payload() const { return keep_list_packed_payload_; }

    // =========================================================================
    // 直方图剪枝（Histogram Pruning）
    // =========================================================================
    void set_enable_histogram_prune(bool enabled) { enable_histogram_prune_ = enabled; }
    bool get_enable_histogram_prune() const { return enable_histogram_prune_; }

    /// 当估计选择率 < threshold 时跳过该簇（默认 0.005 = 0.5%）
    void set_histogram_prune_threshold(float t) { histogram_prune_threshold_ = (t < 0.0f ? 0.0f : t); }
    float get_histogram_prune_threshold() const { return histogram_prune_threshold_; }

    // =========================================================================
    // 早停级联探测（Cascade Early Exit）
    // =========================================================================
    void set_enable_cascade_early_exit(bool enabled) { enable_cascade_early_exit_ = enabled; }
    bool get_enable_cascade_early_exit() const { return enable_cascade_early_exit_; }
    /// 松弛因子 α：当 heap.worst_dist < α × next_centroid_dist² 时停止探测（默认 1.0）
    void set_cascade_early_exit_alpha(float alpha) { cascade_early_exit_alpha_ = (alpha > 0.0f ? alpha : 1.0f); }
    float get_cascade_early_exit_alpha() const { return cascade_early_exit_alpha_; }

    // =========================================================================
    // Module PQ：乘积量化压缩向量存储，检索用 ADC 近似平方 L2
    // =========================================================================
    /// 在 train 前打开：会在 train() 末尾用训练集拟合码本；插入后只保留 m 字节/向量的 PQ 码 + 标量等，释放原始 float
    void set_enable_pq_compression(bool enabled) { enable_pq_compression_ = enabled; }
    bool get_enable_pq_compression() const { return enable_pq_compression_; }

    /// 段数 m（维度必须整除 m）；默认 8
    void set_pq_m(int m) { pq_m_ = (m < 1 ? 8 : m); }
    int get_pq_m() const { return pq_m_; }

    /// 每段码字数 ksub（2–256）；默认 256
    void set_pq_ksub(int ksub) { pq_ksub_ = (ksub < 2 ? 2 : (ksub > 256 ? 256 : ksub)); }
    int get_pq_ksub() const { return pq_ksub_; }

    /// PQ 后是否释放原始 float 向量存储（默认 true：省内存但召回可能下降；置 false 可用全精度距离精排）
    void set_pq_release_floats(bool release) { pq_release_floats_ = release; }
    bool get_pq_release_floats() const { return pq_release_floats_; }

    /// 额外用一批向量训练 PQ（维数须与索引一致；适用于 train 后才打开 PQ、或需二次拟合码本）
    void train_pq_from_samples(const float* data, size_t n_vectors, int dimension);

    bool is_pq_trained() const { return pq_.is_trained(); }

private:
    // 基本参数
    int n_clusters_;
    int n_probe_;
    size_t max_cluster_size_;
    float recluster_threshold_;
    int dimension_;
    size_t n_vectors_;
    bool is_trained_;
    
    // 配置参数
    AdaIVFConfig config_;
    
    // 索引结构
    std::vector<std::vector<float>> centroids_;      // 聚类中心
    std::vector<std::vector<float>> initial_centroids_;  // 初始质心（用于计算漂移）
    std::vector<RangeAwareList> ivf_lists_;         // IVF lists（按scalar排序）
    std::vector<std::unique_ptr<ClusterStats>> cluster_stats_;  // 聚类统计
    
    // 向量存储（连续数组，减少 1M 级小 vector 分配与碎片）
    // - 当未启用 PQ 或 pq_release_floats_=false 时：保存全精度 float 向量，长度为 n_vectors_ * dimension_
    // - 当启用 PQ 且 pq_release_floats_=true 时：不保存 float（vectors_flat_ 为空/不增长），检索使用 PQ+ADC
    std::vector<float> vectors_flat_;
    std::vector<float> vector_scalars_;               // 标量值
    std::vector<float> vector_norms_;                 // 预计算的范数（全精度路径用于 L2SIMD；PQ+ADC 路径可不依赖）
    std::vector<float> centroid_norms_sq_;            // 质心 ||c||^2，路由打分 AVX 复用
    std::vector<double> cluster_scalar_sums_;
    std::vector<size_t> cluster_scalar_counts_;
    std::vector<float> cluster_scalar_mins_;
    std::vector<float> cluster_scalar_maxs_;
    std::vector<uint8_t> cluster_scalar_bounds_valid_;
    std::vector<ScalarSynopsis> cluster_scalar_synopses_;
    float scalar_synopsis_domain_min_{0.0f};
    float scalar_synopsis_domain_max_{1.0f};
    bool scalar_synopsis_domain_valid_{false};

    // 粗维护：增量维护每个簇的向量和与计数，避免周期性全量扫 list 更新质心。
    std::vector<std::vector<float>> cluster_sums_;
    std::vector<size_t> cluster_counts_;
    std::vector<uint8_t> centroid_dirty_;

    /// Module PQ：与 vec_id 对齐的 PQ 码字；仅 enable 且 train 后第 i 条非空
    ProductQuantizer pq_;
    std::vector<std::vector<uint8_t>> pq_codes_;
    bool enable_pq_compression_{false};
    int pq_m_{8};
    int pq_ksub_{256};
    bool pq_release_floats_{true};
    
    // K-means聚类器
    std::unique_ptr<SimpleKMeans> kmeans_;
    
    // 并发控制
    mutable std::vector<std::unique_ptr<AdaIvfListMutex>> list_locks_;
    // 注意：使用mutex代替shared_mutex（C++11兼容）
    mutable std::mutex centroids_mutex_;
    
    std::atomic<bool> is_rebuilding_{false};  // 重建标志（用于查询保护）
    mutable std::mutex rebuild_mutex_;  // 重建期间的互斥锁
    std::vector<std::vector<float>> centroids_snapshot_;  // 重建期间的快照
    float last_G_value_{0.0f};  // 上次G值（用于诊断）
    
    // 内部方法
    void compute_cluster_scores(const float* query, std::vector<float>& scores) const;
    void select_top_k_clusters(const std::vector<float>& scores, int k, 
                               std::vector<int>& selected) const;
    float compute_distance_sq(const float* vec1, const float* vec2) const;
    void rebuild_centroid_norms_sq();
    
    // 预分配容量
    void reserve_capacity(size_t estimated_size);

    // 粗维护 helper：增量质心与 hot 簇小步迁移。
    void reset_incremental_centroid_state();
    void add_vector_to_cluster_sum(int cluster_id, const float* vec);
    void subtract_vector_from_cluster_sum(int cluster_id, const float* vec);
    void add_scalar_to_cluster_sum(int cluster_id, float scalar);
    void observe_cluster_scalar_bounds(int cluster_id, float scalar);
    void observe_cluster_scalar_synopsis(int cluster_id, float scalar);
    size_t estimate_cluster_hit_count_from_synopsis(int cluster_id, float range_min, float range_max) const;
    void rebuild_cluster_scalar_bounds_from_list(int cluster_id);
    void refresh_centroid_from_sum(int cluster_id);
    void refresh_dirty_centroids_from_sums();
    void rebuild_cluster_sum_from_list(int cluster_id);
    void invalidate_centroid_neighbor_cache();
    std::vector<int> get_cached_centroid_neighbors(int cluster_id, int count);
    void mark_hot_cluster(int cluster_id);
    void process_hot_clusters();
    size_t migrate_hot_cluster(int cluster_id, size_t max_vectors_to_move,
                               size_t target_size = 0);
    size_t run_global_boundary_repair(size_t max_clusters, size_t max_vectors_per_cluster);

    // Baseline 模式开关（默认关维护，纯 IVF benchmark）
    bool enable_maintenance_{false};

    // fine 维护节流与可观测状态。所有间隔均按插入量计算，避免墙钟时间导致实验不可复现。
    size_t fine_refresh_min_avg_cluster_size_{64};
    size_t fine_refresh_base_interval_insertions_{200000};
    size_t inserted_since_fine_refresh_{0};
    size_t fine_refresh_backoff_multiplier_{1};
    size_t fine_refresh_backoff_max_{8};
    float fine_refresh_min_G_drop_{0.02f};
    size_t fine_refresh_count_{0};
    size_t local_recluster_count_{0};
    size_t migrated_vector_count_{0};
    std::vector<std::vector<int>> centroid_neighbor_cache_;
    int centroid_neighbor_cache_k_{0};
    float last_maintenance_G_before_{0.0f};
    float last_maintenance_G_after_{0.0f};

    // ========== Module A：标量 LSM（有序/无序段）==========
    bool enable_scalar_filter_lsm_{false};
    bool enable_scalar_range_prune_{false};
    size_t lsm_merge_threshold_{256};
    float attr_lambda_{0.0f};
    float scalar_span_lambda_{0.0f};
    uint32_t kmeans_seed_{42};
    bool use_kmeanspp_{true};

    // ========== 删除支持（Tombstone）==========
    mutable std::unordered_set<uint32_t> tombstone_ids_;
    size_t compacted_deleted_count_{0};
    size_t compact_cursor_cluster_{0};
    bool keep_list_packed_payload_{true};

    // ========== 直方图剪枝 ==========
    bool enable_histogram_prune_{false};
    float histogram_prune_threshold_{0.005f};

    // ========== 级联探测早停 ==========
    bool enable_cascade_early_exit_{false};
    float cascade_early_exit_alpha_{1.0f};

    // 维护计数必须按实例保存；不能用 add() 内 static，否则多个索引会互相污染。
    size_t insert_count_since_centroid_update_{0};

    // 粗维护：超限簇先入队，每批只做小步迁移，降低插入尾延迟。
    std::deque<int> hot_clusters_;
    std::vector<uint8_t> hot_cluster_mark_;
    size_t hot_cluster_budget_per_batch_{2};
    size_t hot_vector_migration_budget_{512};

    // ---------- 内部：Module A 子步骤 ----------
    /// 函数2：检测无序段是否达到阈值并触发合并（在持锁状态下调用）
    void lsm_trigger_merge_if_needed(RangeAwareList& list);

    /// 在 cluster_ids 子集内扫描并返回 top-k（保序逐簇 gather → L2 + max-heap，至多 max_num_distances 次 L2）
    std::vector<std::pair<float, int>> scan_cluster_list_for_topk(
        const float* query, int k, int max_num_distances,
        const std::vector<int>& cluster_ids,
        float range_min, float range_max,
        SearchStats* stats = nullptr) const;

    /// 从单簇收集候选 vec_id（main 段 binary_search_range；delta 段 SIMD 批量标量过滤）
    void gather_cluster_candidate_ids(int cluster_id, bool use_range_filter,
                                      float range_min, float range_max,
                                      std::vector<uint32_t>& out) const;

    /// 是否对 vec_id 使用 PQ 距离（adc_table 须非空）
    bool vector_uses_pq(uint32_t vec_id) const;

    /// 将 vec_id 解码到 buf（全精度槽有数据则拷贝）
    void decode_vector_for_id(uint32_t vec_id, std::vector<float>& buf) const;

};
