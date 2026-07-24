// lsm_ivf_list.h
// IVF 桶内列表：Module A 为「有序段 main + 无序段 delta（LSM 风格）」；
// main 段按标量升序，并 co-locate main_vecs / main_norms_sq（方案 A：batch IPSIMD 扫描）。
//
#pragma once

#include <vector>
#include <mutex>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <cstring>

// RangeAwareList：每个 IVF list 维护
// - main_* : 按标量值全局有序（支持二分范围查询 + 连续向量块 batch L2）
// - delta_*: 无序追加（insert 高频时避免每次二分挪动整块内存）
struct RangeAwareList {
    int dimension_{0};

    // -------- 有序段（按 main_scalars 升序）--------
    std::vector<uint32_t> main_ids;
    std::vector<float> main_scalars;
    /// 与 main_ids 同序：||x||^2（全量，非 L2SIMD 半范数）
    std::vector<float> main_norms_sq;
    /// row-major: main_vecs[i*d + d] = 第 i 条 main 向量
    std::vector<float> main_vecs;

    // -------- 无序段（append-only）--------
    std::vector<uint32_t> delta_ids;
    std::vector<float> delta_scalars;
    std::vector<float> delta_norms_sq;
    std::vector<float> delta_vecs;
    float delta_min_{0.0f};
    float delta_max_{0.0f};
    bool delta_stats_valid_{false};
    static constexpr size_t kDeltaBlockSize = 64;
    std::vector<float> delta_block_min_;
    std::vector<float> delta_block_max_;

    RangeAwareList() = default;

    RangeAwareList(RangeAwareList&& other) noexcept { move_from(std::move(other)); }

    RangeAwareList& operator=(RangeAwareList&& other) noexcept {
        if (this != &other) {
            move_from(std::move(other));
        }
        return *this;
    }

    RangeAwareList(const RangeAwareList&) = delete;
    RangeAwareList& operator=(const RangeAwareList&) = delete;

    void set_dimension(int dim) {
        if (dim > 0) {
            dimension_ = dim;
        }
    }

    bool main_is_packed() const {
        return dimension_ > 0 &&
               main_ids.size() == main_scalars.size() &&
               main_ids.size() == main_norms_sq.size() &&
               main_vecs.size() == main_ids.size() * static_cast<size_t>(dimension_);
    }

    /// 预分配有序段/无序段容量
    void reserve_main(size_t capacity) {
        main_ids.reserve(capacity);
        main_scalars.reserve(capacity);
        main_norms_sq.reserve(capacity);
        if (dimension_ > 0) {
            main_vecs.reserve(capacity * static_cast<size_t>(dimension_));
        }
        const size_t dcap = std::min(capacity / 4 + 64, static_cast<size_t>(4096));
        delta_ids.reserve(dcap);
        delta_scalars.reserve(dcap);
        delta_norms_sq.reserve(dcap);
        if (dimension_ > 0) {
            delta_vecs.reserve(dcap * static_cast<size_t>(dimension_));
        }
    }

    size_t main_size() const { return main_ids.size(); }
    size_t delta_size() const { return delta_ids.size(); }
    size_t total_size() const { return main_ids.size() + delta_ids.size(); }

    bool scalar_may_overlap(float L, float R) const {
        if (total_size() == 0 || L > R) {
            return false;
        }
        if (!main_scalars.empty()) {
            if (!(R < main_scalars.front() || L > main_scalars.back())) {
                return true;
            }
        }
        if (delta_stats_valid_) {
            return !(R < delta_min_ || L > delta_max_);
        }
        return false;
    }

    void set_histogram_domain(float domain_min, float domain_max) {
        if (!(domain_min < domain_max)) {
            return;
        }
        hist_min_ = domain_min;
        hist_max_ = domain_max;
        hist_valid_ = true;
        rebuild_histogram_from_all();
    }

    // -------------------------------------------------------------------------
    // Module A — 函数1：add_into_unorder() — 追加到无序段（可选 co-locate 向量）
    // -------------------------------------------------------------------------
    void add_into_unordered(uint32_t id, float scalar) {
        delta_ids.push_back(id);
        delta_scalars.push_back(scalar);
        observe_delta_scalar(scalar);
        observe_histogram_scalar(scalar);
    }

    void add_into_unordered(uint32_t id, float scalar, const float* vec, float norm_sq_full, int dim) {
        delta_ids.push_back(id);
        delta_scalars.push_back(scalar);
        observe_delta_scalar(scalar);
        observe_histogram_scalar(scalar);
        if (vec == nullptr || dim <= 0 || norm_sq_full < 0.0f) {
            return;
        }
        if (dimension_ <= 0) {
            dimension_ = dim;
        }
        delta_norms_sq.push_back(norm_sq_full);
        const size_t off = delta_vecs.size();
        delta_vecs.resize(off + static_cast<size_t>(dim));
        std::memcpy(delta_vecs.data() + off, vec, static_cast<size_t>(dim) * sizeof(float));
    }

    // -------------------------------------------------------------------------
    // 有序段：按标量序插入单条（含向量 payload）
    // -------------------------------------------------------------------------
    void insert_main_at(size_t pos, uint32_t id, float scalar, const float* vec, float norm_sq_full) {
        main_scalars.insert(main_scalars.begin() + static_cast<std::ptrdiff_t>(pos), scalar);
        main_ids.insert(main_ids.begin() + static_cast<std::ptrdiff_t>(pos), id);
        observe_histogram_scalar(scalar);
        if (vec != nullptr && dimension_ > 0 && norm_sq_full >= 0.0f) {
            main_norms_sq.insert(main_norms_sq.begin() + static_cast<std::ptrdiff_t>(pos), norm_sq_full);
            const size_t byte_pos = pos * static_cast<size_t>(dimension_);
            main_vecs.insert(main_vecs.begin() + static_cast<std::ptrdiff_t>(byte_pos),
                             vec, vec + dimension_);
        } else if (dimension_ > 0) {
            main_norms_sq.insert(main_norms_sq.begin() + static_cast<std::ptrdiff_t>(pos), 0.0f);
            const size_t byte_pos = pos * static_cast<size_t>(dimension_);
            main_vecs.insert(main_vecs.begin() + static_cast<std::ptrdiff_t>(byte_pos),
                             static_cast<size_t>(dimension_), 0.0f);
        }
    }

    void insert_main_metadata_at(size_t pos, uint32_t id, float scalar) {
        main_scalars.insert(main_scalars.begin() + static_cast<std::ptrdiff_t>(pos), scalar);
        main_ids.insert(main_ids.begin() + static_cast<std::ptrdiff_t>(pos), id);
        observe_histogram_scalar(scalar);
    }

    void append_main(uint32_t id, float scalar, const float* vec, float norm_sq_full) {
        main_scalars.push_back(scalar);
        main_ids.push_back(id);
        observe_histogram_scalar(scalar);
        if (dimension_ > 0) {
            main_norms_sq.push_back(norm_sq_full >= 0.0f ? norm_sq_full : 0.0f);
            if (vec != nullptr) {
                const size_t off = main_vecs.size();
                main_vecs.resize(off + static_cast<size_t>(dimension_));
                std::memcpy(main_vecs.data() + off, vec, static_cast<size_t>(dimension_) * sizeof(float));
            } else {
                main_vecs.resize(main_vecs.size() + static_cast<size_t>(dimension_), 0.0f);
            }
        }
    }

    /// 仅 metadata 批量赋值后，由外部按 main_ids 顺序重建 packed 向量
    void set_main_metadata(std::vector<float>&& scalars,
                           std::vector<uint32_t>&& ids) {
        main_scalars = std::move(scalars);
        main_ids = std::move(ids);
        main_norms_sq.clear();
        main_vecs.clear();
        rebuild_histogram_from_all();
    }

    void rebuild_main_packed(int dim,
                             const float* vectors_flat,
                             size_t n_vectors_total,
                             const std::vector<float>& vector_norms_half) {
        if (dim <= 0) {
            return;
        }
        dimension_ = dim;
        const size_t n = main_ids.size();
        main_norms_sq.resize(n);
        main_vecs.resize(n * static_cast<size_t>(dim));
        const size_t dim_sz = static_cast<size_t>(dim);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t vid = main_ids[i];
            float norm_full = 0.0f;
            if (vid < vector_norms_half.size()) {
                norm_full = vector_norms_half[vid] * 2.0f;
            }
            main_norms_sq[i] = norm_full;
            float* dst = main_vecs.data() + i * dim_sz;
            if (vectors_flat != nullptr && vid < n_vectors_total) {
                const float* src = vectors_flat + static_cast<size_t>(vid) * dim_sz;
                std::memcpy(dst, src, dim_sz * sizeof(float));
            } else {
                std::fill(dst, dst + dim_sz, 0.0f);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Module A — 函数3：merge() — delta 排序后与 main 双路归并
    // -------------------------------------------------------------------------
    void merge_unordered_into_main(bool keep_packed_payload = true) {
        if (delta_ids.empty()) {
            return;
        }
        const size_t dn = delta_ids.size();
        const bool main_packed =
            main_ids.empty() ||
            (dimension_ > 0 &&
             main_vecs.size() == main_ids.size() * static_cast<size_t>(dimension_) &&
             main_norms_sq.size() == main_ids.size());
        const bool delta_packed =
            dimension_ > 0 && delta_vecs.size() == dn * static_cast<size_t>(dimension_) &&
            delta_norms_sq.size() == dn;
        const bool keep_payload =
            keep_packed_payload && dimension_ > 0 && main_packed && delta_packed;

        struct DeltaEntry {
            float scalar;
            uint32_t id;
            size_t index;
        };

        std::vector<DeltaEntry> buf;
        buf.reserve(dn);
        for (size_t i = 0; i < dn; ++i) {
            buf.push_back(DeltaEntry{delta_scalars[i], delta_ids[i], i});
        }
        std::sort(buf.begin(), buf.end(),
                  [](const DeltaEntry& a, const DeltaEntry& b) {
                      if (a.scalar != b.scalar) {
                          return a.scalar < b.scalar;
                      }
                      if (a.id != b.id) {
                          return a.id < b.id;
                      }
                      return a.index < b.index;
                  });

        std::vector<float> new_scalars;
        std::vector<uint32_t> new_ids;
        std::vector<float> new_norms;
        std::vector<float> new_vecs;
        new_scalars.reserve(main_scalars.size() + dn);
        new_ids.reserve(main_ids.size() + dn);
        if (keep_payload) {
            new_norms.reserve(main_norms_sq.size() + dn);
        }
        if (keep_payload) {
            new_vecs.reserve(main_vecs.size() + dn * static_cast<size_t>(dimension_));
        }

        auto append_delta_entry = [&](size_t delta_idx, float scalar, uint32_t id) {
            new_scalars.push_back(scalar);
            new_ids.push_back(id);
            if (!keep_payload) {
                return;
            }
            if (delta_packed) {
                new_norms.push_back(delta_norms_sq[delta_idx]);
            } else {
                new_norms.push_back(0.0f);
            }
            {
                const float* src = delta_vecs.data() + delta_idx * static_cast<size_t>(dimension_);
                const size_t off = new_vecs.size();
                new_vecs.resize(off + static_cast<size_t>(dimension_));
                std::memcpy(new_vecs.data() + off, src, static_cast<size_t>(dimension_) * sizeof(float));
            }
        };

        size_t i = 0, j = 0;
        while (i < main_scalars.size() && j < buf.size()) {
            if (main_scalars[i] <= buf[j].scalar) {
                new_scalars.push_back(main_scalars[i]);
                new_ids.push_back(main_ids[i]);
                if (keep_payload) {
                    new_norms.push_back(i < main_norms_sq.size() ? main_norms_sq[i] : 0.0f);
                    const float* src = main_vecs.data() + i * static_cast<size_t>(dimension_);
                    const size_t off = new_vecs.size();
                    new_vecs.resize(off + static_cast<size_t>(dimension_));
                    std::memcpy(new_vecs.data() + off, src, static_cast<size_t>(dimension_) * sizeof(float));
                }
                ++i;
            } else {
                append_delta_entry(buf[j].index, buf[j].scalar, buf[j].id);
                ++j;
            }
        }
        while (i < main_scalars.size()) {
            new_scalars.push_back(main_scalars[i]);
            new_ids.push_back(main_ids[i]);
            if (dimension_ > 0) {
                new_norms.push_back(i < main_norms_sq.size() ? main_norms_sq[i] : 0.0f);
                if (keep_payload) {
                    const float* src = main_vecs.data() + i * static_cast<size_t>(dimension_);
                    const size_t off = new_vecs.size();
                    new_vecs.resize(off + static_cast<size_t>(dimension_));
                    std::memcpy(new_vecs.data() + off, src, static_cast<size_t>(dimension_) * sizeof(float));
                }
            }
            ++i;
        }
        while (j < buf.size()) {
            append_delta_entry(buf[j].index, buf[j].scalar, buf[j].id);
            ++j;
        }

        main_scalars = std::move(new_scalars);
        main_ids = std::move(new_ids);
        if (keep_payload) {
            main_norms_sq = std::move(new_norms);
            main_vecs = std::move(new_vecs);
        } else {
            main_norms_sq.clear();
            main_vecs.clear();
        }
        delta_ids.clear();
        delta_scalars.clear();
        delta_norms_sq.clear();
        delta_vecs.clear();
        reset_delta_stats();
        rebuild_histogram_from_all();
    }

    // -------------------------------------------------------------------------
    // 在有序段上对 [L,R] 二分，返回下标区间 [start, end)
    // -------------------------------------------------------------------------
    std::pair<size_t, size_t> binary_search_range(float L, float R) const {
        if (main_scalars.empty()) {
            return {0, 0};
        }
        auto lower_it = std::lower_bound(main_scalars.begin(), main_scalars.end(), L);
        size_t start_idx = static_cast<size_t>(std::distance(main_scalars.begin(), lower_it));
        auto upper_it = std::upper_bound(main_scalars.begin(), main_scalars.end(), R);
        size_t end_idx = static_cast<size_t>(std::distance(main_scalars.begin(), upper_it));
        return {start_idx, end_idx};
    }

    struct MainRangeView {
        size_t start{0};
        size_t count{0};
        const uint32_t* ids{nullptr};
        const float* norms_sq{nullptr};
        const float* vecs{nullptr};  // count * dimension row-major
    };

    MainRangeView main_range_view(float L, float R) const {
        MainRangeView view;
        const auto pr = binary_search_range(L, R);
        view.start = pr.first;
        view.count = (pr.second > pr.first) ? (pr.second - pr.first) : 0;
        if (view.count == 0) {
            return view;
        }
        view.ids = main_ids.data() + view.start;
        if (main_is_packed()) {
            view.norms_sq = main_norms_sq.data() + view.start;
            view.vecs = main_vecs.data() + view.start * static_cast<size_t>(dimension_);
        }
        return view;
    }

    size_t count_in_range(float L, float R) const {
        const auto pr = binary_search_range(L, R);
        size_t count = (pr.second > pr.first) ? (pr.second - pr.first) : 0;
        for (float scalar : delta_scalars) {
            if (scalar >= L && scalar <= R) {
                ++count;
            }
        }
        return count;
    }

    // -------------------------------------------------------------------------
    // 固定全局 domain 标量直方图：插入 O(1) 更新，用于选择率估算（不能硬剪枝）。
    // -------------------------------------------------------------------------
    static constexpr int kHistBins = 32;
    static constexpr int kHistSlots = kHistBins + 2;  // [0]=underflow, [1..bins]=domain, [last]=overflow

    float hist_min_{0.0f};
    float hist_max_{0.0f};
    uint32_t hist_counts_[kHistSlots]{};
    uint32_t hist_total_{0};
    bool hist_valid_{false};

    /// 兼容旧调用名：现在从 main + delta 重建固定 domain 直方图。
    void build_histogram_from_main() {
        rebuild_histogram_from_all();
    }

    void rebuild_histogram_from_all() {
        for (int b = 0; b < kHistSlots; ++b) hist_counts_[b] = 0;
        hist_total_ = 0;
        if (!hist_valid_ || !(hist_min_ < hist_max_)) return;
        for (float scalar : main_scalars) observe_histogram_scalar(scalar);
        for (float scalar : delta_scalars) observe_histogram_scalar(scalar);
    }

    /// 估计 [L, R] 的向量占比（0..1）：main+delta 都来自固定 domain 直方图。
    float estimate_selectivity_combined(float L, float R) const {
        if (total_size() == 0) return 0.0f;
        return estimate_selectivity_hist(L, R);
    }

    /// 使用固定 domain 直方图估计 [L, R] 的向量占比（0..1）。
    /// 若直方图无效，返回 1.0（不剪枝/不降权）。
    float estimate_selectivity_hist(float L, float R) const {
        if (!hist_valid_ || hist_total_ == 0 || !(hist_min_ < hist_max_)) return 1.0f;
        if (L > R) return 0.0f;
        const float range = (hist_max_ - hist_min_);
        double hit = 0.0;
        if (L < hist_min_) hit += hist_counts_[0];
        if (R > hist_max_) hit += hist_counts_[kHistSlots - 1];
        if (R < hist_min_ || L > hist_max_) {
            return static_cast<float>(hit / static_cast<double>(hist_total_));
        }

        const float inv_range = static_cast<float>(kHistBins) / range;
        const float b_lo = std::max(0.0f, (L - hist_min_) * inv_range);
        const float b_hi = std::min(static_cast<float>(kHistBins), (R - hist_min_) * inv_range);

        for (int b = 0; b < kHistBins; ++b) {
            const float blo = static_cast<float>(b);
            const float bhi = static_cast<float>(b + 1);
            if (bhi <= b_lo || blo > b_hi) continue;
            const float lo_clip = std::max(blo, b_lo);
            const float hi_clip = std::min(bhi, b_hi);
            const int slot = b + 1;
            if (hi_clip > lo_clip && hist_counts_[slot] > 0) {
                hit += static_cast<double>(hist_counts_[slot]) * (hi_clip - lo_clip);
            }
        }
        return static_cast<float>(hit / static_cast<double>(hist_total_));
    }

    bool delta_block_may_overlap(size_t block_idx, float L, float R) const {
        if (block_idx >= delta_block_min_.size() || block_idx >= delta_block_max_.size()) {
            return true;
        }
        return !(R < delta_block_min_[block_idx] || L > delta_block_max_[block_idx]);
    }

    void rebuild_delta_stats_from_delta() {
        reset_delta_stats();
        for (float scalar : delta_scalars) {
            observe_delta_scalar(scalar);
        }
    }


    std::pair<uint32_t, float> get_main_entry(size_t idx) const {
        if (idx >= main_ids.size()) {
            return {0, 0.0f};
        }
        return {main_ids[idx], main_scalars[idx]};
    }

    void flush_delta_before_read() { merge_unordered_into_main(); }

    void clear_all() {
        main_ids.clear();
        main_scalars.clear();
        main_norms_sq.clear();
        main_vecs.clear();
        delta_ids.clear();
        delta_scalars.clear();
        delta_norms_sq.clear();
        delta_vecs.clear();
        reset_delta_stats();
        hist_valid_ = false;
        hist_total_ = 0;
        for (int b = 0; b < kHistSlots; ++b) hist_counts_[b] = 0;
    }

    void release_packed_payload() {
        main_norms_sq.clear();
        main_vecs.clear();
        delta_norms_sq.clear();
        delta_vecs.clear();
        main_norms_sq.shrink_to_fit();
        main_vecs.shrink_to_fit();
        delta_norms_sq.shrink_to_fit();
        delta_vecs.shrink_to_fit();
    }

private:
    void observe_delta_scalar(float scalar) {
        if (!delta_stats_valid_) {
            delta_min_ = scalar;
            delta_max_ = scalar;
            delta_stats_valid_ = true;
        } else {
            delta_min_ = std::min(delta_min_, scalar);
            delta_max_ = std::max(delta_max_, scalar);
        }
        const size_t idx = delta_scalars.empty() ? 0 : (delta_scalars.size() - 1);
        const size_t block = idx / kDeltaBlockSize;
        if (delta_block_min_.size() <= block) {
            delta_block_min_.push_back(scalar);
            delta_block_max_.push_back(scalar);
        } else {
            delta_block_min_[block] = std::min(delta_block_min_[block], scalar);
            delta_block_max_[block] = std::max(delta_block_max_[block], scalar);
        }
    }

    void observe_histogram_scalar(float scalar) {
        if (!hist_valid_ || !(hist_min_ < hist_max_)) return;
        int slot = 0;
        if (scalar < hist_min_) {
            slot = 0;
        } else if (scalar > hist_max_) {
            slot = kHistSlots - 1;
        } else {
            const float inv_range = static_cast<float>(kHistBins) / (hist_max_ - hist_min_);
            int b = static_cast<int>((scalar - hist_min_) * inv_range);
            if (b < 0) b = 0;
            if (b >= kHistBins) b = kHistBins - 1;
            slot = b + 1;
        }
        ++hist_counts_[slot];
        ++hist_total_;
    }

    void reset_delta_stats() {
        delta_min_ = 0.0f;
        delta_max_ = 0.0f;
        delta_stats_valid_ = false;
        delta_block_min_.clear();
        delta_block_max_.clear();
    }

    void move_from(RangeAwareList&& other) noexcept {
        dimension_ = other.dimension_;
        main_ids = std::move(other.main_ids);
        main_scalars = std::move(other.main_scalars);
        main_norms_sq = std::move(other.main_norms_sq);
        main_vecs = std::move(other.main_vecs);
        delta_ids = std::move(other.delta_ids);
        delta_scalars = std::move(other.delta_scalars);
        delta_norms_sq = std::move(other.delta_norms_sq);
        delta_vecs = std::move(other.delta_vecs);
        delta_min_ = other.delta_min_;
        delta_max_ = other.delta_max_;
        delta_stats_valid_ = other.delta_stats_valid_;
        delta_block_min_ = std::move(other.delta_block_min_);
        delta_block_max_ = std::move(other.delta_block_max_);
        hist_min_ = other.hist_min_;
        hist_max_ = other.hist_max_;
        hist_total_ = other.hist_total_;
        hist_valid_ = other.hist_valid_;
        for (int b = 0; b < kHistSlots; ++b) hist_counts_[b] = other.hist_counts_[b];
        other.dimension_ = 0;
        other.reset_delta_stats();
    }
};
