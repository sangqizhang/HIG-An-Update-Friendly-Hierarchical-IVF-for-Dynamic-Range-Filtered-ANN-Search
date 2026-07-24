// product_quantizer.h
// 乘积量化（PQ）：将向量切成 m 段子向量，每段用 ksub 个质心之一编码（默认 uint8，ksub<=256）。
// 用于 Module PQ：以 m 字节/向量的码本索引替代全精度浮点存储，检索用 ADC（非对称距离计算）。
//
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

class ProductQuantizer {
public:
    ProductQuantizer() = default;

    /// m：子向量段数；ksub：每段子码本大小（≤256，每段 1 字节）
    void set_params(int m_segments, int ksub) {
        if (m_segments < 1) {
            throw std::invalid_argument("PQ: m_segments must be >= 1");
        }
        if (ksub < 2 || ksub > 256) {
            throw std::invalid_argument("PQ: ksub must be in [2, 256]");
        }
        m_ = m_segments;
        ksub_ = ksub;
        trained_ = false;
        codebooks_.clear();
    }

    int m() const { return m_; }
    int ksub() const { return ksub_; }
    int dsub() const { return dsub_; }
    int dimension() const { return dim_; }
    bool is_trained() const { return trained_; }

    /// 在 n 个 dim 维向量上训练码本（要求 dim % m == 0）
    void train(const float* data, size_t n_vectors, int dimension) {
        if (n_vectors == 0) {
            throw std::runtime_error("PQ train: n_vectors == 0");
        }
        if (dimension < m_) {
            throw std::runtime_error("PQ train: dimension < m");
        }
        if (dimension % m_ != 0) {
            throw std::runtime_error("PQ train: dimension must be divisible by m");
        }
        dim_ = dimension;
        dsub_ = dimension / m_;
        codebooks_.assign(static_cast<size_t>(m_),
                          std::vector<std::vector<float>>(
                              static_cast<size_t>(ksub_),
                              std::vector<float>(static_cast<size_t>(dsub_), 0.0f)));

        std::vector<float> segment_points(n_vectors * static_cast<size_t>(dsub_));
        for (int seg = 0; seg < m_; ++seg) {
            const int offset = seg * dsub_;
            for (size_t i = 0; i < n_vectors; ++i) {
                std::memcpy(segment_points.data() + i * static_cast<size_t>(dsub_),
                            data + i * static_cast<size_t>(dim_) + offset,
                            static_cast<size_t>(dsub_) * sizeof(float));
            }
            kmeans_segment(segment_points.data(), n_vectors, seg);
        }
        trained_ = true;
    }

    /// 单向量编码，输出长度 m 的字节序列（每段一个字节）
    void encode(const float* vec, std::vector<uint8_t>& codes_out) const {
        if (!trained_) {
            throw std::runtime_error("PQ encode: not trained");
        }
        codes_out.resize(static_cast<size_t>(m_));
        for (int j = 0; j < m_; ++j) {
            const float* sub = vec + j * dsub_;
            float best = std::numeric_limits<float>::max();
            uint8_t best_idx = 0;
            for (int k = 0; k < ksub_; ++k) {
                float d = l2sq(sub, codebooks_[static_cast<size_t>(j)][static_cast<size_t>(k)].data(),
                               dsub_);
                if (d < best) {
                    best = d;
                    best_idx = static_cast<uint8_t>(k);
                }
            }
            codes_out[static_cast<size_t>(j)] = best_idx;
        }
    }

    void decode(const uint8_t* codes, float* vec_out) const {
        if (!trained_) {
            throw std::runtime_error("PQ decode: not trained");
        }
        for (int j = 0; j < m_; ++j) {
            const float* c = codebooks_[static_cast<size_t>(j)][codes[j]].data();
            std::memcpy(vec_out + j * dsub_, c, static_cast<size_t>(dsub_) * sizeof(float));
        }
    }

    /// ADC：对查询 q 预计算表，大小 m * ksub，行主序 table[j*ksub + k] = ||q_j - c_jk||^2
    void adc_query_precompute(const float* query, std::vector<float>& table_out) const {
        if (!trained_) {
            throw std::runtime_error("PQ adc_query_precompute: not trained");
        }
        table_out.resize(static_cast<size_t>(m_ * ksub_));
        for (int j = 0; j < m_; ++j) {
            const float* qsub = query + j * dsub_;
            for (int k = 0; k < ksub_; ++k) {
                table_out[static_cast<size_t>(j * ksub_ + k)] =
                    l2sq(qsub, codebooks_[static_cast<size_t>(j)][static_cast<size_t>(k)].data(), dsub_);
            }
        }
    }

    /// 近似（对重构向量 x̂ 的）平方 L2：sum_j ||q_j - ĉ_{j,z_j}||^2
    float adc_sum_distance(const uint8_t* codes, const float* adc_table) const {
        float s = 0.0f;
        for (int j = 0; j < m_; ++j) {
            s += adc_table[j * ksub_ + static_cast<int>(codes[j])];
        }
        return s;
    }

    void clear() {
        trained_ = false;
        dim_ = 0;
        dsub_ = 0;
        codebooks_.clear();
    }

private:
    static float l2sq(const float* a, const float* b, int d) {
        float s = 0.0f;
        for (int t = 0; t < d; ++t) {
            float df = a[t] - b[t];
            s += df * df;
        }
        return s;
    }

    void kmeans_segment(const float* points, size_t n, int seg) {
        const int d = dsub_;
        auto& centroids = codebooks_[static_cast<size_t>(seg)];
        const int kcent = ksub_;
        // 初始化：均匀抽样起点
        for (int k = 0; k < kcent; ++k) {
            size_t idx = (static_cast<size_t>(k) * n / static_cast<size_t>(kcent)) % n;
            std::memcpy(centroids[static_cast<size_t>(k)].data(),
                        points + idx * static_cast<size_t>(d),
                        static_cast<size_t>(d) * sizeof(float));
        }

        std::vector<int> assign(static_cast<size_t>(n), 0);
        const int max_iter = 20;
        for (int it = 0; it < max_iter; ++it) {
            // assign
            for (size_t i = 0; i < n; ++i) {
                const float* p = points + i * static_cast<size_t>(d);
                float best = std::numeric_limits<float>::max();
                int bk = 0;
                for (int k = 0; k < kcent; ++k) {
                    float ds = l2sq(p, centroids[static_cast<size_t>(k)].data(), d);
                    if (ds < best) {
                        best = ds;
                        bk = k;
                    }
                }
                assign[i] = bk;
            }
            // update
            std::vector<std::vector<float>> newc(static_cast<size_t>(kcent),
                                                 std::vector<float>(static_cast<size_t>(d), 0.0f));
            std::vector<int> cnt(static_cast<size_t>(kcent), 0);
            for (size_t i = 0; i < n; ++i) {
                int k = assign[i];
                for (int t = 0; t < d; ++t) {
                    newc[static_cast<size_t>(k)][static_cast<size_t>(t)] +=
                        points[i * static_cast<size_t>(d) + static_cast<size_t>(t)];
                }
                cnt[static_cast<size_t>(k)]++;
            }
            for (int k = 0; k < kcent; ++k) {
                if (cnt[static_cast<size_t>(k)] == 0) {
                    continue; // 空簇：保留上一轮质心
                }
                for (int t = 0; t < d; ++t) {
                    centroids[static_cast<size_t>(k)][static_cast<size_t>(t)] =
                        newc[static_cast<size_t>(k)][static_cast<size_t>(t)] /
                        static_cast<float>(cnt[static_cast<size_t>(k)]);
                }
            }
        }
    }

    int m_{8};
    int ksub_{256};
    int dim_{0};
    int dsub_{0};
    bool trained_{false};
    std::vector<std::vector<std::vector<float>>> codebooks_;
};
