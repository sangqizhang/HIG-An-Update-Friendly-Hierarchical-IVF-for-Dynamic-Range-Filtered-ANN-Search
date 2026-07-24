// simd_utils.cpp
// SIMD优化的距离计算实现（基于参考实现）

#include "../include/simd_utils.h"
#include <cmath>
#include <cstdio>

#if USE_AVX
float IPSIMD4ExtAVX(float *pVect1, float *pVect2, size_t qty) {
    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;
    
    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4 * qty4;
    const float* pEnd3 = pVect1 + qty;
    
    __m256 sum256 = _mm256_setzero_ps();
    
    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1);
        __m256 v2 = _mm256_loadu_ps(pVect2);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        
        pVect1 += 8;
        pVect2 += 8;
        
        v1 = _mm256_loadu_ps(pVect1);
        v2 = _mm256_loadu_ps(pVect2);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        
        pVect1 += 8;
        pVect2 += 8;
    }
    
    __m128 v1, v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
    
    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        v2 = _mm_loadu_ps(pVect2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        
        pVect1 += 4;
        pVect2 += 4;
    }
    
    float sum = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(sum_prod, sum_prod), sum_prod));
    
    while (pVect1 < pEnd3) {
        sum += (*pVect1) * (*pVect2);
        pVect1++;
        pVect2++;
    }
    
    return sum;
}

float L2SIMD4ExtAVX(float *pVect1, float *pVect2, float norm_bsq, size_t qty) {
    // 参考实现：返回 (IP - norm_bsq)
    // 其中 norm_bsq = ||b||^2 / 2
    // 实际使用时：dist_score = -L2SIMD4ExtAVX(...) = -(IP - norm_bsq) = norm_bsq - IP
    // 这个值用于相似度排序（值越大越相似），不是实际距离
    // 要得到实际L2距离平方，需要：||a||^2 + ||b||^2 - 2*IP = 2*(norm_bsq - IP) + ||a||^2
    // 但为了保持与参考实现一致，直接返回 (IP - norm_bsq)
    return (IPSIMD4ExtAVX(pVect1, pVect2, qty) - norm_bsq);
}

#elif USE_NEON
// ARM NEON实现（如果需要支持ARM架构）
float IPSIMD4ExtAVX(float *pVect1, float *pVect2, size_t qty) {
    float sum = 0.0f;
    for (size_t i = 0; i < qty; ++i) {
        sum += pVect1[i] * pVect2[i];
    }
    return sum;
}

float L2SIMD4ExtAVX(float *pVect1, float *pVect2, float norm_bsq, size_t qty) {
    return (IPSIMD4ExtAVX(pVect1, pVect2, qty) - norm_bsq);
}

#else
// 回退到标量实现
float IPSIMD4ExtAVX(float *pVect1, float *pVect2, size_t qty) {
    float sum = 0.0f;
    for (size_t i = 0; i < qty; ++i) {
        sum += pVect1[i] * pVect2[i];
    }
    return sum;
}

float L2SIMD4ExtAVX(float *pVect1, float *pVect2, float norm_bsq, size_t qty) {
    return (IPSIMD4ExtAVX(pVect1, pVect2, qty) - norm_bsq);
}
#endif

void batch_l2_sq_q_rows(const float* query,
                        int dim,
                        const float* rows,
                        const float* norms_sq,
                        int n_rows,
                        float query_norm_sq,
                        float* dist_sq_out) {
    if (query == nullptr || rows == nullptr || norms_sq == nullptr ||
        dist_sq_out == nullptr || dim <= 0 || n_rows <= 0) {
        return;
    }
    const size_t dim_sz = static_cast<size_t>(dim);
    for (int i = 0; i < n_rows; ++i) {
        const float* row = rows + static_cast<size_t>(i) * dim_sz;
        const float ip = IPSIMD4ExtAVX(const_cast<float*>(query),
                                       const_cast<float*>(row),
                                       dim_sz);
        float dsq = query_norm_sq + norms_sq[static_cast<size_t>(i)] - 2.0f * ip;
        dist_sq_out[static_cast<size_t>(i)] = dsq > 0.0f ? dsq : 0.0f;
    }
}
