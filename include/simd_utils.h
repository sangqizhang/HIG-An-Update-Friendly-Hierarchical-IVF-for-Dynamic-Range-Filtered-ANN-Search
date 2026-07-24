// simd_utils.h
// SIMD优化的距离计算函数（基于参考实现）

#pragma once

#include <cstddef>

#ifdef __AVX__
  #include <immintrin.h>
  #define USE_AVX 1
#else
  #define USE_AVX 0
  #ifdef __ARM_NEON
    #include <arm_neon.h>
    #define USE_NEON 1
  #else
    #define USE_NEON 0
  #endif
#endif

// SIMD优化的内积计算（用于L2距离）
float IPSIMD4ExtAVX(float *pVect1, float *pVect2, size_t qty);

// SIMD优化的L2距离计算（使用预计算的范数）
// pVect1: 查询向量
// pVect2: 数据向量
// norm_bsq: 数据向量的范数平方的一半 (||b||^2 / 2)
// qty: 向量维度
// 返回: -L2距离的负值（用于相似度排序，值越大越相似）
float L2SIMD4ExtAVX(float *pVect1, float *pVect2, float norm_bsq, size_t qty);

// 方案 A：对连续 row-major 矩阵 X (n×dim) 批量计算 ||q - x_i||^2
// dist_sq_out[i] = query_norm_sq + norms_sq[i] - 2*<q, x_i>
void batch_l2_sq_q_rows(const float* query,
                        int dim,
                        const float* rows,
                        const float* norms_sq,
                        int n_rows,
                        float query_norm_sq,
                        float* dist_sq_out);
