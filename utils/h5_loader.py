"""
H5数据加载工具
用于读取处理好的SIFT数据集的h5格式文件
"""

import h5py
import numpy as np
from typing import Tuple, Optional


def load_h5_data(
    h5_file: str,
    max_base: Optional[int] = None,
    max_query: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从h5文件加载数据
    
    Args:
        h5_file: h5文件路径
        max_base: 最大读取基础向量数量
        max_query: 最大读取查询数量
        
    Returns:
        (base_vectors, base_scalars, query_vectors, query_ranges, groundtruth, learn_vectors)
    """
    print(f"从H5文件加载数据: {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
        # 加载基础向量
        base_vectors = f['base'][:]
        base_scalars = f['base_scalars'][:]
        
        if max_base is not None and max_base < len(base_vectors):
            base_vectors = base_vectors[:max_base]
            base_scalars = base_scalars[:max_base]
        
        print(f"  基础向量: {base_vectors.shape}, 属性值范围: [{base_scalars.min()}, {base_scalars.max()}]")
        
        # 加载查询向量
        query_vectors = f['test'][:]
        if max_query is not None and max_query < len(query_vectors):
            query_vectors = query_vectors[:max_query]

        print(f"  查询向量: {query_vectors.shape}")

        n_base_actual = len(base_vectors)

        # 范围 + GT：若存在按数据量生成的 gt_hybrid_size_*（DEEP/WIT 均可），优先用其配对的
        # ranges_size_*，与动态实验中 get_gt_and_ranges_for_size(use_hybrid_gt=True) 口径一致；
        # 否则再回退到全局 test_hybrid_knn + test_ranges，最后才是纯向量 gt_size_*。
        gt_hybrid_keys = [k for k in f.keys() if k.startswith('gt_hybrid_size_')]
        if gt_hybrid_keys:
            sizes = sorted(int(k.split('_')[-1]) for k in gt_hybrid_keys)
            best = min(sizes, key=lambda s: abs(s - n_base_actual))
            gt_key = f'gt_hybrid_size_{best}'
            ranges_key = f'ranges_size_{best}'
            if ranges_key not in f:
                raise KeyError(
                    f"H5 含 {gt_key} 但缺少 {ranges_key}，无法对齐混合检索的范围真值。"
                )
            query_ranges = f[ranges_key][:]
            groundtruth = f[gt_key][:]
            print(
                f"  使用按数据量混合GT: {gt_key} + {ranges_key} "
                f"(库规模 {n_base_actual}, 选中规模 {best})"
            )
        elif 'test_hybrid_knn' in f:
            query_ranges = f['test_ranges'][:]
            groundtruth = f['test_hybrid_knn'][:]
            print(f"  使用全局 test_hybrid_knn + test_ranges（H5 无 gt_hybrid_size_*）")
        else:
            query_ranges = f['test_ranges'][:]
            gt_keys = [k for k in f.keys() if k.startswith('gt_size_') and 'hybrid' not in k]
            if not gt_keys:
                raise KeyError(
                    "H5 中既无 gt_hybrid_size_* 也无 test_hybrid_knn / gt_size_*。"
                    "请先运行 generate_hybrid_gt.py 或 generate_*_h5_with_sizes.py"
                )
            sizes = sorted(int(k.split('_')[-1]) for k in gt_keys)
            best = min(sizes, key=lambda s: abs(s - n_base_actual))
            gt_key = f'gt_size_{best}'
            groundtruth = f[gt_key][:]
            print(f"  使用 {gt_key} 作为 GT (数据量 {n_base_actual}, 选中规模 {best})")

        if max_query is not None and max_query < len(query_ranges):
            query_ranges = query_ranges[:max_query]
        if max_query is not None and max_query < len(groundtruth):
            groundtruth = groundtruth[:max_query]

        print(f"  查询范围: {query_ranges.shape}, 范围值: [{query_ranges.min()}, {query_ranges.max()}]")
        
        print(f"  真实结果: {groundtruth.shape}")
        
        # 根据数据量自动调整训练集大小
        # 策略：对于大规模数据（>=100万），使用10万作为训练集（符合论文要求）
        # 对于小规模数据，使用30-40%作为训练集
        if max_base is not None:
            # 如果指定了max_base，根据数据量动态调整训练集大小
            if max_base >= 1000000:
                # 100万以上数据：使用10万作为训练集（符合论文要求）
                train_size = 100000
            elif max_base >= 500000:
                # 50-100万数据：使用40%作为训练集
                train_size = int(max_base * 0.4)
            else:
                # 50万以下数据：使用30%作为训练集
                train_size = int(max_base * 0.3)
            
            train_size = max(100000, train_size)  # 至少10万条
            train_size = min(train_size, 1000000)  # 最多100万条（提高上限）
            train_size = min(train_size, len(base_vectors))  # 不超过实际数据量
        else:
            # 如果没有指定max_base，使用全部数据的30%，但至少10万条
            train_size = max(100000, int(len(base_vectors) * 0.3))
            train_size = min(train_size, len(base_vectors))
        
        learn_vectors = base_vectors[:train_size]
        print(f"  训练向量: {learn_vectors.shape} (训练集大小: {train_size}, 占总数据量的 {train_size/len(base_vectors)*100:.1f}%)")
    
    return base_vectors, base_scalars, query_vectors, query_ranges, groundtruth, learn_vectors


def get_gt_and_ranges_for_size(
    h5_file: str,
    current_size: int,
    with_selected: bool = False,
    use_hybrid_gt: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    根据当前数据量获取对应的groundtruth和ranges
    
    Args:
        h5_file: h5文件路径
        current_size: 当前已插入的向量数量
        use_hybrid_gt: True 时使用 gt_hybrid_size_*（混合查询真值），否则用 gt_size_*（纯向量真值）
        
    Returns:
        - 如果with_selected=False（默认）：(groundtruth, query_ranges)
        - 如果with_selected=True ：(groundtruth, query_ranges, selected_size)
    """
    gt_prefix = 'gt_hybrid_size_' if use_hybrid_gt else 'gt_size_'
    with h5py.File(h5_file, 'r') as f:
        # 找到最接近的gt和ranges
        # 查找所有可用的数据量
        available_sizes = []
        for key in f.keys():
            if key.startswith(gt_prefix):
                size = int(key.split('_')[-1])
                available_sizes.append(size)

        # 请求混合分规模 GT，但 H5 中无 gt_hybrid_size_*（常见于 SIFT：仅有 gt_size_* + test_hybrid_knn）
        if use_hybrid_gt and not available_sizes:
            has_vector_gt = any(k.startswith('gt_size_') and 'hybrid' not in k for k in f.keys())
            if has_vector_gt:
                import sys
                print(
                    "  [GT选择] H5 无 gt_hybrid_size_*，回退为 gt_size_* + ranges_size_*（分规模纯向量真值）",
                    file=sys.stderr,
                )
                return get_gt_and_ranges_for_size(
                    h5_file, current_size, with_selected=with_selected, use_hybrid_gt=False
                )
            raise FileNotFoundError(
                f"H5 中无 gt_hybrid_size_* 且无可用 gt_size_*，无法为混合检索选 GT。"
                f"可运行: python utils/generate_hybrid_gt.py --h5-file {h5_file}"
            )
        
        # 这样可以确保不同数据量使用对应的GT和查询范围
        available_sizes.sort()
        
        # 找到最接近current_size的数据量
        selected_size = None
        min_diff = float('inf')
        
        for size in available_sizes:
            diff = abs(size - current_size)
            if diff < min_diff:
                min_diff = diff
                selected_size = size
        
        # 如果没有找到，使用最小的
        if selected_size is None:
            selected_size = min(available_sizes) if available_sizes else 10000

        gt_key = f'{gt_prefix}{selected_size}'
        ranges_key = f'ranges_size_{selected_size}'
        
        if gt_key in f and ranges_key in f:
            groundtruth = f[gt_key][:]
            query_ranges = f[ranges_key][:]

            # 则需要过滤超出 current_size 的 GT 索引，否则召回率会被系统性拉低。
            if selected_size > current_size:
                filtered_gt = groundtruth.copy()
                invalid_mask = filtered_gt >= current_size
                filtered_gt[invalid_mask] = -1
                groundtruth = filtered_gt
            # 使用print输出，会被重定向到日志（stderr -> tee）
            import sys
            tag = "混合GT" if use_hybrid_gt else "纯向量GT"
            print(f"  [GT选择] {tag} 数据量 {selected_size} (当前插入: {current_size})", file=sys.stderr)
            print(f"  [GT选择] GT shape: {groundtruth.shape}, 索引范围: [{groundtruth.min()}, {groundtruth.max()}]", file=sys.stderr)
            print(f"  [GT选择] Ranges shape: {query_ranges.shape}, 范围值: [{query_ranges.min():.1f}, {query_ranges.max():.1f}]", file=sys.stderr)
            return (groundtruth, query_ranges, selected_size) if with_selected else (groundtruth, query_ranges)
        else:
            # 回退到test_hybrid_knn和test_ranges（仅 DEEP 等数据集有此 key）
            # WIT 等仅含 gt_hybrid_size_*，无 test_hybrid_knn，需抛清晰错误
            if 'test_hybrid_knn' not in f or 'test_ranges' not in f:
                raise KeyError(
                    f"H5 中无 {gt_key} 或 {ranges_key}，且无 test_hybrid_knn/test_ranges 可回退。"
                    f"请确保已运行 generate_hybrid_gt.py 生成 gt_{'hybrid_' if use_hybrid_gt else ''}size_* 与 ranges_size_*"
                )
            groundtruth = f['test_hybrid_knn'][:]
            query_ranges = f['test_ranges'][:]
            
            # 过滤超出范围的GT索引
            # 将超出范围的索引替换为-1，然后在计算召回率时忽略
            filtered_gt = groundtruth.copy()
            invalid_mask = filtered_gt >= current_size
            filtered_gt[invalid_mask] = -1
            
            invalid_count = np.sum(invalid_mask)
            if invalid_count > 0:
                print(f"  使用默认GT和ranges (当前: {current_size})")
                print(f"  ⚠️  过滤掉 {invalid_count} 个超出范围的GT索引 (>= {current_size})")
                print(f"  GT索引范围: [{groundtruth.min()}, {groundtruth.max()}] -> 过滤后: [0, {current_size-1}]")
            else:
                print(f"  使用默认GT和ranges (当前: {current_size})")
                print(f"  ✓ GT索引全部有效 (最大索引 {groundtruth.max()} < 数据量 {current_size})")
            
            return (filtered_gt, query_ranges, current_size) if with_selected else (filtered_gt, query_ranges)



def get_query_gt_and_ranges_for_size(
    h5_file: str,
    current_size: int,
    fallback_queries: np.ndarray,
    with_selected: bool = False,
    use_hybrid_gt: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return per-size query vectors, GT and ranges for dynamic workloads.

    Older H5 files store one global `test` matrix plus per-size ranges/GT.
    YT8M timestamp workloads can additionally store `test_size_N`, so query
    time positions evolve with the inserted prefix. This keeps old files
    compatible while enabling dynamic per-batch query sets.
    """
    gt, ranges, selected_size = get_gt_and_ranges_for_size(
        h5_file, current_size, with_selected=True, use_hybrid_gt=use_hybrid_gt
    )
    query_key = f'test_size_{selected_size}'
    with h5py.File(h5_file, 'r') as f:
        if query_key in f:
            queries = f[query_key][:]
            import sys
            print(f"  [Query选择] 使用动态查询 {query_key} shape={queries.shape}", file=sys.stderr)
        else:
            queries = fallback_queries
            import sys
            print(f"  [Query选择] 使用全局 test 查询 shape={queries.shape}", file=sys.stderr)
    if with_selected:
        return queries, gt, ranges, selected_size
    return queries, gt, ranges
