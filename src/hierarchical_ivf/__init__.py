"""
层次化Ada-IVF索引库
基于论文: Incremental IVF Index Maintenance for Streaming Vector Search

主要模块:
- hierarchical_ada_ivf: 层次化Ada-IVF实现（Ada-IVF扩展到两级结构）
  - 支持范围混合过滤
  - 支持动态插入
  - 支持索引增量式更新
  - 优化搜索策略（先范围过滤再k-NN）
  - 向量化距离计算
"""

from .hierarchical_ada_ivf import HierarchicalAdaIVFIndex

# 内部依赖（不对外暴露）
from .ada_ivf import ClusterStats  # 仅用于类型提示

__version__ = "1.0.0"
__all__ = [
    "HierarchicalAdaIVFIndex",
]


