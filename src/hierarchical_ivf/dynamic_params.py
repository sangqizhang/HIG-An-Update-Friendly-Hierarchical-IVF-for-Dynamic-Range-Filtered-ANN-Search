"""
动态参数调整模块
根据数据量、索引质量和性能指标动态调整搜索参数，平衡召回率和QPS
"""

import numpy as np
from typing import Tuple, Optional, Dict
import math


class DynamicParameterManager:
    """
    动态参数管理器
    根据数据量、索引状态和性能反馈动态调整搜索参数
    """
    
    def __init__(
        self,
        base_n_probe: int = 10,
        base_search_k: int = 5000,
        min_n_probe: int = 10,
        max_n_probe: int = 100,
        min_search_k: int = 2000,
        max_search_k: int = 50000,
        target_recall: float = 0.5,  # 目标召回率
        min_recall: float = 0.3,     # 最低可接受召回率
        qps_weight: float = 0.5,      # QPS权重（0-1，1表示更重视QPS）
        recall_weight: float = 0.5    # 召回率权重（0-1，1表示更重视召回率）
    ):
        """
        初始化动态参数管理器
        
        Args:
            base_n_probe: 基础n_probe值
            base_search_k: 基础search_k值
            min_n_probe: 最小n_probe值
            max_n_probe: 最大n_probe值
            min_search_k: 最小search_k值
            max_search_k: 最大search_k值
            target_recall: 目标召回率
            min_recall: 最低可接受召回率
            qps_weight: QPS权重（用于平衡召回率和QPS）
            recall_weight: 召回率权重
        """
        self.base_n_probe = base_n_probe
        self.base_search_k = base_search_k
        self.min_n_probe = min_n_probe
        self.max_n_probe = max_n_probe
        self.min_search_k = min_search_k
        self.max_search_k = max_search_k
        
        # 目标：确保QPS能达到三位数（100+）
        # 保持合理的上限，不降低search_k和n_probe，通过算法优化提升QPS
        self.max_n_probe = min(self.max_n_probe, 10)  # 限制n_probe最大为10
        self.max_search_k = min(self.max_search_k, 5000)  # 保持5000，通过算法优化提升QPS
        self.target_recall = target_recall
        self.min_recall = min_recall
        self.qps_weight = qps_weight
        self.recall_weight = recall_weight
        
        # 性能历史记录（用于自适应调整）
        self.recall_history = []
        self.qps_history = []
        self.param_history = []
        
    def get_dynamic_n_probe(
        self,
        data_size: int,
        n_fine_clusters: int,
        avg_cluster_size: float,
        max_cluster_size: int,
        initial_data_size: int = 10000
    ) -> int:
        """
        根据数据量和索引状态动态计算n_probe
        
        策略：
        1. 基于数据量增长的对数增长
        2. 基于聚类大小的线性调整
        3. 基于聚类不均匀度的惩罚
        
        Args:
            data_size: 当前数据量
            n_fine_clusters: 细粒度聚类数
            avg_cluster_size: 平均聚类大小
            max_cluster_size: 最大聚类大小
            initial_data_size: 初始数据量（用于归一化）
            
        Returns:
            调整后的n_probe值
        """
        # 1. 基础n_probe
        n_probe = self.base_n_probe
        
        # 2. 数据量增长因子（更保守的对数增长，避免过度增长）
        if data_size > initial_data_size:
            # 使用更保守的增长因子，确保n_probe不会增长过快
            size_factor = 1 + math.log(data_size / initial_data_size) / math.log(20)  # 从log(10)改为log(20)
            n_probe = int(n_probe * size_factor)
        
        # 3. 聚类大小因子（平均聚类越大，需要更多n_probe，但更保守）
        if avg_cluster_size > 1000:
            cluster_factor = 1 + (avg_cluster_size - 1000) / 2000  # 降低增长速率
            n_probe = int(n_probe * cluster_factor)
        
        # 4. 聚类不均匀度惩罚（最大/平均比值越大，需要更多n_probe，但更保守）
        if avg_cluster_size > 0:
            imbalance_ratio = max_cluster_size / avg_cluster_size
            if imbalance_ratio > 2.0:
                imbalance_factor = 1 + (imbalance_ratio - 2.0) * 0.1
                n_probe = int(n_probe * imbalance_factor)
        
        # 5. 限制在合理范围内（确保QPS能达到三位数）
        n_probe = max(self.min_n_probe, min(n_probe, self.max_n_probe))
        
        # 6. 不超过粗粒度聚类数（如果提供）
        # 注意：这里不限制，因为层次化搜索中n_probe是粗粒度聚类的n_probe
        
        return n_probe
    
    def get_dynamic_search_k(
        self,
        data_size: int,
        n_probe: int,
        avg_cluster_size: float,
        current_recall: Optional[float] = None,
        initial_data_size: int = 10000
    ) -> int:
        """
        根据n_probe和召回率动态计算search_k
        
        策略：
        1. 基于n_probe的线性调整（n_probe已经根据数据量动态调整，无需单独考虑数据量）
        2. 基于召回率反馈的自适应调整
        3. 基于平均聚类大小的调整
        
        Args:
            data_size: 当前数据量（保留参数，但不再用于计算）
            n_probe: 当前n_probe值（已经根据数据量动态调整）
            avg_cluster_size: 平均聚类大小
            current_recall: 当前召回率（如果提供，用于自适应调整）
            initial_data_size: 初始数据量（保留参数，但不再用于计算）
            
        Returns:
            调整后的search_k值
        """
        # 1. 基础search_k
        search_k = self.base_search_k
        
        # 2. 召回率反馈调整（优先考虑，如果提供当前召回率）
        recall_adjustment_factor = 1.0
        if current_recall is not None:
            if current_recall < self.min_recall:
                # 召回率过低，大幅增加search_k
                recall_adjustment_factor = 1 + (self.min_recall - current_recall) * 2
            elif current_recall < self.target_recall:
                # 召回率低于目标，适度增加search_k
                recall_adjustment_factor = 1 + (self.target_recall - current_recall) * 0.5
            elif current_recall > self.target_recall + 0.1:
                # 召回率高于目标，可以适度减少search_k
                # balanced策略：适度减少（因为qps_weight=0.5，平衡考虑）
                # qps_optimized策略：更激进地减少（因为qps_weight=0.7）
                # recall_optimized策略：不减少（因为qps_weight=0.3，优先召回率）
                if self.qps_weight >= 0.5:
                    # balanced和qps_optimized都可以减少
                    # 当recall很高时（如>0.9），应该更激进地减少
                    if current_recall > 0.9:
                        # 召回率非常高，大幅减少数据量因子和n_probe因子的影响
                        excess_recall = current_recall - self.target_recall - 0.1
                        reduction_factor = 0.5 if self.qps_weight > 0.5 else 0.3  # qps_optimized更激进
                        recall_adjustment_factor = 1 - excess_recall * reduction_factor
                        # 限制减少幅度，但允许更大幅度的减少
                        recall_adjustment_factor = max(0.5, recall_adjustment_factor)
                    else:
                        # 召回率略高于目标，适度减少
                        excess_recall = current_recall - self.target_recall - 0.1
                        reduction_factor = 0.3 if self.qps_weight > 0.5 else 0.15
                        recall_adjustment_factor = 1 - excess_recall * reduction_factor
                        recall_adjustment_factor = max(0.8, recall_adjustment_factor)
        
        # 3. n_probe因子（n_probe越大，search_k也应该相应增加，但受召回率影响，更保守）
        # 注意：不单独考虑数据量因子，因为n_probe已经根据数据量动态调整了
        probe_factor = n_probe / self.base_n_probe
        # 如果召回率很高，抑制n_probe因子的影响
        if current_recall is not None and current_recall > self.target_recall + 0.1:
            if current_recall > 0.9:
                # 召回率非常高，大幅抑制n_probe因子
                probe_factor = 1 + (probe_factor - 1) * 0.2
            else:
                # 召回率略高，适度抑制
                probe_factor = 1 + (probe_factor - 1) * 0.5
        search_k = int(search_k * probe_factor)
        
        # 5. 应用召回率调整因子
        search_k = int(search_k * recall_adjustment_factor)
        
        # 6. 平均聚类大小因子（聚类越大，需要更多候选，但更保守）
        if avg_cluster_size > 1000:
            cluster_factor = 1 + (avg_cluster_size - 1000) / 3000
            search_k = int(search_k * cluster_factor)
        
        # 6. 限制在合理范围内
        search_k = max(self.min_search_k, min(search_k, self.max_search_k))
        
        return search_k
    
    def get_adaptive_params(
        self,
        data_size: int,
        n_fine_clusters: int,
        avg_cluster_size: float,
        max_cluster_size: int,
        current_recall: Optional[float] = None,
        current_qps: Optional[float] = None,
        initial_data_size: int = 10000
    ) -> Tuple[int, int]:
        """
        获取自适应参数（n_probe和search_k）
        
        这是主要的接口方法，综合考虑所有因素
        
        Args:
            data_size: 当前数据量
            n_fine_clusters: 细粒度聚类数
            avg_cluster_size: 平均聚类大小
            max_cluster_size: 最大聚类大小
            current_recall: 当前召回率（可选，用于反馈调整）
            current_qps: 当前QPS（可选，用于反馈调整）
            initial_data_size: 初始数据量
            
        Returns:
            (n_probe, search_k) 元组
        """
        # 计算n_probe
        n_probe = self.get_dynamic_n_probe(
            data_size=data_size,
            n_fine_clusters=n_fine_clusters,
            avg_cluster_size=avg_cluster_size,
            max_cluster_size=max_cluster_size,
            initial_data_size=initial_data_size
        )
        
        # 计算search_k（使用刚计算的n_probe）
        search_k = self.get_dynamic_search_k(
            data_size=data_size,
            n_probe=n_probe,
            avg_cluster_size=avg_cluster_size,
            current_recall=current_recall,
            initial_data_size=initial_data_size
        )
        
        # 记录历史（用于后续分析）
        if current_recall is not None:
            self.recall_history.append(current_recall)
        if current_qps is not None:
            self.qps_history.append(current_qps)
        self.param_history.append((n_probe, search_k))
        
        return n_probe, search_k
    
    def update_weights(
        self,
        target_recall: Optional[float] = None,
        qps_weight: Optional[float] = None,
        recall_weight: Optional[float] = None
    ):
        """
        更新权重参数（用于运行时调整策略）
        
        Args:
            target_recall: 新的目标召回率
            qps_weight: 新的QPS权重
            recall_weight: 新的召回率权重
        """
        if target_recall is not None:
            self.target_recall = target_recall
        if qps_weight is not None:
            self.qps_weight = qps_weight
        if recall_weight is not None:
            self.recall_weight = recall_weight
    
    def get_statistics(self) -> Dict:
        """
        获取参数调整统计信息
        
        Returns:
            包含统计信息的字典
        """
        stats = {
            'total_adjustments': len(self.param_history),
            'avg_n_probe': np.mean([p[0] for p in self.param_history]) if self.param_history else 0,
            'avg_search_k': np.mean([p[1] for p in self.param_history]) if self.param_history else 0,
            'max_n_probe': max([p[0] for p in self.param_history]) if self.param_history else 0,
            'max_search_k': max([p[1] for p in self.param_history]) if self.param_history else 0,
        }
        
        if self.recall_history:
            stats['avg_recall'] = np.mean(self.recall_history)
            stats['min_recall'] = np.min(self.recall_history)
            stats['max_recall'] = np.max(self.recall_history)
        
        if self.qps_history:
            stats['avg_qps'] = np.mean(self.qps_history)
            stats['min_qps'] = np.min(self.qps_history)
            stats['max_qps'] = np.max(self.qps_history)
        
        return stats
    
    def reset_history(self):
        """重置历史记录"""
        self.recall_history = []
        self.qps_history = []
        self.param_history = []


def create_balanced_manager(
    base_n_probe: int = 10,
    base_search_k: int = 5000,
    target_recall: float = 0.5
) -> DynamicParameterManager:
    """
    创建一个平衡的（召回率和QPS各占50%权重）参数管理器
    
    Args:
        base_n_probe: 基础n_probe值
        base_search_k: 基础search_k值
        target_recall: 目标召回率
        
    Returns:
        DynamicParameterManager实例
    """
    return DynamicParameterManager(
        base_n_probe=base_n_probe,
        base_search_k=base_search_k,
        target_recall=target_recall,
        qps_weight=0.5,
        recall_weight=0.5
    )


def create_recall_optimized_manager(
    base_n_probe: int = 10,
    base_search_k: int = 5000,
    target_recall: float = 0.7
) -> DynamicParameterManager:
    """
    创建一个召回率优化的参数管理器（更重视召回率）
    
    Args:
        base_n_probe: 基础n_probe值
        base_search_k: 基础search_k值
        target_recall: 目标召回率
        
    Returns:
        DynamicParameterManager实例
    """
    return DynamicParameterManager(
        base_n_probe=base_n_probe,
        base_search_k=base_search_k,
        target_recall=target_recall,
        qps_weight=0.3,
        recall_weight=0.7
    )


def create_qps_optimized_manager(
    base_n_probe: int = 10,
    base_search_k: int = 5000,
    target_recall: float = 0.4
) -> DynamicParameterManager:
    """
    创建一个QPS优化的参数管理器（更重视QPS）
    
    Args:
        base_n_probe: 基础n_probe值
        base_search_k: 基础search_k值
        target_recall: 目标召回率
        
    Returns:
        DynamicParameterManager实例
    """
    return DynamicParameterManager(
        base_n_probe=base_n_probe,
        base_search_k=base_search_k,
        target_recall=target_recall,
        qps_weight=0.7,
        recall_weight=0.3
    )

