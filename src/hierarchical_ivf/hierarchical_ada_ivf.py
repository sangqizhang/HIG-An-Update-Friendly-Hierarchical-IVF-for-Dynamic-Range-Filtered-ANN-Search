"""
层次化Ada-IVF实现
基于论文：将Ada-IVF扩展到分层IVF索引
- 两级结构：细粒度聚类（底层）和粗粒度聚类（顶层）
- 每个层次使用独立的Ada-IVF更新管理器
- 底层质心变化通过质心更新影响顶层，但影响有限
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import pickle
import time

from .ada_ivf import AdaIVFIndex, ClusterStats


class HierarchicalAdaIVFIndex:
    """
    层次化Ada-IVF索引类
    
    两级结构：
    - 底层（细粒度）：向量被分组成细粒度聚类，更多聚类
    - 顶层（粗粒度）：细粒度聚类的质心进一步聚集成粗聚类，较少聚类
    
    每个层次使用独立的Ada-IVF更新管理器
    """
    
    def __init__(
        self,
        n_fine_clusters: int = 100,  # 底层细粒度聚类数
        n_coarse_clusters: int = 10,  # 顶层粗粒度聚类数
        n_probe: int = 10,
        max_cluster_size: int = 10000,
        recluster_threshold: float = 0.3,
        centroid_update_threshold: float = 0.1  # 质心更新阈值，控制对顶层的影响
    ):
        """
        初始化层次化Ada-IVF索引
        
        Args:
            n_fine_clusters: 底层细粒度聚类数量
            n_coarse_clusters: 顶层粗粒度聚类数量
            n_probe: 搜索时探测的聚类中心数量
            max_cluster_size: 单个聚类的最大向量数量
            recluster_threshold: 重聚类质量阈值
            centroid_update_threshold: 质心更新阈值，当底层质心变化超过此值时才更新顶层
        """
        self.n_fine_clusters = n_fine_clusters
        self.n_coarse_clusters = n_coarse_clusters
        self.n_probe = n_probe
        self.max_cluster_size = max_cluster_size
        self.recluster_threshold = recluster_threshold
        self.centroid_update_threshold = centroid_update_threshold
        
        # 底层索引（细粒度，Ada-IVF）
        self.fine_index = AdaIVFIndex(
            n_clusters=n_fine_clusters,
            n_probe=n_probe,
            max_cluster_size=max_cluster_size,
            recluster_threshold=recluster_threshold
        )
        
        # 顶层索引（粗粒度，Ada-IVF）
        self.coarse_index = AdaIVFIndex(
            n_clusters=n_coarse_clusters,
            n_probe=min(n_probe, n_coarse_clusters),
            max_cluster_size=max_cluster_size * 10,  # 顶层聚类可以更大
            recluster_threshold=recluster_threshold
        )
        
        self.dimension = None
        self.is_trained = False
        
        # 层次关系：记录每个细粒度聚类属于哪个粗粒度聚类
        self.fine_to_coarse = {}  # {fine_cluster_id: coarse_cluster_id}
        
        # 记录底层质心的历史值，用于检测变化
        self.fine_centroids_history = None

        # 为了避免每次插入都在底层向量字典上做 O(N) 的 max(key) 扫描，
        # 这里维护一个简单的自增 ID 计数器，用于给“未显式指定ID”的向量分配连续ID。
        # 这可以显著降低插入阶段的时间复杂度，避免随着数据量增长而出现插入时间急剧上升。
        self._next_vector_id: int = 0

        # 结构示例：
        # {
        #   "searched_fine_clusters": int,
        #   "scanned_vectors": int,                 # 扫描（参与距离计算）的向量数（范围过滤前）
        #   "range_filtered_candidates": int,        # 范围过滤后候选数
        #   "budgeted_candidates": int,              # early_stop_threshold 截断后候选数
        #   "returned_k": int
        # }
        self.last_search_stats: Dict[str, int] = {}
        
    def train(self, vectors: np.ndarray):
        """
        训练层次化索引
        
        1. 先训练底层（细粒度聚类）
        2. 使用底层质心训练顶层（粗粒度聚类）
        """
        if len(vectors) == 0:
            raise ValueError("训练向量不能为空")
            
        self.dimension = vectors.shape[1]
        print(f"开始训练层次化Ada-IVF索引")
        print(f"  底层细粒度聚类数: {self.n_fine_clusters}")
        print(f"  顶层粗粒度聚类数: {self.n_coarse_clusters}")
        
        # 1. 训练底层（细粒度）
        print(f"\n1. 训练底层（细粒度聚类）...")
        self.fine_index.train(vectors)
        
        # 2. 使用底层质心训练顶层（粗粒度）
        print(f"\n2. 训练顶层（粗粒度聚类，基于底层质心）...")
        fine_centroids = self.fine_index.centroids
        self.coarse_index.train(fine_centroids)
        
        # 建立层次关系：每个细粒度聚类属于哪个粗粒度聚类
        self._update_hierarchy()
        
        # 记录底层质心历史
        self.fine_centroids_history = fine_centroids.copy()
        
        self.is_trained = True
        print("\n层次化Ada-IVF索引训练完成")
        
    def _update_hierarchy(self):
        """更新层次关系：将细粒度聚类映射到粗粒度聚类"""
        self.fine_to_coarse = {}
        
        # 计算每个细粒度聚类中心到粗粒度聚类中心的距离
        fine_centroids = self.fine_index.centroids
        coarse_centroids = self.coarse_index.centroids
        
        # 使用FAISS加速（如果可用）
        from .ada_ivf import FAISS_AVAILABLE
        if FAISS_AVAILABLE:
            try:
                import faiss
                fine_centroids_f32 = fine_centroids.astype(np.float32)
                coarse_centroids_f32 = coarse_centroids.astype(np.float32)
                index = faiss.IndexFlatL2(coarse_centroids_f32.shape[1])
                index.add(coarse_centroids_f32)
                _, I = index.search(fine_centroids_f32, 1)
                nearest = I.reshape(-1)
            except:
                # 回退到numpy
                distances = np.linalg.norm(
                    fine_centroids[:, np.newaxis, :] - coarse_centroids[np.newaxis, :, :],
                    axis=2
                )
                nearest = np.argmin(distances, axis=1)
        else:
            distances = np.linalg.norm(
                fine_centroids[:, np.newaxis, :] - coarse_centroids[np.newaxis, :, :],
                axis=2
            )
            nearest = np.argmin(distances, axis=1)
        
        # 每个细粒度聚类属于最近的粗粒度聚类
        for fine_id in range(self.n_fine_clusters):
            coarse_id = int(nearest[fine_id])
            self.fine_to_coarse[fine_id] = coarse_id
    
    def _update_coarse_from_fine(self):
        """
        根据底层质心变化更新顶层
        只有当底层质心变化超过阈值时才更新，影响有限
        """
        if self.fine_centroids_history is None:
            return
        
        current_fine_centroids = self.fine_index.centroids
        
        # 计算质心变化
        centroid_changes = np.linalg.norm(
            current_fine_centroids - self.fine_centroids_history,
            axis=1
        )
        
        # 检查是否有显著变化
        max_change = np.max(centroid_changes)
        mean_change = np.mean(centroid_changes)
        
        # 只有当变化超过阈值时才更新顶层
        if max_change > self.centroid_update_threshold or mean_change > self.centroid_update_threshold * 0.5:
            print(f"  检测到底层质心变化 (最大: {max_change:.4f}, 平均: {mean_change:.4f})，更新顶层...")
            
            # 使用新的底层质心重新训练顶层（但使用较少的迭代，影响有限）
            self._update_hierarchy()
            
            # 更新粗粒度聚类的质心（基于其包含的细粒度聚类质心）
            for coarse_id in range(self.n_coarse_clusters):
                # 找到属于该粗粒度聚类的所有细粒度聚类
                fine_ids = [fid for fid, cid in self.fine_to_coarse.items() if cid == coarse_id]
                
                if len(fine_ids) > 0:
                    # 使用这些细粒度聚类质心的平均值作为粗粒度聚类质心
                    fine_centroids_subset = current_fine_centroids[fine_ids]
                    new_coarse_centroid = fine_centroids_subset.mean(axis=0)
                    self.coarse_index.centroids[coarse_id] = new_coarse_centroid
            
            # 更新粗粒度索引的边界
            self.coarse_index._update_cluster_bounds()
            
            # 更新历史记录
            self.fine_centroids_history = current_fine_centroids.copy()
        else:
            # 变化很小，只更新层次关系，不更新粗粒度质心
            self._update_hierarchy()
    
    def add(self, vectors: np.ndarray, ids: Optional[List] = None, auto_recluster: bool = True, scalars: Optional[np.ndarray] = None):
        """
        动态插入向量到层次化索引（支持增量式更新）
        
        1. 向量插入到底层（细粒度）
        2. 底层使用独立的Ada-IVF更新管理器进行自适应维护
        3. 如果底层质心变化显著，更新顶层（影响有限）
        
        Args:
            vectors: 要添加的向量
            ids: 可选的向量ID列表
            auto_recluster: 是否自动触发重聚类（增量式更新）
            scalars: 可选的属性值数组，用于范围过滤
        """
        if not self.is_trained:
            raise ValueError("索引尚未训练，请先调用train()方法")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度不匹配：期望 {self.dimension}，得到 {vectors.shape[1]}")
        
        n_vectors = vectors.shape[0]
        if ids is None:
            # 使用内部的自增ID分配器，避免对已有ID做全量扫描
            start_id = self._next_vector_id
            ids = list(range(start_id, start_id + n_vectors))
            self._next_vector_id += n_vectors
        else:
            # 如果外部显式传入ID，确保内部计数器不落后于最大ID
            try:
                max_id = max(ids)
                if max_id + 1 > self._next_vector_id:
                    self._next_vector_id = max_id + 1
            except ValueError:
                # 空ID列表，忽略
                pass
        
        # 1. 插入到底层（细粒度），支持属性值
        self.fine_index.add(vectors, ids=ids, auto_recluster=auto_recluster, scalars=scalars)
        
        # 2. 检查底层质心是否变化，如果变化显著则更新顶层
        if auto_recluster:
            self._update_coarse_from_fine()
        
        print(f"已添加 {n_vectors} 个向量到层次化Ada-IVF索引")
    
    def _get_dynamic_n_probe(self, base_n_probe: int, use_mid_filtering: bool = False) -> int:
        """
        根据当前数据量、聚类大小和聚类数动态调整n_probe
        QPS优化：对于中过滤，使用更保守的n_probe策略
        
        Args:
            base_n_probe: 基础n_probe值
            use_mid_filtering: 是否使用中过滤策略（影响n_probe调整策略）
            
        Returns:
            调整后的n_probe值
        """
        n_vectors = len(self.fine_index.vectors)
        if n_vectors == 0:
            return base_n_probe
        
        # 计算平均聚类大小
        cluster_sizes = [len(self.fine_index.inverted_lists[cid]) 
                        for cid in range(self.n_fine_clusters)]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        
        # QPS优先版本：
        # - 对于 use_mid_filtering=True 的场景，不再随数据量和聚类大小增长，始终使用传入的 base_n_probe，
        #   这样可以把顶层/粗粒度探测的成本固定在一个较小值，便于控制整体复杂度。
        # - 非 mid_filtering 场景仍保留原来的自适应增长逻辑。
        min_n_probe = 3
        if use_mid_filtering:
            return max(base_n_probe, min_n_probe)
        
        # 非中过滤：保留原有的“随数据量/聚类大小增大”的动态策略
        max_n_probe_ratio = 0.50
        max_n_probe = max(min_n_probe, int(self.n_coarse_clusters * max_n_probe_ratio))
        n_probe = max(base_n_probe, min_n_probe)
        
        if n_vectors > 100000:
            data_factor = 1.0 + (n_vectors - 100000) / 100000 * 0.15
            n_probe = int(n_probe * data_factor)
        elif n_vectors > 50000:
            data_factor = 1.0 + (n_vectors - 50000) / 100000 * 0.10
            n_probe = int(n_probe * data_factor)
        elif n_vectors > 20000:
            data_factor = 1.0 + (n_vectors - 20000) / 100000 * 0.05
            n_probe = int(n_probe * data_factor)
        
        if avg_cluster_size > 500:
            cluster_factor = 1.0 + (avg_cluster_size - 500) / 2000 * 0.05
            n_probe = int(n_probe * cluster_factor)
        
        return min(n_probe, max_n_probe)
    
    def _get_dynamic_search_k(self, base_search_k: int, k: int = 10, use_mid_filtering: bool = False) -> int:
        """
        根据当前数据量动态调整search_k（early_stop_threshold）
        QPS优化：基础值设小一点，随数据量逐渐增大
        
        Args:
            base_search_k: 基础search_k值（early_stop_threshold的基础值）
            k: 查询的k值（用于计算相对阈值）
            use_mid_filtering: 是否使用中过滤策略
            
        Returns:
            调整后的search_k值
        """
        n_vectors = len(self.fine_index.vectors)
        if n_vectors == 0:
            # 初始值：使用k的倍数或基础值（降低以提升QPS）
            if base_search_k is None or base_search_k == 0:
                return min(k * 1.5, 1500) if use_mid_filtering else k * 2.5
            return base_search_k
        
        # 动态调整search_k
        # 基础值较小，随数据量逐渐增大（降低以提升QPS）
        min_search_k = min(k * 1.5, 1500) if use_mid_filtering else k * 2.5
        if base_search_k is None or base_search_k == 0:
            search_k = min_search_k
        else:
            search_k = max(base_search_k, min_search_k)
        
        # 根据数据量逐渐增大search_k（降低增长率以提升QPS）
        if n_vectors > 100000:
            data_factor = 1.0 + (n_vectors - 100000) / 100000 * 0.15
            search_k = int(search_k * data_factor)
        elif n_vectors > 50000:
            data_factor = 1.0 + (n_vectors - 50000) / 100000 * 0.12
            search_k = int(search_k * data_factor)
        elif n_vectors > 20000:
            data_factor = 1.0 + (n_vectors - 20000) / 100000 * 0.08
            search_k = int(search_k * data_factor)
        
        # 中过滤时设置上限（降低以提升QPS）
        if use_mid_filtering:
            max_search_k = 3000
        else:
            max_search_k = 6000

        # 确保返回整数，避免在numpy.argpartition等调用中出现类型错误
        search_k = int(max(1, min(search_k, max_search_k)))
        return search_k
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None,
        scalar_range: Optional[Tuple[int, int]] = None,
        use_hierarchy: bool = True,
        early_stop_threshold: Optional[int] = None,
        use_mid_filtering: bool = False  # 是否使用中过滤策略
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        层次化搜索（支持范围混合过滤）
        
        Args:
            query: 查询向量
            k: 返回的最近邻数量
            radius: 可选的范围查询半径
            scalar_range: 可选的属性值范围 (min, max)，先范围过滤再k-NN
            use_hierarchy: 是否使用层次化搜索策略
            early_stop_threshold: 提前终止阈值，找到这么多结果后提前停止（默认k*3）
            
        Returns:
            (distances, indices): 距离和索引的元组
        """
        if not self.is_trained:
            raise ValueError("索引未训练")
            
        if query.shape[0] != self.dimension:
            raise ValueError(f"查询向量维度不匹配：期望 {self.dimension}，得到 {query.shape[0]}")
        
        # 如果提供了early_stop_threshold，使用它；否则根据数据量动态调整
        if early_stop_threshold is None:
            # 使用动态search_k作为early_stop_threshold
            early_stop_threshold = self._get_dynamic_search_k(None, k=k, use_mid_filtering=use_mid_filtering)
        else:
            # 否则会出现 search_k=5000 但内部 cap 到 3000 的情况，导致召回异常偏低。
            early_stop_threshold = int(max(1, early_stop_threshold))
        
        if use_hierarchy:
            # 层次化搜索：先在顶层找到候选粗粒度聚类，再在底层搜索
            dynamic_n_probe = self._get_dynamic_n_probe(self.n_probe, use_mid_filtering=use_mid_filtering)
            
            from .ada_ivf import FAISS_AVAILABLE
            if FAISS_AVAILABLE:
                try:
                    import faiss
                    query_f32 = query.astype(np.float32).reshape(1, -1)
                    centroids_f32 = self.coarse_index.centroids.astype(np.float32)
                    index = faiss.IndexFlatL2(centroids_f32.shape[1])
                    index.add(centroids_f32)
                    _, I = index.search(query_f32, int(min(dynamic_n_probe, self.n_coarse_clusters)))
                    top_coarse_clusters = I.flatten().astype(np.int64)
                except:
                    # 回退：计算平方距离，使用argpartition优化
                    diff = self.coarse_index.centroids - query
                    distances_sq_to_coarse = np.sum(diff * diff, axis=1)
                    if len(distances_sq_to_coarse) > dynamic_n_probe:
                        kth = max(0, dynamic_n_probe - 1)
                        top_coarse_indices = np.argpartition(distances_sq_to_coarse, kth)[:dynamic_n_probe]
                        top_coarse_indices = top_coarse_indices[np.argsort(distances_sq_to_coarse[top_coarse_indices])]
                    else:
                        top_coarse_indices = np.argsort(distances_sq_to_coarse)
                    top_coarse_clusters = top_coarse_indices
            else:
                # 回退：计算平方距离，使用argpartition优化
                diff = self.coarse_index.centroids - query
                distances_sq_to_coarse = np.sum(diff * diff, axis=1)
                if len(distances_sq_to_coarse) > dynamic_n_probe:
                    kth = max(0, dynamic_n_probe - 1)
                    top_coarse_indices = np.argpartition(distances_sq_to_coarse, kth)[:dynamic_n_probe]
                    top_coarse_indices = top_coarse_indices[np.argsort(distances_sq_to_coarse[top_coarse_indices])]
                else:
                    top_coarse_indices = np.argsort(distances_sq_to_coarse)
                top_coarse_clusters = top_coarse_indices
            
            candidate_fine_clusters = set()
            coarse_to_fine = defaultdict(list)
            for fine_id, coarse_id in self.fine_to_coarse.items():
                coarse_to_fine[coarse_id].append(fine_id)
            # 使用反向索引快速查找
            for coarse_id in top_coarse_clusters:
                if coarse_id in coarse_to_fine:
                    candidate_fine_clusters.update(coarse_to_fine[coarse_id])

            # 重要修正（召回优先）：仅依赖 coarse->fine 映射在某些训练/更新阶段会漏掉真正近邻的 fine cluster。
            # 这里增加一个“细粒度回退探测”：直接从所有 fine centroids 里选一批最近的 fine clusters，补齐候选集合。
            fine_centroids = self.fine_index.centroids
            if fine_centroids is not None and fine_centroids.shape[0] == self.n_fine_clusters:
                # QPS优先版本：减小细粒度回退探测规模
                # - 只按 dynamic_n_probe 的一个小倍数扩展，且设置较小下限
                # - 避免随着 n_fine_clusters 增大而线性放大
                fallback_fine_probe = int(min(self.n_fine_clusters, max(30, dynamic_n_probe * 3)))
                # 计算所有 fine centroids 到 query 的平方距离
                diff_all = fine_centroids - query
                dist_sq_all = np.sum(diff_all * diff_all, axis=1)
                if self.n_fine_clusters > fallback_fine_probe:
                    kth = max(0, fallback_fine_probe - 1)
                    top_fine = np.argpartition(dist_sq_all, kth)[:fallback_fine_probe]
                else:
                    top_fine = np.arange(self.n_fine_clusters, dtype=np.int64)
                candidate_fine_clusters.update(top_fine.tolist())
            
            # 关键：边查询边过滤时，必须按距离排序，否则提前终止会错过更近的向量
            fine_centroids = self.fine_index.centroids
            candidate_fine_clusters_list = list(candidate_fine_clusters)
            if candidate_fine_clusters_list:  # 空列表判断更快
                # 批量计算：提取候选聚类的中心点，使用numpy向量化计算（足够快）
                candidate_centroids = fine_centroids[candidate_fine_clusters_list]
                diff = candidate_centroids - query
                distances_sq = np.sum(diff * diff, axis=1)
                sorted_indices = np.argsort(distances_sq)
                candidate_fine_clusters = [candidate_fine_clusters_list[i] for i in sorted_indices]
            else:
                candidate_fine_clusters = []
            
            # 3. 在候选细粒度聚类中搜索（向量化优化 + 提前终止）
            # 中过滤策略：边收集边计算距离边过滤
            if use_mid_filtering and scalar_range is not None and len(self.fine_index.vector_scalars) > 0:
                # 中过滤：对每个聚类，计算距离并过滤，然后合并结果
                # 智能提前终止：当收集到足够候选且已搜索足够聚类时提前终止
                range_min, range_max = scalar_range
                all_distances_sq_parts = []  # 收集各部分的平方距离数组
                all_indices_parts = []  # 收集各部分的索引数组
                total_candidates = 0  # 跟踪已收集的候选数量
                searched_clusters = 0  # 跟踪已搜索的聚类数量
                scanned_vectors = 0  # 跟踪参与距离计算的向量数（范围过滤前）
                # QPS优先版本：只强制搜索一部分候选聚类，其余依赖早停控制整体开销
                # 纯py基线大约在 Recall≈90% 时 QPS 还能保持在几十，这里以 ~50% 候选簇作为经验值。
                min_clusters_to_search = max(1, int(len(candidate_fine_clusters) * 0.5))
                vectors_dict = self.fine_index.vectors
                scalars_dict = self.fine_index.vector_scalars
                
                for fine_id in candidate_fine_clusters:
                    # 智能提前终止：当收集到足够候选且已搜索足够聚类时提前终止
                    # 这样可以平衡QPS和召回率
                    if (early_stop_threshold is not None and total_candidates >= early_stop_threshold and 
                        searched_clusters >= min_clusters_to_search):
                        break
                    
                    vector_ids = self.fine_index.inverted_lists[fine_id]
                    if not vector_ids:  # 空列表判断更快
                        continue

                    scanned_vectors += len(vector_ids)
                    
                    vector_ids_array = np.asarray(vector_ids, dtype=np.int64)
                    
                    # 获取该聚类的向量
                    vectors_list = [vectors_dict[vid] for vid in vector_ids]  # 使用缓存的引用
                    vectors_batch = np.array(vectors_list)
                    
                    # 如果丢掉 indices 并假设与原顺序对齐，会导致距离/ID错位，召回率极低。
                    # 这里统一：若用FAISS，则用返回的 indices 重排 vector_ids/scalars；否则用numpy保持原顺序。
                    from .ada_ivf import FAISS_AVAILABLE
                    use_faiss = bool(FAISS_AVAILABLE and len(vectors_batch) > 5)
                    if use_faiss:
                        try:
                            import faiss
                            query_f32 = query.astype(np.float32).reshape(1, -1)
                            vectors_f32 = vectors_batch.astype(np.float32)
                            fidx = faiss.IndexFlatL2(vectors_f32.shape[1])
                            fidx.add(vectors_f32)
                            D, I = fidx.search(query_f32, len(vectors_f32))  # sorted by distance
                            distances_sq_batch = D.flatten().astype(np.float64)
                            order = I.flatten().astype(np.int64)
                            ordered_vector_ids_array = vector_ids_array[order]
                        except:
                            use_faiss = False
                    if not use_faiss:
                        diff = vectors_batch - query
                        distances_sq_batch = np.sum(diff * diff, axis=1)
                        ordered_vector_ids_array = vector_ids_array
                    
                    # 获取属性值并过滤（使用缓存的引用）
                    # 注意：如果使用了FAISS排序，这里必须与 ordered_vector_ids_array 对齐
                    scalars_batch = np.array([scalars_dict.get(int(vid), -1) for vid in ordered_vector_ids_array], dtype=np.float64)
                    range_mask = (scalars_batch >= range_min) & (scalars_batch <= range_max)
                    
                    # 只保留满足范围条件的向量（保留平方距离，不计算sqrt）
                    filtered_distances_sq = distances_sq_batch[range_mask]
                    filtered_indices = ordered_vector_ids_array[range_mask]
                    
                    # 只有当有满足条件的向量时才添加（避免添加空数组）
                    if len(filtered_indices) > 0:
                        all_distances_sq_parts.append(filtered_distances_sq)
                        all_indices_parts.append(filtered_indices)
                        total_candidates += len(filtered_indices)
                    
                    searched_clusters += 1  # 无论是否有满足条件的向量，都算搜索了一个聚类
                
                if not all_indices_parts:  # 空列表判断更快
                    self.last_search_stats = {
                        "searched_fine_clusters": int(searched_clusters),
                        "scanned_vectors": int(scanned_vectors),
                        "range_filtered_candidates": 0,
                        "budgeted_candidates": 0,
                        "returned_k": 0,
                    }
                    return np.array([]), np.array([])
                
                # 简化条件判断：只有一个元素时直接取，否则拼接
                n_parts = len(all_distances_sq_parts)
                if n_parts == 1:
                    all_distances_sq_array = all_distances_sq_parts[0]
                    all_indices_array = all_indices_parts[0]
                else:
                    all_distances_sq_array = np.concatenate(all_distances_sq_parts)
                    all_indices_array = np.concatenate(all_indices_parts)
                
                # 这样可以避免处理过多候选，同时保证搜索了所有聚类
                array_len = len(all_distances_sq_array)
                if early_stop_threshold is not None and array_len > early_stop_threshold:
                    # 使用argpartition找到最近的early_stop_threshold个候选
                    kth = max(0, early_stop_threshold - 1)
                    top_candidate_indices = np.argpartition(all_distances_sq_array, kth)[:early_stop_threshold]
                    top_candidate_indices = top_candidate_indices[np.argsort(all_distances_sq_array[top_candidate_indices])]
                    all_distances_sq_array = all_distances_sq_array[top_candidate_indices]
                    all_indices_array = all_indices_array[top_candidate_indices]
                    array_len = len(all_distances_sq_array)

                budgeted_candidates = int(array_len)
                
                if array_len > k and array_len > 100:  # 大于k且大于阈值才用argpartition
                    kth = max(0, k - 1)
                    top_k_indices = np.argpartition(all_distances_sq_array, kth)[:k]
                    top_k_indices = top_k_indices[np.argsort(all_distances_sq_array[top_k_indices])]
                else:
                    top_k_indices = np.argsort(all_distances_sq_array)[:k] if array_len > k else np.argsort(all_distances_sq_array)
                
                # 只对最终top-k计算sqrt
                distances_all = np.sqrt(all_distances_sq_array[top_k_indices])
                all_vector_ids_array = all_indices_array[top_k_indices]

                self.last_search_stats = {
                    "searched_fine_clusters": int(searched_clusters),
                    "scanned_vectors": int(scanned_vectors),
                    "range_filtered_candidates": int(total_candidates),
                    "budgeted_candidates": int(budgeted_candidates),
                    "returned_k": int(len(all_vector_ids_array)),
                }
            else:
                # 预过滤或没有范围过滤：先收集，再计算
                candidate_vector_ids = []
                for fine_id in candidate_fine_clusters:
                    # 取消距离上界提前终止：确保遍历所有候选聚类以保证召回率
                    
                    vector_ids = self.fine_index.inverted_lists[fine_id]
                    if not vector_ids:  # 空列表判断更快
                        continue
                    
                    # 预过滤：在收集时就过滤
                    if scalar_range is not None and len(self.fine_index.vector_scalars) > 0:
                        range_min, range_max = scalar_range
                        scalars_batch = np.array([self.fine_index.vector_scalars.get(vid, -1) for vid in vector_ids])
                        range_mask = (scalars_batch >= range_min) & (scalars_batch <= range_max)
                        filtered_vector_ids = [vid for vid, mask in zip(vector_ids, range_mask) if mask]
                        candidate_vector_ids.extend(filtered_vector_ids)
                    else:
                        candidate_vector_ids.extend(vector_ids)
                    
                    # 取消提前终止：确保遍历所有候选聚类以保证召回率
                
                # 批量获取向量
                if candidate_vector_ids:  # 空列表判断更快
                    batch_size = 10000
                    if len(candidate_vector_ids) > batch_size:
                        all_vectors_list = []
                        for i in range(0, len(candidate_vector_ids), batch_size):
                            batch_ids = candidate_vector_ids[i:i+batch_size]
                            batch_vectors = [self.fine_index.vectors[vid] for vid in batch_ids]
                            all_vectors_list.extend(batch_vectors)
                        all_vectors_array = np.array(all_vectors_list)
                    else:
                        all_vectors_list = [self.fine_index.vectors[vid] for vid in candidate_vector_ids]
                        all_vectors_array = np.array(all_vectors_list)
                    all_vector_ids = candidate_vector_ids
                else:
                    all_vectors_array = np.array([])
                    all_vector_ids = []
                
                if not all_vector_ids:  # 空列表判断更快
                    self.last_search_stats = {
                        "searched_fine_clusters": int(len(candidate_fine_clusters)),
                        "scanned_vectors": 0,
                        "range_filtered_candidates": 0,
                        "budgeted_candidates": 0,
                        "returned_k": 0,
                    }
                    return np.array([]), np.array([])
                
                all_vector_ids_array = np.array(all_vector_ids)
                
                # 计算距离
                from .ada_ivf import FAISS_AVAILABLE
                if FAISS_AVAILABLE and len(all_vectors_array) > 10:
                    try:
                        import faiss
                        query_f32 = query.astype(np.float32).reshape(1, -1)
                        vectors_f32 = all_vectors_array.astype(np.float32)
                        index = faiss.IndexFlatL2(vectors_f32.shape[1])
                        index.add(vectors_f32)
                        distances_sq_faiss, _ = index.search(query_f32, len(vectors_f32))
                        distances_sq = distances_sq_faiss.flatten().astype(np.float64)
                    except:
                        diff = all_vectors_array - query
                        distances_sq = np.sum(diff * diff, axis=1)
                else:
                    diff = all_vectors_array - query
                    distances_sq = np.sum(diff * diff, axis=1)
                
                # 排序并返回top-k
                if len(distances_sq) > k:
                    kth = max(0, k - 1)
                    top_k_indices_sq = np.argpartition(distances_sq, kth)[:k]
                    distances_all = np.sqrt(distances_sq[top_k_indices_sq])
                    all_vector_ids_array = all_vector_ids_array[top_k_indices_sq]
                    sorted_indices = np.argsort(distances_all)
                    distances_all = distances_all[sorted_indices]
                    all_vector_ids_array = all_vector_ids_array[sorted_indices]
                else:
                    distances_all = np.sqrt(distances_sq)

                self.last_search_stats = {
                    "searched_fine_clusters": int(len(candidate_fine_clusters)),
                    "scanned_vectors": int(len(all_vector_ids_array)),
                    "range_filtered_candidates": int(len(all_vector_ids_array)),
                    "budgeted_candidates": int(len(all_vector_ids_array)),
                    "returned_k": int(min(k, len(all_vector_ids_array))),
                }
            
            # 范围查询过滤（如果有radius）
            if radius is not None:
                valid_mask = distances_all <= radius
                if not np.any(valid_mask):
                    return np.array([]), np.array([])
                distances_all = distances_all[valid_mask]
                all_vector_ids_array = all_vector_ids_array[valid_mask]
            
            if len(distances_all) == 0:  # 数组用len，空数组检查
                self.last_search_stats = {
                    **(self.last_search_stats or {}),
                    "returned_k": 0,
                }
                return np.array([]), np.array([])
            
            # 返回top-k
            if len(distances_all) <= k:
                return distances_all, all_vector_ids_array
            
            return distances_all[:k], all_vector_ids_array[:k]
        else:
            # 非层次化搜索：直接在底层搜索（使用优化）
            return self.fine_index.search(query, k=k, radius=radius, scalar_range=scalar_range, use_mid_filtering=use_mid_filtering,
                                         early_stop_threshold=early_stop_threshold)
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None,
        scalar_ranges: Optional[np.ndarray] = None,  # (n_queries, 2) array of [min, max]
        use_hierarchy: bool = True,
        early_stop_threshold: Optional[int] = None,
        use_mid_filtering: bool = False,
        batch_size: int = 100  # 内部批处理大小
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        批量查询：对多个查询向量进行搜索（QPS优化）
        
        Args:
            queries: 查询向量数组 (n_queries, dimension)
            k: 返回的最近邻数量
            radius: 可选的范围限制
            scalar_ranges: 可选的属性值范围数组 (n_queries, 2)，每行为 [min, max]
            use_hierarchy: 是否使用层次化搜索策略
            early_stop_threshold: 提前终止阈值
            use_mid_filtering: 是否使用中过滤策略
            batch_size: 内部批处理大小（用于优化内存使用）
            
        Returns:
            (distances_list, indices_list): 距离和索引列表，每个元素是一个查询的结果
        """
        n_queries = queries.shape[0]
        distances_list = []
        indices_list = []
        
        # 如果scalar_ranges为None，使用单个None作为默认值
        if scalar_ranges is None:
            scalar_ranges = [None] * n_queries
        elif isinstance(scalar_ranges, np.ndarray):
            # 转换为列表，每行为一个元组
            scalar_ranges = [(scalar_ranges[i][0], scalar_ranges[i][1]) for i in range(n_queries)]
        
        # 分批处理查询（减少内存压力）
        for i in range(0, n_queries, batch_size):
            end_idx = min(i + batch_size, n_queries)
            batch_queries = queries[i:end_idx]
            batch_scalar_ranges = scalar_ranges[i:end_idx] if scalar_ranges else [None] * (end_idx - i)
            
            # 对每个查询调用search（当前实现，未来可以进一步向量化优化）
            for j, query in enumerate(batch_queries):
                scalar_range = batch_scalar_ranges[j] if batch_scalar_ranges else None
                distances, indices = self.search(
                    query, k=k, radius=radius, scalar_range=scalar_range,
                    use_hierarchy=use_hierarchy, early_stop_threshold=early_stop_threshold,
                    use_mid_filtering=use_mid_filtering
                )
                distances_list.append(distances)
                indices_list.append(indices)
        
        return distances_list, indices_list
    
    def range_search(
        self,
        query: np.ndarray,
        radius: float,
        max_results: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """范围查询"""
        return self.search(query, k=max_results, radius=radius, use_hierarchy=True)
    
    def hybrid_search(
        self,
        query: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None,
        scalar_range: Optional[Tuple[int, int]] = None,
        weight: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        混合查询：结合范围查询和近似最近邻搜索（支持范围混合过滤）
        
        Args:
            query: 查询向量
            k: 返回的最近邻数量
            radius: 可选的范围限制
            scalar_range: 可选的属性值范围 (min, max)
            weight: 范围查询和最近邻的权重（0-1之间）
        """
        # 先进行范围查询（如果有radius或scalar_range）
        if radius is not None or scalar_range is not None:
            range_distances, range_indices = self.search(
                query, k=k*2, radius=radius, scalar_range=scalar_range, use_hierarchy=True
            )
        else:
            range_distances, range_indices = np.array([]), np.array([])
        
        # 进行近似最近邻搜索
        nn_distances, nn_indices = self.search(query, k=k*2, radius=None, scalar_range=None, use_hierarchy=True)
        
        # 合并结果
        all_candidates = {}
        
        # 添加范围查询结果（加权）
        for dist, idx in zip(range_distances, range_indices):
            if idx not in all_candidates:
                all_candidates[idx] = dist * weight
        
        # 添加最近邻结果（加权）
        for dist, idx in zip(nn_distances, nn_indices):
            if idx not in all_candidates:
                all_candidates[idx] = dist * (1 - weight)
            else:
                all_candidates[idx] = min(all_candidates[idx], dist * (1 - weight))
        
        # 排序并返回top-k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1])
        top_k = sorted_candidates[:k]
        
        if not top_k:
            return np.array([]), np.array([])
        
        distances = np.array([d for _, d in top_k])
        indices = np.array([idx for idx, _ in top_k])
        
        return distances, indices
    
    def save(self, filepath: str):
        """保存层次化索引到文件"""
        # 保存底层和顶层索引
        fine_data_file = filepath + "_fine.pkl"
        coarse_data_file = filepath + "_coarse.pkl"
        
        self.fine_index.save(fine_data_file)
        self.coarse_index.save(coarse_data_file)
        
        # 保存层次关系和其他信息
        data = {
            'n_fine_clusters': self.n_fine_clusters,
            'n_coarse_clusters': self.n_coarse_clusters,
            'n_probe': self.n_probe,
            'max_cluster_size': self.max_cluster_size,
            'recluster_threshold': self.recluster_threshold,
            'centroid_update_threshold': self.centroid_update_threshold,
            'fine_to_coarse': self.fine_to_coarse,
            'fine_centroids_history': self.fine_centroids_history,
            'dimension': self.dimension,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"层次化Ada-IVF索引已保存到 {filepath}")
        
    def load(self, filepath: str):
        """从文件加载层次化索引"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.n_fine_clusters = data['n_fine_clusters']
        self.n_coarse_clusters = data['n_coarse_clusters']
        self.n_probe = data['n_probe']
        self.max_cluster_size = data['max_cluster_size']
        self.recluster_threshold = data['recluster_threshold']
        self.centroid_update_threshold = data.get('centroid_update_threshold', 0.1)
        self.fine_to_coarse = data['fine_to_coarse']
        self.fine_centroids_history = data.get('fine_centroids_history')
        self.dimension = data['dimension']
        self.is_trained = data['is_trained']
        
        # 加载底层和顶层索引
        fine_data_file = filepath + "_fine.pkl"
        coarse_data_file = filepath + "_coarse.pkl"
        
        self.fine_index = AdaIVFIndex()
        self.fine_index.load(fine_data_file)
        
        self.coarse_index = AdaIVFIndex()
        self.coarse_index.load(coarse_data_file)
        
        print(f"层次化Ada-IVF索引已从 {filepath} 加载")
        
    def get_stats(self) -> dict:
        """获取索引统计信息"""
        fine_stats = self.fine_index.get_stats()
        coarse_stats = self.coarse_index.get_stats()
        
        return {
            'n_fine_clusters': self.n_fine_clusters,
            'n_coarse_clusters': self.n_coarse_clusters,
            'dimension': self.dimension,
            'is_trained': self.is_trained,
            'fine_level': {
                'n_clusters': fine_stats['n_clusters'],
                'n_vectors': fine_stats['n_vectors'],
                'cluster_sizes': fine_stats['cluster_sizes'],
                'quality_scores': fine_stats['quality_scores']
            },
            'coarse_level': {
                'n_clusters': coarse_stats['n_clusters'],
                'n_vectors': coarse_stats['n_vectors'],
                'cluster_sizes': coarse_stats['cluster_sizes'],
                'quality_scores': coarse_stats['quality_scores']
            }
        }

