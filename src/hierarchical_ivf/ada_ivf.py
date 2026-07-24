"""
Ada-IVF (Adaptive Inverted File Index) 实现
基于论文: Incremental IVF Index Maintenance for Streaming Vector Search

核心特性:
1. 自适应维护策略：识别性能下降的索引分区
2. 局部重聚类机制：对选定分区进行重新聚类
3. 层次化结构：支持多级索引
4. 范围混合查询：支持范围查询和近似最近邻搜索
5. 动态插入：高效的增量更新
6. 优化搜索：向量化距离计算，先范围过滤再k-NN
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import pickle
import time

# FAISS优化：用于加速K-means和距离计算
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: FAISS未安装，将使用numpy实现。建议安装FAISS以提升性能: pip install faiss-cpu")


class ClusterStats:
    """聚类统计信息"""
    def __init__(self, cluster_id: int):
        self.cluster_id = cluster_id
        self.size = 0
        self.last_recluster_time = time.time()
        self.insert_count_since_recluster = 0
        self.avg_distance_to_centroid = 0.0
        self.max_distance_to_centroid = 0.0
        self.quality_score = 1.0  # 质量分数，越低表示需要重聚类


class AdaIVFIndex:
    """Ada-IVF索引类 - 支持自适应维护和动态插入"""
    
    def __init__(
        self,
        n_clusters: int = 100,
        n_probe: int = 10,
        max_cluster_size: int = 10000,
        recluster_threshold: float = 0.3,
        recluster_ratio: float = 0.5
    ):
        """
        初始化Ada-IVF索引
        
        Args:
            n_clusters: 聚类中心数量
            n_probe: 搜索时探测的聚类中心数量
            max_cluster_size: 单个聚类的最大向量数量，超过后触发重聚类
            recluster_threshold: 重聚类质量阈值，低于此值触发重聚类
            recluster_ratio: 重聚类时使用的子聚类比例
        """
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.max_cluster_size = max_cluster_size
        self.recluster_threshold = recluster_threshold
        self.recluster_ratio = recluster_ratio
        
        self.centroids = None
        self.inverted_lists = defaultdict(list)  # {cluster_id: [vector_ids]}
        self.vectors = {}  # {vector_id: vector} 使用字典提高查找效率
        self.cluster_stats = {}  # {cluster_id: ClusterStats}
        self.dimension = None
        self.is_trained = False
        
        # 用于范围查询的边界信息
        self.cluster_bounds = {}  # {cluster_id: (min_bound, max_bound)}
        
        # 用于属性值范围过滤的索引（如果提供）
        self.vector_scalars = {}  # {vector_id: scalar_value} 用于范围过滤
        
        # 优化：为每个聚类维护排序的scalar数组（SoA模式，支持二分查找）
        # cluster_scalars[cluster_id] = 排序的scalar数组（与inverted_lists[cluster_id]对应）
        self.cluster_scalars = defaultdict(list)  # {cluster_id: [sorted_scalars]}
        # cluster_scalar_indices[cluster_id] = 排序后的索引数组，用于映射回vector_ids
        self.cluster_scalar_indices = defaultdict(list)  # {cluster_id: [indices]}
        
        # 优化：预计算向量范数（用于距离计算优化）
        self.vector_norms = {}  # {vector_id: norm_squared / 2}
        
    def train(self, vectors: np.ndarray):
        """训练索引：使用K-means找到聚类中心（使用FAISS加速）"""
        if len(vectors) == 0:
            raise ValueError("训练向量不能为空")
            
        self.dimension = vectors.shape[1]
        n_vectors = vectors.shape[0]
        
        print(f"开始训练Ada-IVF索引，聚类数量: {self.n_clusters}, 训练向量数: {n_vectors}")
        
        # 确保向量是float32类型（FAISS要求）
        vectors_f32 = vectors.astype(np.float32)
        
        # 使用FAISS的K-means（如果可用）进行加速
        if FAISS_AVAILABLE and n_vectors > 1000:
            # FAISS K-means优化：使用GPU或SIMD优化的CPU实现
            # 根据数据量和聚类数动态调整迭代次数
            # 大数据量或大聚类数需要更多迭代才能充分收敛
            if n_vectors >= 500000 or self.n_clusters >= 500:
                niter = 300  # 大数据量或大聚类数：300次迭代
            elif n_vectors >= 200000 or self.n_clusters >= 300:
                niter = 200  # 中等数据量：200次迭代
            else:
                niter = 100  # 小数据量：100次迭代
            
            print(f"  使用FAISS K-means训练，迭代次数: {niter}")
            train_start = time.time()
            kmeans = faiss.Kmeans(
                self.dimension,
                self.n_clusters,
                niter=niter,
                verbose=True,
                gpu=False,  # 如果安装了faiss-gpu，可以设置为True
                seed=42
            )
            kmeans.train(vectors_f32)
            train_elapsed = time.time() - train_start
            print(f"  FAISS K-means训练完成，耗时: {train_elapsed:.2f}秒")
            self.centroids = kmeans.centroids.astype(np.float64)
            
            # 初始化聚类统计信息
            for i in range(self.n_clusters):
                if i not in self.cluster_stats:
                    self.cluster_stats[i] = ClusterStats(i)
            
            # 初始化聚类边界
            self._update_cluster_bounds()
            self.is_trained = True
        else:
            # 回退到numpy实现（如果FAISS不可用或数据量小）
            # K-means初始化
            np.random.seed(42)
            indices = np.random.choice(n_vectors, self.n_clusters, replace=False)
            self.centroids = vectors[indices].copy()
            
            # K-means迭代
            max_iter = 100
            for iteration in range(max_iter):
                # 向量化计算距离
                distances = np.linalg.norm(
                    vectors[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
                    axis=2
                )
                assignments = np.argmin(distances, axis=1)
                
                new_centroids = np.zeros_like(self.centroids)
                for i in range(self.n_clusters):
                    mask = assignments == i
                    if np.sum(mask) > 0:
                        new_centroids[i] = vectors[mask].mean(axis=0)
                        # 初始化聚类统计信息
                        if i not in self.cluster_stats:
                            self.cluster_stats[i] = ClusterStats(i)
                    else:
                        new_centroids[i] = self.centroids[i]
                
                if np.allclose(self.centroids, new_centroids, atol=1e-6):
                    print(f"K-means在第 {iteration + 1} 次迭代后收敛")
                    break
                    
                self.centroids = new_centroids
            
            # 初始化聚类边界
            self._update_cluster_bounds()
            self.is_trained = True
            print("Ada-IVF索引训练完成")
        
    def _compute_distances_faiss(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        使用FAISS计算向量到质心的距离（L2距离）
        
        Args:
            vectors: 查询向量数组 (n_vectors, dimension)
            centroids: 质心数组 (n_clusters, dimension)
            
        Returns:
            距离矩阵 (n_vectors, n_clusters)
        """
        if FAISS_AVAILABLE and len(vectors) > 10:  # 降低阈值，更多使用FAISS SIMD优化
            try:
                # 使用FAISS的IndexFlatL2进行批量距离计算（SIMD优化）
                vectors_f32 = vectors.astype(np.float32)
                centroids_f32 = centroids.astype(np.float32)
                
                # 创建FAISS索引
                index = faiss.IndexFlatL2(centroids_f32.shape[1])
                index.add(centroids_f32)
                
                # 批量搜索：找到每个向量到所有质心的距离
                distances_sq, _ = index.search(vectors_f32, centroids_f32.shape[0])
                
                # 返回距离（FAISS返回的是平方距离）
                return np.sqrt(distances_sq.astype(np.float64))
            except:
                # FAISS失败时回退到numpy
                return np.linalg.norm(
                    vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                    axis=2
                )
        else:
            # 回退到numpy实现
            return np.linalg.norm(
                vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                axis=2
            )
    
    def _compute_distances_faiss_single(self, query: np.ndarray, vectors: np.ndarray, 
                                       vector_norms: Optional[Dict] = None) -> np.ndarray:
        """
        使用FAISS计算单个查询向量到多个向量的距离（L2平方距离）
        优化：使用FAISS SIMD优化，降低阈值以更多使用FAISS
        
        Args:
            query: 查询向量 (dimension,)
            vectors: 向量数组 (n_vectors, dimension)
            vector_norms: 可选的预计算范数字典 {vector_id: norm_squared/2}
            
        Returns:
            平方距离数组 (n_vectors,)
        """
        if FAISS_AVAILABLE and len(vectors) > 5:  # 降低阈值，更多使用FAISS SIMD优化
            try:
                # 使用FAISS的IndexFlatL2进行批量距离计算（SIMD优化）
                query_f32 = query.astype(np.float32).reshape(1, -1)
                vectors_f32 = vectors.astype(np.float32)
                
                # 创建FAISS索引并添加向量
                index = faiss.IndexFlatL2(vectors_f32.shape[1])
                index.add(vectors_f32)
                
                # 批量搜索：计算所有向量到查询向量的距离
                distances_sq, _ = index.search(query_f32, len(vectors_f32))
                distances_sq_all = distances_sq.flatten().astype(np.float64)  # FAISS返回平方距离
                
                return distances_sq_all
            except:
                # FAISS失败时回退到numpy
                diff = vectors - query
                distances_sq_all = np.sum(diff * diff, axis=1)
                return distances_sq_all
        else:
            # 回退到numpy实现（使用平方距离，避免开方）
            diff = vectors - query
            distances_sq_all = np.sum(diff * diff, axis=1)
            return distances_sq_all
        
    def _update_cluster_bounds(self):
        """更新聚类边界信息（用于范围查询）"""
        for cluster_id in range(self.n_clusters):
            vector_ids = self.inverted_lists[cluster_id]
            if len(vector_ids) > 0:
                vectors_in_cluster = np.array([self.vectors[vid] for vid in vector_ids])
                min_bound = vectors_in_cluster.min(axis=0)
                max_bound = vectors_in_cluster.max(axis=0)
                self.cluster_bounds[cluster_id] = (min_bound, max_bound)
            else:
                # 空聚类使用中心点作为边界
                centroid = self.centroids[cluster_id]
                self.cluster_bounds[cluster_id] = (centroid, centroid)
    
    def _compute_cluster_quality(self, cluster_id: int) -> float:
        """计算聚类质量分数"""
        if cluster_id not in self.cluster_stats:
            return 1.0
            
        stats = self.cluster_stats[cluster_id]
        vector_ids = self.inverted_lists[cluster_id]
        
        if len(vector_ids) == 0:
            return 1.0
        
        # 向量化计算平均距离到中心（使用FAISS优化）
        centroid = self.centroids[cluster_id]
        vectors_in_cluster = np.array([self.vectors[vid] for vid in vector_ids])
        # 使用FAISS计算距离（批量计算）
        if len(vectors_in_cluster) > 0:
            distances = self._compute_distances_faiss(
                vectors_in_cluster, 
                centroid.reshape(1, -1)
            ).flatten()
        else:
            distances = np.array([])
        
        avg_dist = np.mean(distances)
        max_dist = np.max(distances)
        
        stats.avg_distance_to_centroid = avg_dist
        stats.max_distance_to_centroid = max_dist
        
        # 质量分数：考虑大小、平均距离、最大距离
        size_factor = min(1.0, self.max_cluster_size / max(len(vector_ids), 1))
        dist_factor = 1.0 / (1.0 + avg_dist)  # 距离越小，质量越高
        
        quality = size_factor * dist_factor
        stats.quality_score = quality
        
        return quality
    
    def _identify_clusters_to_recluster(self) -> List[int]:
        """识别需要重聚类的聚类（自适应维护策略）"""
        clusters_to_recluster = []
        
        for cluster_id in range(self.n_clusters):
            stats = self.cluster_stats.get(cluster_id)
            if stats is None:
                continue
                
            quality = self._compute_cluster_quality(cluster_id)
            vector_ids = self.inverted_lists[cluster_id]
            
            # 检查是否需要重聚类
            needs_recluster = False
            
            # 条件1：质量分数低于阈值
            if quality < self.recluster_threshold:
                needs_recluster = True
                
            # 条件2：聚类大小超过最大值
            if len(vector_ids) > self.max_cluster_size:
                needs_recluster = True
                
            # 条件3：插入次数过多
            if stats.insert_count_since_recluster > self.max_cluster_size * 0.5:
                needs_recluster = True
            
            if needs_recluster:
                clusters_to_recluster.append(cluster_id)
        
        return clusters_to_recluster
    
    def _recluster_cluster(self, cluster_id: int):
        """对指定聚类进行局部重聚类"""
        vector_ids = self.inverted_lists[cluster_id]
        if len(vector_ids) < 2:
            return
        
        vectors_in_cluster = np.array([self.vectors[vid] for vid in vector_ids])
        centroid = self.centroids[cluster_id]
        
        # 计算子聚类数量（基于当前大小）
        n_subclusters = max(2, min(self.n_clusters // 10, len(vector_ids) // 10))
        n_subclusters = min(n_subclusters, len(vector_ids))
        
        if n_subclusters < 2:
            return
        
        # 使用K-means进行局部重聚类（使用FAISS加速）
        vectors_f32 = vectors_in_cluster.astype(np.float32)
        
        if FAISS_AVAILABLE and len(vectors_in_cluster) > 100:
            # 使用FAISS的K-means进行重聚类（更快）
            kmeans = faiss.Kmeans(
                vectors_f32.shape[1],
                n_subclusters,
                niter=20,
                verbose=False,
                gpu=False,
                seed=42
            )
            kmeans.train(vectors_f32)
            sub_centroids = kmeans.centroids.astype(np.float64)
            
            # 计算分配（使用FAISS）
            distances = self._compute_distances_faiss(vectors_in_cluster, sub_centroids)
            assignments = np.argmin(distances, axis=1)
        else:
            # 回退到numpy实现
            np.random.seed(42)
            indices = np.random.choice(len(vectors_in_cluster), n_subclusters, replace=False)
            sub_centroids = vectors_in_cluster[indices].copy()
            
            # 简化的K-means迭代（向量化）
            for _ in range(20):
                distances = self._compute_distances_faiss(vectors_in_cluster, sub_centroids)
                assignments = np.argmin(distances, axis=1)
                
                new_sub_centroids = np.zeros_like(sub_centroids)
                for i in range(n_subclusters):
                    mask = assignments == i
                    if np.sum(mask) > 0:
                        new_sub_centroids[i] = vectors_in_cluster[mask].mean(axis=0)
                    else:
                        new_sub_centroids[i] = sub_centroids[i]
                
                if np.allclose(sub_centroids, new_sub_centroids, atol=1e-6):
                    break
                sub_centroids = new_sub_centroids
        
        # 将向量重新分配到新的子聚类或最近的现有聚类
        # 计算每个子聚类中心到所有现有聚类中心的距离（向量化）
        distances_to_all_centroids = np.linalg.norm(
            sub_centroids[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
            axis=2
        )
        
        # 为每个向量找到最佳分配（向量化）
        vector_distances_to_sub = np.linalg.norm(
            vectors_in_cluster[:, np.newaxis, :] - sub_centroids[np.newaxis, :, :],
            axis=2
        )
        nearest_sub = np.argmin(vector_distances_to_sub, axis=1)
        
        new_assignments = {}
        for i, vid in enumerate(vector_ids):
            nearest_cluster = np.argmin(distances_to_all_centroids[nearest_sub[i]])
            new_assignments[vid] = nearest_cluster
        
        # 重新分配向量
        self.inverted_lists[cluster_id] = []
        for vid, new_cluster_id in new_assignments.items():
            if new_cluster_id != cluster_id:
                self.inverted_lists[new_cluster_id].append(vid)
            else:
                self.inverted_lists[cluster_id].append(vid)
        
        # 更新聚类中心（使用新分配的向量）
        if len(self.inverted_lists[cluster_id]) > 0:
            vectors_in_new = np.array([self.vectors[vid] for vid in self.inverted_lists[cluster_id]])
            self.centroids[cluster_id] = vectors_in_new.mean(axis=0)
        
        # 更新统计信息
        if cluster_id in self.cluster_stats:
            self.cluster_stats[cluster_id].last_recluster_time = time.time()
            self.cluster_stats[cluster_id].insert_count_since_recluster = 0
        
        # 更新边界
        self._update_cluster_bounds()
        
        # 优化：更新所有受影响聚类的标量范围（重聚类后向量可能被重新分配）
        # 添加层级标识，便于区分细粒度层和粗粒度层的重聚类
        layer_name = "细粒度" if self.n_clusters >= 100 else "粗粒度"
        print(f"[{layer_name}层] 聚类 {cluster_id} 重聚类完成，子聚类数: {n_subclusters}")
    
    def add(self, vectors: np.ndarray, ids: Optional[List] = None, auto_recluster: bool = True, scalars: Optional[np.ndarray] = None):
        """
        动态插入向量到索引
        
        Args:
            vectors: 要添加的向量
            ids: 可选的向量ID列表
            auto_recluster: 是否自动触发重聚类
            scalars: 可选的属性值数组，用于范围过滤
        """
        if not self.is_trained:
            raise ValueError("索引尚未训练，请先调用train()方法")
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度不匹配：期望 {self.dimension}，得到 {vectors.shape[1]}")
        
        n_vectors = vectors.shape[0]
        if ids is None:
            start_id = max(self.vectors.keys(), default=-1) + 1
            ids = list(range(start_id, start_id + n_vectors))
        
        # 向量分配：基于距离（使用FAISS加速）
        distances = self._compute_distances_faiss(vectors, self.centroids)
        assignments = np.argmin(distances, axis=1)
        
        # 添加向量
        for i, (vector, vector_id, cluster_id) in enumerate(zip(vectors, ids, assignments)):
            self.inverted_lists[cluster_id].append(vector_id)
            self.vectors[vector_id] = vector.copy()
            
            # 优化：预计算向量范数（用于距离计算优化）
            norm_sq = np.sum(vector * vector)
            self.vector_norms[vector_id] = norm_sq / 2.0
            
            # 保存属性值（如果提供）
            if scalars is not None:
                scalar_val = scalars[i]
                self.vector_scalars[vector_id] = scalar_val
                
                # 优化：维护排序的scalar数组（SoA模式）
                # 使用bisect插入，保持有序
                import bisect
                scalar_list = self.cluster_scalars[cluster_id]
                idx_list = self.cluster_scalar_indices[cluster_id]
                # 找到插入位置
                insert_pos = bisect.bisect_left(scalar_list, scalar_val)
                scalar_list.insert(insert_pos, scalar_val)
                # 存储vector_id在inverted_lists中的位置（即list中的索引）
                # 注意：此时vector_id刚被append，所以位置是len-1
                idx_list.insert(insert_pos, len(self.inverted_lists[cluster_id]) - 1)
            
            # 更新统计信息
            if cluster_id not in self.cluster_stats:
                self.cluster_stats[cluster_id] = ClusterStats(cluster_id)
            self.cluster_stats[cluster_id].size = len(self.inverted_lists[cluster_id])
            self.cluster_stats[cluster_id].insert_count_since_recluster += 1
            
            # 优化：更新聚类的标量范围（用于快速预过滤）
        # 更新边界
        self._update_cluster_bounds()
        
        # 自适应维护：检查并重聚类
        if auto_recluster:
            clusters_to_recluster = self._identify_clusters_to_recluster()
            for cluster_id in clusters_to_recluster[:5]:
                self._recluster_cluster(cluster_id)
        
        print(f"已添加 {n_vectors} 个向量到索引")
    
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
        if len(self.vectors) == 0:
            return base_n_probe
        
        # 计算平均聚类大小
        cluster_sizes = [len(self.inverted_lists[cid]) for cid in range(self.n_clusters)]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        
        # 动态调整n_probe
        n_probe = base_n_probe
        
        if use_mid_filtering:
            # 中过滤策略：更激进的n_probe调整以提升QPS
            # 对于非层次化Ada-IVF，需要更大的min_n_probe，因为只有一层
            # 层次化中n_probe是粗粒度层，每个粗粒度聚类包含多个细粒度聚类
            # 非层次化中n_probe直接对应搜索的聚类数，需要更多才能保证召回率
            min_n_probe_by_clusters = max(1, int(self.n_clusters * 0.05))
            max_n_probe_by_clusters = int(self.n_clusters * 0.20)
            
            if n_probe < min_n_probe_by_clusters:
                n_probe = min_n_probe_by_clusters
            n_probe = min(n_probe, max_n_probe_by_clusters)
            
            # 进一步降低数据量增长幅度（中过滤时更保守）
            if len(self.vectors) > 100000:
                data_factor = 1.0 + (len(self.vectors) - 100000) / 100000 * 0.05
                n_probe = int(n_probe * data_factor)
            elif len(self.vectors) > 50000:
                data_factor = 1.0 + (len(self.vectors) - 50000) / 100000 * 0.03
                n_probe = int(n_probe * data_factor)
            
            # 进一步降低聚类大小影响（中过滤时更保守）
            if avg_cluster_size > 1500:
                n_probe = int(n_probe * (1 + (avg_cluster_size - 1500) / 5000))
            
            max_n_probe = max(1, int(self.n_clusters * 0.20))  # 最多20%（提高以提升召回率）
        else:
            # 预过滤策略：保持原有策略
            min_n_probe_by_clusters = max(1, int(self.n_clusters * 0.05))  # 至少5%，但至少1
            max_n_probe_by_clusters = int(self.n_clusters * 0.30)  # 最多30%
            
            if n_probe < min_n_probe_by_clusters:
                n_probe = min_n_probe_by_clusters
            n_probe = min(n_probe, max_n_probe_by_clusters)
            
            if avg_cluster_size > 500:
                n_probe = int(n_probe * (1 + (avg_cluster_size - 500) / 2000))
            
            if len(self.vectors) > 100000:
                data_factor = 1.0 + (len(self.vectors) - 100000) / 100000 * 0.15
                n_probe = int(n_probe * data_factor)
            elif len(self.vectors) > 50000:
                data_factor = 1.0 + (len(self.vectors) - 50000) / 100000 * 0.1
                n_probe = int(n_probe * data_factor)
            elif len(self.vectors) > 20000:
                data_factor = 1.0 + (len(self.vectors) - 20000) / 100000 * 0.05
                n_probe = int(n_probe * data_factor)
            
            max_n_probe = max(1, int(self.n_clusters * 0.30))  # 最多30%的聚类
        
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
        n_vectors = len(self.vectors)
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
        
        return min(search_k, max_search_k)
    
    def _compute_scalar_selection_rate(self, range_min: float, range_max: float) -> float:
        """
        计算标量选择率（标量范围内的数据量 / 总数据量）
        优化：利用已排序的cluster_scalars，通过二分查找快速计算，避免逐个遍历
        
        Args:
            range_min: 标量范围最小值
            range_max: 标量范围最大值
            
        Returns:
            选择率（0.0-1.0）
        """
        if len(self.vector_scalars) == 0:
            return 1.0
        
        count_in_range = 0
        
        # 遍历所有聚类，利用已排序的cluster_scalars快速统计
        for cluster_id in range(self.n_clusters):
            scalar_list = self.cluster_scalars[cluster_id]
            if not scalar_list:
                continue
            
            # 使用二分查找定位范围
            scalar_array = np.array(scalar_list)
            start_idx = np.searchsorted(scalar_array, range_min, side='left')
            end_idx = np.searchsorted(scalar_array, range_max, side='right')
            count_in_range += (end_idx - start_idx)
        
        return count_in_range / len(self.vector_scalars) if len(self.vector_scalars) > 0 else 0.0
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None,
        scalar_range: Optional[Tuple[int, int]] = None,
        early_stop_threshold: Optional[int] = None,
        use_mid_filtering: bool = False  # 是否使用中过滤策略
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的k个向量（支持范围查询和属性值范围过滤）
        
        Args:
            query: 查询向量
            k: 返回的最近邻数量
            radius: 可选的范围查询半径，如果提供则只返回距离小于radius的向量
            scalar_range: 可选的属性值范围 (min, max)，先进行范围过滤再k-NN
            early_stop_threshold: 提前终止阈值，找到这么多结果后提前停止（默认k*3）
            
        Returns:
            (distances, indices): 距离和索引的元组
        """
        if not self.is_trained or len(self.vectors) == 0:
            raise ValueError("索引为空或未训练")
            
        if query.shape[0] != self.dimension:
            raise ValueError(f"查询向量维度不匹配：期望 {self.dimension}，得到 {query.shape[0]}")
        
        dynamic_n_probe = self._get_dynamic_n_probe(self.n_probe, use_mid_filtering=use_mid_filtering)
        
        # 找到最近的n_probe个聚类中心（使用FAISS加速）
        distances_to_centroids = self._compute_distances_faiss(
            query.reshape(1, -1), 
            self.centroids
        ).flatten()
        probe_clusters = np.argsort(distances_to_centroids)[:dynamic_n_probe]
        
        cluster_sizes = [(cid, len(self.inverted_lists[cid])) for cid in probe_clusters]
        cluster_sizes.sort(key=lambda x: x[1])  # 从小到大排序
        probe_clusters = [cid for cid, _ in cluster_sizes]
        
        # 如果未提供early_stop_threshold，使用动态调整后的值
        if early_stop_threshold is None:
            # 使用动态调整的search_k（与层次化保持一致）
            early_stop_threshold = self._get_dynamic_search_k(0, k=k, use_mid_filtering=use_mid_filtering)
        else:
            # 如果提供了early_stop_threshold，也进行动态调整（与层次化保持一致）
            early_stop_threshold = self._get_dynamic_search_k(early_stop_threshold, k=k, use_mid_filtering=use_mid_filtering)
        
        # 初始化当前最大距离为无穷大
        current_max_distance = float('inf')
        
        # 优化策略1: 先范围过滤再k-NN（如果提供scalar_range）- 向量化优化 + 提前终止
        # 关键优化：在收集向量时就进行范围过滤，而不是收集后再过滤
        # 如果use_mid_filtering=True，使用中过滤策略（先k-NN再范围过滤）
        if scalar_range is not None and len(self.vector_scalars) > 0 and not use_mid_filtering:
            range_min, range_max = scalar_range
            
            # 向量化优化：批量收集候选向量，并在收集时立即进行范围过滤（支持提前终止）
            all_vector_ids = []
            all_vectors_list = []
            
            for cluster_id in probe_clusters:
                # 只在收集到足够结果后才应用（避免过早跳过包含最近邻的聚类）
                if current_max_distance < float('inf') and len(all_vector_ids) >= k * 2:
                    # 计算距离下界
                    distance_to_centroid = distances_to_centroids[cluster_id]
                    cluster_stats = self.cluster_stats.get(cluster_id)
                    cluster_radius = cluster_stats.max_distance_to_centroid if cluster_stats else 0.0
                    
                    # 距离下界 = 查询到中心距离 - 聚类半径
                    distance_lower_bound = max(0.0, distance_to_centroid - cluster_radius)
                    
                    # 添加安全系数（1.1倍），避免过于激进的提前终止
                    # 如果距离下界已经大于当前top-k最大距离的1.1倍，跳过该聚类
                    if distance_lower_bound > current_max_distance * 1.1:
                        continue  # 跳过这个聚类
                
                vector_ids = self.inverted_lists[cluster_id]
                if len(vector_ids) > 0:
                    # 关键优化：在收集向量时就进行范围过滤
                    # 批量获取属性值并过滤
                    scalars_batch = np.array([self.vector_scalars.get(vid, -1) for vid in vector_ids])
                    range_mask = (scalars_batch >= range_min) & (scalars_batch <= range_max)
                    
                    # 只收集满足范围条件的向量
                    filtered_vector_ids = [vid for vid, mask in zip(vector_ids, range_mask) if mask]
                    if len(filtered_vector_ids) > 0:
                        # 优化：批量获取满足条件的向量（减少字典查找）
                        # 使用列表推导式批量获取，比循环更快
                        vectors_list = [self.vectors[vid] for vid in filtered_vector_ids]
                        vectors_batch = np.array(vectors_list)
                        all_vector_ids.extend(filtered_vector_ids)
                        all_vectors_list.append(vectors_batch)
                
                stop_threshold = early_stop_threshold
                if use_mid_filtering:
                    # 中过滤：如果已收集到足够候选（k*2），可以提前停止
                    # 因为中过滤会先排序再过滤，不需要太多候选
                    stop_threshold = min(early_stop_threshold, k * 2)
                if len(all_vector_ids) >= stop_threshold:
                    break
            
            if not all_vectors_list:
                return np.array([]), np.array([])
            
            # 合并所有向量（向量化）
            all_vectors_array = np.vstack(all_vectors_list)
            all_vector_ids_array = np.array(all_vector_ids)
            
            # 注意：此时所有向量已经满足范围条件，无需再次过滤
            filtered_vectors = all_vectors_array
            filtered_vector_ids = all_vector_ids_array
            
            # 优化：使用FAISS加速距离计算（统一使用辅助方法）
            # 使用_compute_distances_faiss_single获取平方距离，然后开方
            distances_sq_all = self._compute_distances_faiss_single(query, filtered_vectors, self.vector_norms)
            distances_all = np.sqrt(distances_sq_all)  # 开方得到实际距离
            
            # 范围查询过滤（如果有radius）
            if radius is not None:
                valid_mask = distances_all <= radius
                distances_all = distances_all[valid_mask]
                filtered_vector_ids = filtered_vector_ids[valid_mask]
            
            if len(distances_all) == 0:
                return np.array([]), np.array([])
            
            # 如果已有k个结果，更新最大距离
            if len(distances_all) >= k:
                # 使用部分排序找到第k小的距离
                if len(distances_all) > k:
                    kth_distance = np.partition(distances_all, k)[k]
                else:
                    kth_distance = np.max(distances_all)
                current_max_distance = min(current_max_distance, kth_distance)
            
            # 排序并返回top-k（使用部分排序优化）
            if len(distances_all) > k * 2:
                # 如果结果很多，使用argpartition优化
                top_k_indices = np.argpartition(distances_all, k)[:k]
                top_k_indices = top_k_indices[np.argsort(distances_all[top_k_indices])]
            else:
                top_k_indices = np.argsort(distances_all)[:k]
            
            distances = distances_all[top_k_indices]
            indices = filtered_vector_ids[top_k_indices]
            
            return distances, indices
        
        # 优化策略2: 传统方法（先k-NN再范围过滤）- 向量化优化 + 提前终止
        # 中过滤优化：在收集阶段进行粗略范围过滤（如果范围很窄）
        all_vector_ids = []
        all_vectors = []
        
        # 这样可以减少需要计算距离的向量数量
        apply_rough_filtering = False
        if use_mid_filtering and scalar_range is not None and len(self.vector_scalars) > 0:
            range_min, range_max = scalar_range
            range_width = range_max - range_min
            # 动态判断：如果范围宽度小于总范围的5%，应用粗略过滤
            # 假设范围是0-10000，如果范围宽度<500，应用粗略过滤
            apply_rough_filtering = (range_width < 500)  # 可调整阈值
        
        for cluster_id in probe_clusters:
            # 只在收集到足够结果后才应用（避免过早跳过包含最近邻的聚类）
            if current_max_distance < float('inf') and len(all_vector_ids) >= k * 2:
                # 计算距离下界
                distance_to_centroid = distances_to_centroids[cluster_id]
                cluster_stats = self.cluster_stats.get(cluster_id)
                cluster_radius = cluster_stats.max_distance_to_centroid if cluster_stats else 0.0
                
                # 距离下界 = 查询到中心距离 - 聚类半径
                distance_lower_bound = max(0.0, distance_to_centroid - cluster_radius)
                
                # 添加安全系数（1.1倍），避免过于激进的提前终止
                # 如果距离下界已经大于当前top-k最大距离的1.1倍，跳过该聚类
                if distance_lower_bound > current_max_distance * 1.1:
                    continue  # 跳过这个聚类
            
            vector_ids = self.inverted_lists[cluster_id]
            if len(vector_ids) > 0:
                if apply_rough_filtering:
                    range_min, range_max = scalar_range
                    # 优化：使用二分查找快速定位标量范围（如果cluster_scalars已排序）
                    scalar_list = self.cluster_scalars[cluster_id]
                    if scalar_list and len(scalar_list) == len(vector_ids):
                        # 使用二分查找定位范围
                        scalar_array = np.array(scalar_list)
                        start_idx = np.searchsorted(scalar_array, range_min, side='left')
                        end_idx = np.searchsorted(scalar_array, range_max, side='right')
                        # 获取范围内的索引
                        idx_list = self.cluster_scalar_indices[cluster_id]
                        if idx_list and len(idx_list) == len(vector_ids):
                            filtered_indices = idx_list[start_idx:end_idx]
                            filtered_vector_ids = [vector_ids[i] for i in filtered_indices]
                        else:
                            # 回退到遍历
                            scalars_list = [self.vector_scalars.get(vid, -1) for vid in vector_ids]
                            scalars_batch = np.array(scalars_list, dtype=np.float32)
                            range_mask = (scalars_batch >= range_min) & (scalars_batch <= range_max)
                            filtered_vector_ids = np.array(vector_ids)[range_mask].tolist()
                    else:
                        # 回退到批量获取scalars并过滤（向量化）
                        scalars_list = [self.vector_scalars.get(vid, -1) for vid in vector_ids]
                        scalars_batch = np.array(scalars_list, dtype=np.float32)
                        range_mask = (scalars_batch >= range_min) & (scalars_batch <= range_max)
                        filtered_vector_ids = np.array(vector_ids)[range_mask].tolist()
                    
                    if len(filtered_vector_ids) > 0:
                        # 优化：批量获取满足范围条件的向量（减少字典查找）
                        vectors_list = [self.vectors[vid] for vid in filtered_vector_ids]
                        vectors = np.array(vectors_list)
                        all_vector_ids.extend(filtered_vector_ids)
                        all_vectors.append(vectors)
                else:
                    # 优化：批量获取向量（减少字典查找开销）
                    # 使用列表推导式批量获取，比循环更快
                    vectors_list = [self.vectors[vid] for vid in vector_ids]
                    vectors = np.array(vectors_list)
                    all_vector_ids.extend(vector_ids)
                    all_vectors.append(vectors)
            
            # 与层次化保持一致：使用early_stop_threshold（动态调整后的值）
            # 对于中过滤，需要收集更多候选，因为先排序再过滤，范围窄时可能很多候选不满足条件
            stop_threshold = early_stop_threshold
            if use_mid_filtering:
                # 中过滤：需要收集更多候选以保证召回率
                # 因为先排序再过滤，如果范围很窄，可能需要更多候选才能找到足够的满足条件的向量
                # 使用early_stop_threshold（已经动态调整），但至少保证收集足够的候选
                # 与层次化保持一致：层次化中early_stop_threshold也会动态调整
                if apply_rough_filtering:
                    # 粗略过滤后，可以稍微降低阈值，但仍需保证足够候选
                    stop_threshold = max(int(early_stop_threshold * 0.8), k * 3)  # 至少k*3
                else:
                    # 没有粗略过滤，使用early_stop_threshold（已经动态调整），但至少k*5
                    stop_threshold = max(early_stop_threshold, k * 5)  # 至少k*5，或early_stop_threshold
            elif apply_rough_filtering:
                # 预过滤：粗略过滤后，需要更少的候选就能保证召回率
                stop_threshold = int(early_stop_threshold * 0.8)
            
            if len(all_vector_ids) >= stop_threshold:
                break
        
        if not all_vectors:
            return np.array([]), np.array([])
        
        # 优化：合并所有向量（向量化）
        # 如果只有一个小数组，直接使用；否则使用vstack
        if len(all_vectors) == 1:
            all_vectors_array = all_vectors[0]
            all_vector_ids_array = np.array(all_vector_ids)
        else:
            all_vectors_array = np.vstack(all_vectors)
            all_vector_ids_array = np.array(all_vector_ids)
        
        # 优化：使用FAISS加速距离计算（如果可用且向量数量足够）
        # 先计算平方距离，只在需要时开方
        # 使用新的辅助方法，统一FAISS调用
        distances_sq_all = self._compute_distances_faiss_single(query, all_vectors_array, self.vector_norms)
        
        # 如果已有k个结果，更新最大距离（使用平方距离）
        if len(distances_sq_all) >= k:
            # 使用部分排序找到第k小的平方距离
            if len(distances_sq_all) > k:
                kth_distance_sq = np.partition(distances_sq_all, k)[k]
                kth_distance = np.sqrt(kth_distance_sq)
            else:
                kth_distance = np.sqrt(np.max(distances_sq_all))
            current_max_distance = min(current_max_distance, kth_distance)
        
        # 中过滤策略：先排序，再应用范围过滤（优化版本）
        if scalar_range is not None and len(self.vector_scalars) > 0 and use_mid_filtering:
            range_min, range_max = scalar_range
            
            # 如果范围很窄，需要更多候选来保证召回率
            range_width = range_max - range_min
            range_ratio = range_width / 10000.0 if range_max > 0 else 1.0  # 假设范围是0-10000
            
            # 根据范围大小调整候选数量（回退到高召回率配置）
            if range_ratio < 0.01:  # 范围很窄（<1%）
                # 极窄范围：需要更多候选
                base_multiplier = 8  # 回退到8
            elif range_ratio < 0.05:  # 窄范围（1-5%）
                base_multiplier = 6  # 回退到6
            elif range_ratio < 0.1:  # 中等范围（5-10%）
                base_multiplier = 5  # 回退到5
            else:  # 宽范围（>10%）
                base_multiplier = 3  # 回退到3
            
            # 根据数据量进一步调整（使用平方距离数组的长度）
            if len(distances_sq_all) > k * base_multiplier:
                candidate_k = min(k * base_multiplier, len(distances_sq_all))
            elif len(distances_sq_all) > k * 3:
                candidate_k = min(k * 3, len(distances_sq_all))
            else:
                candidate_k = len(distances_sq_all)
            
            if len(distances_sq_all) > candidate_k and candidate_k > 200:
                # 大数组：使用argpartition（O(n)）
                top_candidate_indices = np.argpartition(distances_sq_all, candidate_k)[:candidate_k]
                top_candidate_indices = top_candidate_indices[np.argsort(distances_sq_all[top_candidate_indices])]
            elif len(distances_sq_all) > candidate_k:
                top_candidate_indices = np.argsort(distances_sq_all)[:candidate_k]
            else:
                top_candidate_indices = np.argsort(distances_sq_all)
            
            # 只对选中的候选计算精确距离（开方）
            distances_all = np.sqrt(distances_sq_all[top_candidate_indices])
            
            top_candidate_ids = all_vector_ids_array[top_candidate_indices]
            # 优化：使用向量化批量获取，避免逐个字典查找
            # 如果vector_scalars是字典，批量获取会更快
            if hasattr(self.vector_scalars, 'get'):
                # 字典类型：优化为批量获取
                # 使用列表推导式，但预先检查ID是否存在，减少字典查找
                scalars_list = []
                for vid in top_candidate_ids:
                    # 直接使用get方法，避免KeyError
                    scalars_list.append(self.vector_scalars.get(vid, -1))
                scalars_candidates = np.array(scalars_list, dtype=np.float32)
            else:
                # 如果是数组类型，直接索引（更快）
                scalars_candidates = np.array([self.vector_scalars[vid] if vid < len(self.vector_scalars) else -1 for vid in top_candidate_ids], dtype=np.float32)
            range_mask = (scalars_candidates >= range_min) & (scalars_candidates <= range_max)
            
            # 优化：如果已经找到足够的满足范围条件的结果，可以提前应用过滤
            # 但不要直接返回，因为还需要检查radius过滤
            if np.sum(range_mask) >= k:
                # 已经有足够的满足条件的结果，提前应用范围过滤
                # 使用range_mask直接索引distances_all和all_vector_ids_array（它们都是基于top_candidate_indices的）
                distances_all = distances_all[range_mask]
                all_vector_ids_array = all_vector_ids_array[range_mask]
                # 设置标志，跳过后续的范围过滤和扩展搜索
                skip_range_expansion = True
            else:
                skip_range_expansion = False
            
            if not skip_range_expansion:
                filtered_count = np.sum(range_mask)
                if filtered_count < k and len(all_vector_ids) < early_stop_threshold * 2:
                    # 扩大搜索范围：如果当前候选不足k个，尝试更多候选
                    # 根据范围大小动态调整扩展倍数
                    expansion_multiplier = max(8, int(1.0 / max(range_ratio, 0.001)))  # 范围越窄，扩展越多
                    expanded_candidate_k = min(k * expansion_multiplier, len(distances_sq_all))
                    if expanded_candidate_k > candidate_k:
                        # 优化：对于扩展搜索，也使用优化的排序
                        if expanded_candidate_k > 100:
                            expanded_indices = np.argpartition(distances_sq_all, expanded_candidate_k)[:expanded_candidate_k]
                            expanded_indices = expanded_indices[np.argsort(distances_sq_all[expanded_indices])]
                        else:
                            expanded_indices = np.argsort(distances_sq_all)[:expanded_candidate_k]
                        expanded_ids = all_vector_ids_array[expanded_indices]
                        # 优化：批量获取scalars（使用优化的方法）
                        if hasattr(self.vector_scalars, 'get'):
                            expanded_scalars_list = [self.vector_scalars.get(vid, -1) for vid in expanded_ids]
                            expanded_scalars = np.array(expanded_scalars_list, dtype=np.float32)
                        else:
                            expanded_scalars = np.array([self.vector_scalars[vid] if vid < len(self.vector_scalars) else -1 for vid in expanded_ids], dtype=np.float32)
                        expanded_mask = (expanded_scalars >= range_min) & (expanded_scalars <= range_max)
                        
                        if np.sum(expanded_mask) > filtered_count:
                            # 使用扩展后的结果
                            top_candidate_indices = expanded_indices
                            range_mask = expanded_mask
                            # 重新计算精确距离
                            distances_all = np.sqrt(distances_sq_all[top_candidate_indices])
                            # 应用范围过滤：使用range_mask直接索引
                            distances_all = distances_all[range_mask]
                            all_vector_ids_array = all_vector_ids_array[range_mask]
                        else:
                            # 应用范围过滤（扩展后仍不足）：使用range_mask直接索引
                            distances_all = distances_all[range_mask]
                            all_vector_ids_array = all_vector_ids_array[range_mask]
                    else:
                        # 应用范围过滤（正常情况）：使用range_mask直接索引
                        distances_all = distances_all[range_mask]
                        all_vector_ids_array = all_vector_ids_array[range_mask]
                else:
                    # 应用范围过滤（正常情况，结果足够）：使用range_mask直接索引
                    distances_all = distances_all[range_mask]
                    all_vector_ids_array = all_vector_ids_array[range_mask]
            # else: 已经提前应用了范围过滤，跳过后续处理
        elif scalar_range is not None and len(self.vector_scalars) > 0:
            # 传统后过滤：先计算距离，再范围过滤
            range_min, range_max = scalar_range
            # 计算精确距离（从平方距离）
            distances_all = np.sqrt(distances_sq_all)
            # 优化：批量获取scalars
            if hasattr(self.vector_scalars, 'get'):
                scalars_batch = np.array([self.vector_scalars.get(vid, -1) for vid in all_vector_ids_array], dtype=np.float32)
            else:
                scalars_batch = np.array([self.vector_scalars[vid] if vid < len(self.vector_scalars) else -1 for vid in all_vector_ids_array], dtype=np.float32)
            range_mask = (scalars_batch >= range_min) & (scalars_batch <= range_max)
            distances_all = distances_all[range_mask]
            all_vector_ids_array = all_vector_ids_array[range_mask]
        else:
            # 没有范围过滤，直接计算精确距离
            distances_all = np.sqrt(distances_sq_all)
        
        # 范围查询过滤（如果有radius）
        if radius is not None:
            valid_mask = distances_all <= radius
            distances_all = distances_all[valid_mask]
            all_vector_ids_array = all_vector_ids_array[valid_mask]
        
        if len(distances_all) == 0:
            return np.array([]), np.array([])
        
        # 优化排序：如果结果很多，使用argpartition（部分排序）
        if len(distances_all) > k * 3:  # 从k*2提高到k*3
            # 只排序top-k，而不是全部排序
            top_k_indices = np.argpartition(distances_all, k)[:k]
            top_k_indices = top_k_indices[np.argsort(distances_all[top_k_indices])]
        else:
            top_k_indices = np.argsort(distances_all)[:k]
        
        distances = distances_all[top_k_indices]
        indices = all_vector_ids_array[top_k_indices]
        
        return distances, indices
    
    def range_search(
        self,
        query: np.ndarray,
        radius: float,
        max_results: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        范围查询：返回距离查询向量在指定半径内的所有向量
        
        Args:
            query: 查询向量
            radius: 搜索半径
            max_results: 最大返回结果数
            
        Returns:
            (distances, indices): 距离和索引的元组
        """
        return self.search(query, k=max_results, radius=radius)
    
    def hybrid_search(
        self,
        query: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None,
        scalar_range: Optional[Tuple[int, int]] = None,
        weight: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        混合查询：结合范围查询和近似最近邻搜索
        
        Args:
            query: 查询向量
            k: 返回的最近邻数量
            radius: 可选的范围限制
            scalar_range: 可选的属性值范围 (min, max)
            weight: 范围查询和最近邻的权重（0-1之间）
            
        Returns:
            (distances, indices): 距离和索引的元组
        """
        # 先进行范围查询（如果有radius或scalar_range）
        if radius is not None or scalar_range is not None:
            range_distances, range_indices = self.search(
                query, k=k*2, radius=radius, scalar_range=scalar_range
            )
        else:
            range_distances, range_indices = np.array([]), np.array([])
        
        # 进行近似最近邻搜索
        nn_distances, nn_indices = self.search(query, k=k*2, radius=None, scalar_range=None)
        
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
                # 如果同时出现在两个结果中，使用较小值
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
        """保存索引到文件"""
        # 转换cluster_stats为可序列化格式
        stats_dict = {}
        for cid, stats in self.cluster_stats.items():
            stats_dict[cid] = {
                'cluster_id': stats.cluster_id,
                'size': stats.size,
                'last_recluster_time': stats.last_recluster_time,
                'insert_count_since_recluster': stats.insert_count_since_recluster,
                'avg_distance_to_centroid': stats.avg_distance_to_centroid,
                'max_distance_to_centroid': stats.max_distance_to_centroid,
                'quality_score': stats.quality_score
            }
        
        data = {
            'n_clusters': self.n_clusters,
            'n_probe': self.n_probe,
            'max_cluster_size': self.max_cluster_size,
            'recluster_threshold': self.recluster_threshold,
            'recluster_ratio': self.recluster_ratio,
            'centroids': self.centroids,
            'inverted_lists': dict(self.inverted_lists),
            'vectors': {k: v for k, v in self.vectors.items()},
            'vector_scalars': self.vector_scalars,
            'cluster_stats': stats_dict,
            'cluster_bounds': {k: (v[0], v[1]) for k, v in self.cluster_bounds.items()},
            'dimension': self.dimension,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"索引已保存到 {filepath}")
        
    def load(self, filepath: str):
        """从文件加载索引"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.n_clusters = data['n_clusters']
        self.n_probe = data['n_probe']
        self.max_cluster_size = data.get('max_cluster_size', 10000)
        self.recluster_threshold = data.get('recluster_threshold', 0.3)
        self.recluster_ratio = data.get('recluster_ratio', 0.5)
        self.centroids = data['centroids']
        self.inverted_lists = defaultdict(list, data['inverted_lists'])
        self.vectors = {k: v for k, v in data['vectors'].items()}
        self.vector_scalars = data.get('vector_scalars', {})
        self.dimension = data['dimension']
        self.is_trained = data['is_trained']
        
        # 恢复cluster_stats
        self.cluster_stats = {}
        stats_dict = data.get('cluster_stats', {})
        for cid, stats_data in stats_dict.items():
            stats = ClusterStats(cid)
            stats.size = stats_data['size']
            stats.last_recluster_time = stats_data['last_recluster_time']
            stats.insert_count_since_recluster = stats_data['insert_count_since_recluster']
            stats.avg_distance_to_centroid = stats_data['avg_distance_to_centroid']
            stats.max_distance_to_centroid = stats_data['max_distance_to_centroid']
            stats.quality_score = stats_data['quality_score']
            self.cluster_stats[cid] = stats
        
        # 恢复cluster_bounds
        self.cluster_bounds = {}
        for k, (min_b, max_b) in data.get('cluster_bounds', {}).items():
            self.cluster_bounds[k] = (min_b, max_b)
        
        print(f"索引已从 {filepath} 加载")
        
    def get_stats(self) -> dict:
        """获取索引统计信息"""
        cluster_sizes = [len(self.inverted_lists[i]) for i in range(self.n_clusters)]
        quality_scores = [
            self.cluster_stats[i].quality_score if i in self.cluster_stats else 1.0
            for i in range(self.n_clusters)
        ]
        
        return {
            'n_clusters': self.n_clusters,
            'n_vectors': len(self.vectors),
            'dimension': self.dimension,
            'n_probe': self.n_probe,
            'is_trained': self.is_trained,
            'has_scalars': len(self.vector_scalars) > 0,
            'cluster_sizes': {
                'min': min(cluster_sizes) if cluster_sizes else 0,
                'max': max(cluster_sizes) if cluster_sizes else 0,
                'avg': np.mean(cluster_sizes) if cluster_sizes else 0
            },
            'quality_scores': {
                'min': min(quality_scores) if quality_scores else 1.0,
                'max': max(quality_scores) if quality_scores else 1.0,
                'avg': np.mean(quality_scores) if quality_scores else 1.0
            }
        }
