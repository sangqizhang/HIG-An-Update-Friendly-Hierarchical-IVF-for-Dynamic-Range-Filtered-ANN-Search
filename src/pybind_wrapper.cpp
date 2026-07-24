#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../include/ada_ivf_core.h"
#include "../include/ada_ivf_pybind_common.h"
#include <vector>
#include <memory>

namespace py = pybind11;

// Python包装类
class AdaIVFIndex {
public:
    AdaIVFIndex(int n_clusters, int n_probe, size_t max_cluster_size,
                float recluster_threshold)
        : core_(new AdaIVFCore(n_clusters, n_probe, max_cluster_size,
                              recluster_threshold)) {}
    
    ~AdaIVFIndex() {
        delete core_;
    }
    
    void train(py::array_t<float> vectors, py::array_t<float> scalars) {
        auto buf = vectors.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("向量数组必须是2维的 (n_vectors, dimension)");
        }
        
        size_t n_vectors = buf.shape[0];
        int dimension = buf.shape[1];
        const float* data = static_cast<const float*>(buf.ptr);
        
        const float* scalar_data = nullptr;
        // 检查 scalars 是否为空或 None
        if (scalars.size() > 0) {
            auto scalar_buf = scalars.request();
            if (scalar_buf.ndim != 1 || scalar_buf.shape[0] != n_vectors) {
                throw std::runtime_error("scalar数组必须是1维的，且长度与向量数量相同");
            }
            scalar_data = static_cast<const float*>(scalar_buf.ptr);
        }
        
        core_->train(data, n_vectors, dimension, scalar_data);
    }
    
    void add(py::array_t<float> vectors, py::array_t<int> ids, 
            bool auto_recluster, py::array_t<float> scalars) {
        auto vec_buf = vectors.request();
        if (vec_buf.ndim != 2) {
            throw std::runtime_error("向量数组必须是2维的");
        }
        
        size_t n_vectors = vec_buf.shape[0];
        const float* vec_data = static_cast<const float*>(vec_buf.ptr);
        
        const int* id_data = nullptr;
        if (ids.size() > 0) {
            auto id_buf = ids.request();
            if (id_buf.ndim != 1 || static_cast<size_t>(id_buf.shape[0]) != n_vectors) {
                throw std::runtime_error("ID数组必须与向量数量匹配");
            }
            id_data = static_cast<const int*>(id_buf.ptr);
        }
        
        const float* scalar_data = nullptr;
        if (scalars.size() > 0) {
            auto scalar_buf = scalars.request();
            if (scalar_buf.ndim != 1 || static_cast<size_t>(scalar_buf.shape[0]) != n_vectors) {
                throw std::runtime_error("标量数组必须与向量数量匹配");
            }
            scalar_data = static_cast<const float*>(scalar_buf.ptr);
        }
        
        core_->add(vec_data, n_vectors, id_data, auto_recluster, scalar_data);
    }
    
    py::tuple search(py::array_t<float> query, int k, int max_num_distances,
                    float range_min, float range_max) {
        auto buf = query.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("查询向量必须是1维的");
        }
        
        const float* query_data = static_cast<const float*>(buf.ptr);
        auto results = core_->search(query_data, k, max_num_distances, 
                                    range_min, range_max);
        
        // 转换为numpy数组
        size_t n_results = results.size();
        py::array_t<float> distances(n_results);
        py::array_t<int> indices(n_results);
        
        auto dist_buf = distances.request();
        auto idx_buf = indices.request();
        
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);
        int* idx_ptr = static_cast<int*>(idx_buf.ptr);
        
        for (size_t i = 0; i < n_results; ++i) {
            dist_ptr[i] = std::sqrt(results[i].first);  // 转换为实际距离
            idx_ptr[i] = results[i].second;
        }
        
        return py::make_tuple(distances, indices);
    }
    
    py::list batch_search(py::array_t<float> queries, int k, int max_num_distances,
                          float range_min, float range_max) {
        return ada_ivf_py::batch_search_to_python(
            core_, queries, k, max_num_distances, range_min, range_max);
    }

    py::list batch_search_ranges(py::array_t<float> queries, py::array_t<float> ranges,
                                 int k, int max_num_distances) {
        return ada_ivf_py::batch_search_ranges_to_python(
            core_, queries, ranges, k, max_num_distances);
    }

    py::tuple batch_search_with_stats(py::array_t<float> queries, int k, int max_num_distances,
                                      float range_min, float range_max) {
        return ada_ivf_py::batch_search_with_stats_to_python(
            core_, queries, k, max_num_distances, range_min, range_max);
    }

    py::tuple batch_search_ranges_with_stats(py::array_t<float> queries, py::array_t<float> ranges,
                                             int k, int max_num_distances) {
        return ada_ivf_py::batch_search_ranges_with_stats_to_python(
            core_, queries, ranges, k, max_num_distances);
    }

    
    void recluster_cluster(int cluster_id) {
        core_->recluster_cluster(cluster_id);
    }
    
    py::list identify_clusters_to_recluster() {
        auto clusters = core_->identify_clusters_to_recluster();
        py::list result;
        for (int cid : clusters) {
            result.append(cid);
        }
        return result;
    }
    
    float compute_cluster_quality(int cluster_id) {
        return core_->compute_cluster_quality(cluster_id);
    }
    
    
    // 属性访问
    size_t n_vectors() const { return core_->get_n_vectors(); }
    int dimension() const { return core_->get_dimension(); }
    int n_clusters() const { return core_->get_n_clusters(); }
    bool is_trained() const { return core_->is_trained(); }
    int n_probe() const { return core_->get_n_probe(); }
    void set_n_probe(int n_probe) { core_->set_n_probe(n_probe); }
    
    py::dict get_cluster_size_stats() {
        auto stats = core_->get_cluster_size_stats();
        py::dict result;
        result["min_size"] = stats.min_size;
        result["max_size"] = stats.max_size;
        result["avg_size"] = stats.avg_size;
        return result;
    }
    
    py::dict get_imbalance_metrics() {
        auto metrics = core_->get_imbalance_metrics();
        py::dict result;
        result["sigma"] = metrics.sigma;
        result["epsilon"] = metrics.epsilon;
        result["epsilon_prime"] = metrics.epsilon_prime;
        result["G"] = metrics.G;
        return result;
    }

    // Baseline 开关：用于关闭 Ada 维护机制（构造“纯 IVF”baseline）
    bool enable_maintenance() const { return core_->get_enable_maintenance(); }
    void set_enable_maintenance(bool enabled) { core_->set_enable_maintenance(enabled); }

    // Module A/B（与层次化 HierarchicalAdaIVFIndex 的 fine 层接口名保持一致）
    void set_enable_scalar_filter_lsm(bool v) { core_->set_enable_scalar_filter_lsm(v); }
    bool get_enable_scalar_filter_lsm() const { return core_->get_enable_scalar_filter_lsm(); }
    void set_enable_scalar_range_prune(bool v) { core_->set_enable_scalar_range_prune(v); }
    bool get_enable_scalar_range_prune() const { return core_->get_enable_scalar_range_prune(); }
    void set_lsm_merge_threshold(size_t t) { core_->set_lsm_merge_threshold(t); }
    size_t get_lsm_merge_threshold() const { return core_->get_lsm_merge_threshold(); }
    void flush_all_lsm_segments() { core_->flush_all_lsm_segments(); }
    void set_attr_lambda(float v) { core_->set_attr_lambda(v); }
    float get_attr_lambda() const { return core_->get_attr_lambda(); }
    void set_kmeans_seed(uint32_t v) { core_->set_kmeans_seed(v); }
    uint32_t get_kmeans_seed() const { return core_->get_kmeans_seed(); }
    void set_use_kmeanspp(bool v) { core_->set_use_kmeanspp(v); }
    bool get_use_kmeanspp() const { return core_->get_use_kmeanspp(); }

    // Module PQ：乘积量化压缩向量存储
    void set_enable_pq_compression(bool v) { core_->set_enable_pq_compression(v); }
    bool get_enable_pq_compression() const { return core_->get_enable_pq_compression(); }
    void set_pq_m(int m) { core_->set_pq_m(m); }
    int get_pq_m() const { return core_->get_pq_m(); }
    void set_pq_ksub(int k) { core_->set_pq_ksub(k); }
    int get_pq_ksub() const { return core_->get_pq_ksub(); }
    void set_pq_release_floats(bool v) { core_->set_pq_release_floats(v); }
    bool get_pq_release_floats() const { return core_->get_pq_release_floats(); }
    bool is_pq_trained() const { return core_->is_pq_trained(); }

    void train_pq_from_samples(py::array_t<float> vectors) {
        auto buf = vectors.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("train_pq_from_samples: 需要 2 维数组 (n_vectors, dimension)");
        }
        const float* data = static_cast<const float*>(buf.ptr);
        size_t n = static_cast<size_t>(buf.shape[0]);
        int dim = static_cast<int>(buf.shape[1]);
        core_->train_pq_from_samples(data, n, dim);
    }

private:
    AdaIVFCore* core_;
};

PYBIND11_MODULE(ada_ivf_core, m) {
    m.doc() = "Ada-IVF核心实现（C++热路径 + Python胶水层）";
    
    py::class_<AdaIVFIndex>(m, "AdaIVFIndex")
        .def(py::init<int, int, size_t, float>(),
             py::arg("n_clusters"), py::arg("n_probe"), 
             py::arg("max_cluster_size"), py::arg("recluster_threshold"),
             "创建Ada-IVF索引\n"
             "参数:\n"
             "  n_clusters: 聚类数量\n"
             "  n_probe: 搜索时探测的聚类数\n"
             "  max_cluster_size: 单个聚类最大容量\n"
             "  recluster_threshold: 重聚类质量阈值")
        
        .def("train", [](AdaIVFIndex& self, py::array_t<float> vectors, py::object scalars_obj) {
                if (scalars_obj.is_none()) {
                    // 传入空的 py::array_t<float> 作为 None 的替代
                    py::array_t<float> empty_scalars;
                    self.train(vectors, empty_scalars);
                } else {
                    try {
                    py::array_t<float> scalars = py::cast<py::array_t<float>>(scalars_obj);
                    self.train(vectors, scalars);
                    } catch (const py::cast_error&) {
                        // 如果转换失败，使用空数组
                        py::array_t<float> empty_scalars;
                        self.train(vectors, empty_scalars);
                    }
                }
            },
             py::arg("vectors"), py::arg("scalars") = py::none(),
             "训练索引\n"
             "参数:\n"
             "  vectors: 训练向量数组 (n_vectors, dimension)\n"
             "  scalars: 训练向量的scalar值数组 (n_vectors,)，可选")
        
        // 重要：不要在def时构造py::array_t默认值（会在import阶段触发numpy C-API初始化，导致卡住）
        .def("add", [](AdaIVFIndex& self,
                       py::array_t<float> vectors,
                       py::object ids_obj,
                       bool auto_recluster,
                       py::object scalars_obj) {
                py::array_t<int> ids;
                if (!ids_obj.is_none()) {
                    try {
                        ids = py::cast<py::array_t<int>>(ids_obj);
                    } catch (const py::cast_error&) {
                        ids = py::array_t<int>();  // fallback: empty
                    }
                }
                
                py::array_t<float> scalars;
                if (!scalars_obj.is_none()) {
                    try {
                        scalars = py::cast<py::array_t<float>>(scalars_obj);
                    } catch (const py::cast_error&) {
                        scalars = py::array_t<float>();  // fallback: empty
                    }
                }
                
                self.add(vectors, ids, auto_recluster, scalars);
            },
             py::arg("vectors"),
             py::arg("ids") = py::none(),
             py::arg("auto_recluster") = true,
             py::arg("scalars") = py::none(),
             "增量插入向量\n"
             "参数:\n"
             "  vectors: 向量数组 (n_vectors, dimension)\n"
             "  ids: 向量ID数组（可选）\n"
             "  auto_recluster: 是否自动重聚类\n"
             "  scalars: 标量值数组（可选）")
        
        .def("search", &AdaIVFIndex::search,
             py::arg("query"), py::arg("k")=10, py::arg("max_num_distances")=1000,
             py::arg("range_min")=std::numeric_limits<float>::lowest(),
             py::arg("range_max")=std::numeric_limits<float>::max(),
             "单查询搜索\n"
             "返回: (distances, indices)")
        
        .def("batch_search", &AdaIVFIndex::batch_search,
             py::arg("queries"), py::arg("k")=10, py::arg("max_num_distances")=1000,
             py::arg("range_min")=std::numeric_limits<float>::lowest(),
             py::arg("range_max")=std::numeric_limits<float>::max(),
             "批量查询搜索（OpenMP并行化）\n"
             "返回: indices_list（int32 numpy，按距离升序）")
        .def("batch_search_ranges", &AdaIVFIndex::batch_search_ranges,
             py::arg("queries"), py::arg("ranges"), py::arg("k")=10,
             py::arg("max_num_distances")=1000,
             "批量查询搜索：每条 query 使用自己的 range，并保持 C++ OpenMP 并行")
        .def("batch_search_with_stats", &AdaIVFIndex::batch_search_with_stats,
             py::arg("queries"), py::arg("k")=10, py::arg("max_num_distances")=1000,
             py::arg("range_min")=std::numeric_limits<float>::lowest(),
             py::arg("range_max")=std::numeric_limits<float>::max(),
             "批量查询搜索，并返回 (indices_list, stats_dict)")
        .def("batch_search_ranges_with_stats", &AdaIVFIndex::batch_search_ranges_with_stats,
             py::arg("queries"), py::arg("ranges"), py::arg("k")=10,
             py::arg("max_num_distances")=1000,
             "每条 query 独立 range 的批量查询，并返回 (indices_list, stats_dict)")
        
        .def("recluster_cluster", &AdaIVFIndex::recluster_cluster,
             py::arg("cluster_id"),
             "对指定聚类进行局部重聚类")
        
        .def("identify_clusters_to_recluster", &AdaIVFIndex::identify_clusters_to_recluster,
             "识别需要重聚类的聚类\n"
             "返回: 聚类ID列表")
        
        .def("compute_cluster_quality", &AdaIVFIndex::compute_cluster_quality,
             py::arg("cluster_id"),
             "计算聚类质量分数")
        
        
        .def_property_readonly("n_vectors", &AdaIVFIndex::n_vectors,
                              "已插入的向量数量")
        .def_property_readonly("dimension", &AdaIVFIndex::dimension,
                              "向量维度")
        .def_property_readonly("n_clusters", &AdaIVFIndex::n_clusters,
                              "聚类数量")
        .def_property_readonly("is_trained", &AdaIVFIndex::is_trained,
                              "是否已训练")
        .def_property("n_probe", &AdaIVFIndex::n_probe, &AdaIVFIndex::set_n_probe,
                     "搜索时探测的聚类数（可动态修改）")
        .def("get_cluster_size_stats", &AdaIVFIndex::get_cluster_size_stats,
             "获取聚类大小统计信息")
        .def("get_imbalance_metrics", &AdaIVFIndex::get_imbalance_metrics,
             "获取失衡指标（sigma, epsilon, epsilon_prime, G）")
        .def_property("enable_scalar_filter_lsm", &AdaIVFIndex::get_enable_scalar_filter_lsm,
                      &AdaIVFIndex::set_enable_scalar_filter_lsm,
                      "Module A：IVF 桶 main+delta（LSM），默认 False")
        .def_property("enable_scalar_range_prune", &AdaIVFIndex::get_enable_scalar_range_prune,
                      &AdaIVFIndex::set_enable_scalar_range_prune,
                      "簇级 scalar min/max 粗剪枝，默认 False")
        .def_property("lsm_merge_threshold", &AdaIVFIndex::get_lsm_merge_threshold,
                      &AdaIVFIndex::set_lsm_merge_threshold,
                      "Module A：无序段长度阈值 t，默认 100")
        .def_property("attr_lambda", &AdaIVFIndex::get_attr_lambda, &AdaIVFIndex::set_attr_lambda,
                      "属性感知聚类权重 lambda；train 前设置，默认 0")
        .def_property("kmeans_seed", &AdaIVFIndex::get_kmeans_seed, &AdaIVFIndex::set_kmeans_seed,
                      "K-means++ 固定随机种子，默认 42")
        .def_property("use_kmeanspp", &AdaIVFIndex::get_use_kmeanspp, &AdaIVFIndex::set_use_kmeanspp,
                      "是否使用 K-means++ 初始化，默认 True")
        .def("flush_all_lsm_segments", &AdaIVFIndex::flush_all_lsm_segments,
             "将全部桶的 delta 并入 main（保存/导出前建议调用）")
        .def_property("enable_maintenance", &AdaIVFIndex::enable_maintenance, &AdaIVFIndex::set_enable_maintenance,
                     "是否启用Ada维护机制（重聚类/质心更新/全局重建）。关闭后即为“纯IVF”baseline。")
        .def_property("enable_pq_compression", &AdaIVFIndex::get_enable_pq_compression,
                      &AdaIVFIndex::set_enable_pq_compression,
                      "Module PQ：插入后只保留 PQ 码并释放原始 float，检索用 ADC 近似平方 L2，默认 False")
        .def_property("pq_m", &AdaIVFIndex::get_pq_m, &AdaIVFIndex::set_pq_m,
                      "PQ 段数 m（维度须整除 m），默认 8；须在 train 前与 ksub 一起生效于码本训练")
        .def_property("pq_ksub", &AdaIVFIndex::get_pq_ksub, &AdaIVFIndex::set_pq_ksub,
                      "每段子码字数 ksub（2–256），默认 256")
        .def_property("pq_release_floats", &AdaIVFIndex::get_pq_release_floats,
                      &AdaIVFIndex::set_pq_release_floats,
                      "PQ 后是否释放原始 float 向量（默认 True：省内存但召回可能下降；False 可用全精度距离精排）")
        .def_property_readonly("is_pq_trained", &AdaIVFIndex::is_pq_trained,
                               "PQ 码本是否已训练（train 末尾或 train_pq_from_samples 后）")
        .def("train_pq_from_samples", &AdaIVFIndex::train_pq_from_samples,
             py::arg("vectors"),
             "用一批向量单独训练/刷新 PQ 码本（维数须与索引一致；适用于 train 后才打开 PQ 等场景）");
}
