#pragma once

#include "ada_ivf_core.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ada_ivf_py {

inline py::list batch_results_indices_to_python(
    const std::vector<std::vector<std::pair<float, int>>>& batch_results) {
    py::list indices_list;
    for (const auto& results : batch_results) {
        const py::ssize_t n_results = static_cast<py::ssize_t>(results.size());
        py::array_t<int32_t> indices(n_results);
        auto idx_buf = indices.request();
        int32_t* idx_ptr = static_cast<int32_t*>(idx_buf.ptr);
        for (py::ssize_t i = 0; i < n_results; ++i) {
            idx_ptr[i] = results[static_cast<size_t>(i)].second;
        }
        indices_list.append(indices);
    }
    return indices_list;
}

inline py::dict aggregate_search_stats_to_python(
    const std::vector<AdaIVFCore::SearchStats>& stats_rows) {
    AdaIVFCore::SearchStats total;
    for (const auto& s : stats_rows) {
        total.add(s);
    }
    py::dict d;
    d["num_dist_computations"] = total.num_dist_computations;
    d["num_scanned_vectors"] = total.num_scanned_vectors;
    d["list_vectors_considered"] = total.list_vectors_considered;
    d["scalar_rows_scanned"] = total.scalar_rows_scanned;
    d["scalar_blocks_skipped"] = total.scalar_blocks_skipped;
    d["scalar_block_rows_skipped"] = total.scalar_block_rows_skipped;
    d["num_filtered_candidates"] = total.num_filtered_candidates;
    d["num_visited_clusters"] = total.num_visited_clusters;
    d["num_pruned_clusters"] = total.num_pruned_clusters;
    d["num_candidate_clusters"] = total.num_candidate_clusters;
    d["graph_nodes_visited"] = total.graph_nodes_visited;
    d["graph_scalar_positive_nodes"] = total.graph_scalar_positive_nodes;
    d["graph_scalar_negative_nodes"] = total.graph_scalar_negative_nodes;
    d["graph_fallback_expansions"] = total.graph_fallback_expansions;
    d["candidate_clusters"] = total.num_candidate_clusters;
    d["hard_pruned_clusters"] = total.num_pruned_clusters;
    d["actual_scanned_clusters"] = total.num_visited_clusters;
    d["valid_points"] = total.num_filtered_candidates;
    d["distance_calculations"] = total.num_dist_computations;
    d["n_queries"] = stats_rows.size();
    d["avg_dist_computations_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.num_dist_computations) / stats_rows.size();
    d["avg_scanned_vectors_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.num_scanned_vectors) / stats_rows.size();
    d["avg_list_vectors_considered_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.list_vectors_considered) / stats_rows.size();
    d["avg_scalar_rows_scanned_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.scalar_rows_scanned) / stats_rows.size();
    d["avg_scalar_blocks_skipped_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.scalar_blocks_skipped) / stats_rows.size();
    d["avg_scalar_block_rows_skipped_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.scalar_block_rows_skipped) / stats_rows.size();
    d["avg_filtered_candidates_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.num_filtered_candidates) / stats_rows.size();
    d["avg_candidate_clusters_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.num_candidate_clusters) / stats_rows.size();
    d["avg_graph_nodes_visited_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.graph_nodes_visited) / stats_rows.size();
    d["avg_graph_scalar_positive_nodes_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.graph_scalar_positive_nodes) / stats_rows.size();
    d["avg_graph_scalar_negative_nodes_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.graph_scalar_negative_nodes) / stats_rows.size();
    d["avg_graph_fallback_expansions_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.graph_fallback_expansions) / stats_rows.size();
    d["avg_hard_pruned_clusters_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.num_pruned_clusters) / stats_rows.size();
    d["avg_actual_scanned_clusters_per_query"] =
        stats_rows.empty() ? 0.0 : static_cast<double>(total.num_visited_clusters) / stats_rows.size();
    return d;
}

// AdaIVFCore::batch_search → Python indices_list（每条 query 一个 int32 numpy，已按距离升序）
inline py::list batch_search_to_python(
    const AdaIVFCore* core,
    py::array_t<float> queries,
    int k,
    int max_num_distances,
    float range_min,
    float range_max) {
    if (core == nullptr) {
        throw std::runtime_error("fine 层索引未初始化");
    }
    auto qbuf = queries.request();
    if (qbuf.ndim != 2) {
        throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
    }
    size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
    const float* qdata = static_cast<const float*>(qbuf.ptr);
    auto batch_results = core->batch_search(
        qdata, n_queries, k, max_num_distances, range_min, range_max);
    return batch_results_indices_to_python(batch_results);
}

// AdaIVFCore::batch_search_ranges → Python indices_list；每条 query 使用自己的 range。
inline py::list batch_search_ranges_to_python(
    const AdaIVFCore* core,
    py::array_t<float> queries,
    py::array_t<float> ranges,
    int k,
    int max_num_distances) {
    if (core == nullptr) {
        throw std::runtime_error("fine 层索引未初始化");
    }
    auto qbuf = queries.request();
    auto rbuf = ranges.request();
    if (qbuf.ndim != 2) {
        throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
    }
    if (rbuf.ndim != 2 || rbuf.shape[1] != 2 || rbuf.shape[0] != qbuf.shape[0]) {
        throw std::runtime_error("ranges 必须是二维数组 (n_queries, 2)，且长度与 queries 匹配");
    }
    size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
    const float* qdata = static_cast<const float*>(qbuf.ptr);
    const float* rdata = static_cast<const float*>(rbuf.ptr);
    auto batch_results = core->batch_search_ranges(
        qdata, rdata, n_queries, k, max_num_distances);
    return batch_results_indices_to_python(batch_results);
}

inline py::tuple batch_search_with_stats_to_python(
    const AdaIVFCore* core,
    py::array_t<float> queries,
    int k,
    int max_num_distances,
    float range_min,
    float range_max) {
    if (core == nullptr) {
        throw std::runtime_error("fine 层索引未初始化");
    }
    auto qbuf = queries.request();
    if (qbuf.ndim != 2) {
        throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
    }
    size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
    const float* qdata = static_cast<const float*>(qbuf.ptr);
    auto out = core->batch_search_with_stats(
        qdata, n_queries, k, max_num_distances, range_min, range_max);
    return py::make_tuple(
        batch_results_indices_to_python(out.results),
        aggregate_search_stats_to_python(out.per_query_stats));
}

inline py::tuple batch_search_ranges_with_stats_to_python(
    const AdaIVFCore* core,
    py::array_t<float> queries,
    py::array_t<float> ranges,
    int k,
    int max_num_distances) {
    if (core == nullptr) {
        throw std::runtime_error("fine 层索引未初始化");
    }
    auto qbuf = queries.request();
    auto rbuf = ranges.request();
    if (qbuf.ndim != 2) {
        throw std::runtime_error("queries 必须是二维数组 (n_queries, d)");
    }
    if (rbuf.ndim != 2 || rbuf.shape[1] != 2 || rbuf.shape[0] != qbuf.shape[0]) {
        throw std::runtime_error("ranges 必须是二维数组 (n_queries, 2)，且长度与 queries 匹配");
    }
    size_t n_queries = static_cast<size_t>(qbuf.shape[0]);
    const float* qdata = static_cast<const float*>(qbuf.ptr);
    const float* rdata = static_cast<const float*>(rbuf.ptr);
    auto out = core->batch_search_ranges_with_stats(
        qdata, rdata, n_queries, k, max_num_distances);
    return py::make_tuple(
        batch_results_indices_to_python(out.results),
        aggregate_search_stats_to_python(out.per_query_stats));
}

}  // namespace ada_ivf_py
