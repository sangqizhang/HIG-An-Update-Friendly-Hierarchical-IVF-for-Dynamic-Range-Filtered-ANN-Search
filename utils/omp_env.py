# -*- coding: utf-8 -*-
"""OpenMP 环境：训练多线程 / 插入单线程 / 查询单线程（对齐 UNIFY/DSG 公平对比）。"""
import ctypes
import os


def _try_omp_set_num_threads(n: int) -> bool:
    n = int(n)
    for lib_name in ("libgomp.so.1", "libgomp.so", "libiomp5.so", "libomp.so"):
        try:
            lib = ctypes.CDLL(lib_name)
            lib.omp_set_num_threads(n)
            return True
        except OSError:
            continue
    return False


def default_train_omp_threads() -> int:
    for key in ("ADA_IVF_TRAIN_OMP_THREADS", "TRAIN_OMP_THREADS"):
        v = os.environ.get(key, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    return 32


def default_insert_omp_threads() -> int:
    for key in ("INSERT_OMP_THREADS", "ADA_IVF_INSERT_OMP_THREADS", "QUERY_OMP_THREADS"):
        v = os.environ.get(key, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    return 1


def default_query_omp_threads() -> int:
    v = os.environ.get("QUERY_OMP_THREADS", os.environ.get("OMP_NUM_THREADS", "1")).strip()
    if v.isdigit() and int(v) > 0:
        return int(v)
    return 1


def apply_train_omp_env(n_threads: int) -> None:
    """K-means / train 前：多线程（不影响后续 insert，因 train 读 ADA_IVF_TRAIN_OMP_THREADS）。"""
    n = max(1, int(n_threads))
    os.environ["ADA_IVF_TRAIN_OMP_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["ADA_IVF_OMP_THREADS"] = str(n)
    _try_omp_set_num_threads(n)


def apply_insert_omp_env(n_threads: int = 1) -> None:
    """insert/add 前：默认单线程，对齐 UNIFY set_num_threads(1)。"""
    n = max(1, int(n_threads))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["ADA_IVF_OMP_THREADS"] = str(n)
    _try_omp_set_num_threads(n)


def apply_query_omp_env(n_threads: int = 1) -> None:
    """查询 benchmark 前：默认单线程。"""
    n = max(1, int(n_threads))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["ADA_IVF_OMP_THREADS"] = str(n)
    if n <= 1:
        os.environ["HIER_ADA_IVF_OMP"] = "0"
    else:
        os.environ.pop("HIER_ADA_IVF_OMP", None)
    _try_omp_set_num_threads(n)
