# -*- coding: utf-8 -*-
"""
Ada-IVF索引安装配置（基于论文：Incremental IVF Index Maintenance for Streaming Vector Search）

双语言实现策略：
- C++实现热路径（索引结构、局部k-means、并发控制）
- Python负责协调与验证（参数扫描、指标统计、可视化）
- pybind11作为胶水层
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os

_root = os.path.abspath(os.path.dirname(__file__))
_readme = os.path.join(_root, "README.md")
if os.path.isfile(_readme):
    with open(_readme, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = (
        "Ada-IVF core (C++ + pybind11). See MODULE_AB_LSM_Margin使用说明.md in the repo."
    )

_req = os.path.join(_root, "requirements.txt")
if os.path.isfile(_req):
    with open(_req, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = []

# 单层 K-means 训练 OpenMP（默认开；若 .so 导入异常可 ENABLE_OPENMP=0 bash build.sh）
_enable_openmp = os.environ.get("ENABLE_OPENMP", "1").strip().lower() not in ("0", "false", "no")
_openmp_args = ["-fopenmp"] if _enable_openmp else []
_openmp_link = ["-fopenmp"] if _enable_openmp else []

# C++扩展模块配置
ext_modules = [
    Pybind11Extension(
        "ada_ivf_core",
        [
            "src/pybind_wrapper.cpp",
            "src/ada_ivf_core.cpp",
            "src/simd_utils.cpp",
        ],
        include_dirs=[
            "include",
            pybind11.get_include(),
            "/usr/include",  # 系统头文件路径（包含crypt.h）
        ],
        language="c++",
        cxx_std=11,  # 使用C++11（兼容GCC 4.8.5）
        extra_compile_args=[
            "-std=c++11",  # 强制使用C++11（覆盖pybind11的自动检测）
            "-O3",  # 优化级别
            "-march=native",  # 使用本地CPU架构优化
            *_openmp_args,
            "-Wall",
            "-Wextra",
            "-Wno-strict-prototypes",  # 忽略C++中的C警告
            # 注意：不添加-fvisibility=default，避免影响单层性能
            # 层次化模块通过ADA_IVF_EXPORT宏确保符号可见
        ],
        extra_link_args=_openmp_link,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="ada-ivf-core",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Ada-IVF核心实现（C++热路径 + Python胶水层）",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ada-ivf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
    ],
    python_requires=">=3.7",
    install_requires=requirements + ["pybind11>=2.6.0"],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    zip_safe=False,
)











