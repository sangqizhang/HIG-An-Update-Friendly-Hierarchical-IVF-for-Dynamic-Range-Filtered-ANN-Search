#!/bin/bash
# Ada-IVF编译脚本

set -e

echo "开始编译Ada-IVF C++扩展..."

# 优先使用 conda 的编译器（若存在），否则回退到系统 gcc/g++
if command -v x86_64-conda_cos6-linux-gnu-gcc >/dev/null 2>&1 && command -v x86_64-conda_cos6-linux-gnu-g++ >/dev/null 2>&1; then
  echo "检测到 conda 编译器：使用 x86_64-conda_cos6-linux-gnu-gcc/g++"
  export CC=x86_64-conda_cos6-linux-gnu-gcc
  export CXX=x86_64-conda_cos6-linux-gnu-g++
else
  echo "未检测到 conda 编译器：回退使用系统 gcc/g++"
  export CC=${CC:-gcc}
  export CXX=${CXX:-g++}
fi

echo "CC=${CC}"
echo "CXX=${CXX}"

# 添加系统头文件路径
export CFLAGS="-I/usr/include $CFLAGS"
export CXXFLAGS="-I/usr/include $CXXFLAGS"

# AVX/FMA 开关（默认开启）：
# - ENABLE_AVX=ON: 追加 -mavx -mfma（单层 setup.py 走 CXXFLAGS；层次化通过 CMake 开关）
# - ENABLE_AVX=OFF: 不追加 AVX/FMA 参数
ENABLE_AVX=${ENABLE_AVX:-ON}
if [ "${ENABLE_AVX}" = "ON" ]; then
  export CXXFLAGS="-mavx -mfma ${CXXFLAGS}"
fi

# 编译单层模块
echo ""
echo "编译单层模块..."
python3 setup.py build_ext --inplace

# 编译层次化模块（使用 CMake）
echo ""
echo "编译层次化模块..."
if [ -d "build" ]; then
    rm -rf build
fi
mkdir -p build
cd build
# cmake .. || { echo "cmake 失败（请确认已安装 cmake、pybind11 的 CMake 包与 g++）"; exit 1; }
cmake .. -DENABLE_AVX="${ENABLE_AVX}" -DCMAKE_BUILD_TYPE=Release || { echo "cmake 失败（请确认已安装 cmake、pybind11 的 CMake 包与 g++）"; exit 1; }
make -j4 || { echo "make 失败"; exit 1; }
cd ..

echo ""
echo "编译完成！"
echo ""
if [ -f verify_build.sh ]; then
    echo "验证编译结果..."
    bash verify_build.sh
else
    echo "（无 verify_build.sh，跳过验证）"
fi