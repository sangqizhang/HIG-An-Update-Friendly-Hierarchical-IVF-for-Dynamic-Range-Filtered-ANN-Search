# 层次化 Ada-IVF（pybind11），输出到 src/hierarchical_ivf/
# 与 hybridsearch/Hierarchical_ivf 的区别：不链接 FAISS；不含 pybind_wrapper（单层另有 setup.py 构建）
cmake_minimum_required(VERSION 3.5)
project(rbplsm_hierarchical LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckCXXCompilerFlag)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# AVX/FMA 开关：用于显式启用向量化指令路径（影响 hierarchical 模块）
option(ENABLE_AVX "Enable AVX/FMA compile flags for hierarchical modules" ON)
set(RBPLSM_AVX_FLAGS "")
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    list(APPEND RBPLSM_AVX_FLAGS "-march=native")
endif()
if(ENABLE_AVX)
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_MAVX)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_MFMA)
    if(COMPILER_SUPPORTS_MAVX)
        list(APPEND RBPLSM_AVX_FLAGS "-mavx")
    endif()
    if(COMPILER_SUPPORTS_MFMA)
        list(APPEND RBPLSM_AVX_FLAGS "-mfma")
    endif()
    if(RBPLSM_AVX_FLAGS)
        message(STATUS "AVX flags enabled: ${RBPLSM_AVX_FLAGS}")
    else()
        message(WARNING "ENABLE_AVX=ON but compiler does not support -mavx/-mfma")
    endif()
endif()

execute_process(
    COMMAND python3 -c "import pybind11; import os; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_cmake_rc
)
if(_pybind11_cmake_rc EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: found")
else()
    message(STATUS "OpenMP: not found (OpenMP pragmas compile as single-threaded)")
endif()

set(HIER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hierarchical_ada_ivf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ada_ivf_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simd_utils.cpp
)

set(_hier_out ${CMAKE_CURRENT_SOURCE_DIR}/src/hierarchical_ivf)

function(rbplsm_add_hier_target _target _modtoken _soname)
    pybind11_add_module(${_target} MODULE ${HIER_SOURCES})
    target_compile_definitions(${_target} PRIVATE
        HIER_PYBIND_MODULE_NAME=${_modtoken}
        PYBIND11_SIMPLE_GIL_MANAGEMENT)
    target_include_directories(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /usr/include)
    target_compile_definitions(${_target} PRIVATE _GNU_SOURCE)
    if(RBPLSM_AVX_FLAGS)
        target_compile_options(${_target} PRIVATE ${RBPLSM_AVX_FLAGS})
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${_target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(${_target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${_hier_out}
        OUTPUT_NAME ${_soname}
    )
endfunction()

# import hierarchical_ada_ivf_cpp_v2
rbplsm_add_hier_target(hierarchical_ada_ivf_cpp_mod hierarchical_ada_ivf_cpp_v2 hierarchical_ada_ivf_cpp_v2)

# import hierarchical_ada_ivf_cpp（旧名，同源码仅模块名不同）
rbplsm_add_hier_target(hierarchical_ada_ivf_cpp_legacy hierarchical_ada_ivf_cpp hierarchical_ada_ivf_cpp)
