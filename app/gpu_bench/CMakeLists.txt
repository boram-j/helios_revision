# app/gpu_bench/CMakeLists.txt — builds the GPU benchmark driver.

cmake_minimum_required(VERSION 3.20)

add_executable(helios_gpu_bench helios_gpu_bench.cu)

target_include_directories(helios_gpu_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include
)

# -----------------------------------------------------------------------
# GeminiHE (HECache) include path
#
# helios_gpu_bench.cu includes hecache/hecache.h directly (for
# HECache::initialize and HECache::instance().print_stats).
#
# GEMINIHE_INCLUDE_DIR is set (and cached) by src/gpu/CMakeLists.txt
# when nshedb_gpu is built first; if not, define it here with the same
# default so this target can be built standalone.
# -----------------------------------------------------------------------
if(NOT DEFINED GEMINIHE_INCLUDE_DIR)
    set(GEMINIHE_INCLUDE_DIR
        "${CMAKE_SOURCE_DIR}/../geminihe_files/geminiHE/include"
        CACHE PATH "Path to GeminiHE include directory (contains hecache/hecache.h)")
endif()

target_include_directories(helios_gpu_bench SYSTEM PRIVATE
    ${GEMINIHE_INCLUDE_DIR}
)

target_link_libraries(helios_gpu_bench PRIVATE
    nshedb_gpu
    Phantom
    CUDA::cudart
)

set_target_properties(helios_gpu_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD              17
    CUDA_STANDARD_REQUIRED     ON
)

target_compile_options(helios_gpu_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
)
