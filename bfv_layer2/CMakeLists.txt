cmake_minimum_required(VERSION 3.20)

# ============================================================
# BFV Layer 2-1 — context, GPU types, batch encoder
#
# No FHE dependency. Builds on top of libbfv_core (Layer 1).
# ============================================================

project(bfv_layer2_1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA architecture ─────────────────────────────────────────
# Default: sm_89 (RTX 4090 / H100 NVL). Override with:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..   (A100)
#   cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..   (RTX 3090)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
message(STATUS "bfv_layer2_1: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

# ── Output directories ────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# ── Pull in Layer 1 as a subdirectory ────────────────────────
# This creates the `bfv_core` CMake target.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../bfv_core
                 ${CMAKE_CURRENT_BINARY_DIR}/bfv_core_build)

# ── Common NVCC flags ─────────────────────────────────────────
set(BFV_NVCC_FLAGS
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler=-Wall
)

# ── libbfv_layer2_1: static library ──────────────────────────
add_library(bfv_layer2_1 STATIC
    bfv_context.cu
    bfv_types.cu
    bfv_encode.cu
)

target_include_directories(bfv_layer2_1 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../bfv_core
)

target_link_libraries(bfv_layer2_1 PUBLIC bfv_core)

set_target_properties(bfv_layer2_1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

target_compile_options(bfv_layer2_1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${BFV_NVCC_FLAGS}>
)

# ── bfv_layer2_1_verify: test executable ─────────────────────
add_executable(bfv_layer2_1_verify
    verify/verify_layer2_1.cu
)

target_link_libraries(bfv_layer2_1_verify PRIVATE bfv_layer2_1)

target_include_directories(bfv_layer2_1_verify PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../bfv_core
)

set_target_properties(bfv_layer2_1_verify PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(bfv_layer2_1_verify PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${BFV_NVCC_FLAGS}>
)

# ── Summary ───────────────────────────────────────────────────
message(STATUS "bfv_layer2_1 targets:")
message(STATUS "  libbfv_layer2_1      — static library")
message(STATUS "  bfv_layer2_1_verify  — correctness verifier")
message(STATUS "")
message(STATUS "Build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. [-DCMAKE_CUDA_ARCHITECTURES=<sm>]")
message(STATUS "  make -j")
message(STATUS "Run:")
message(STATUS "  ./bin/bfv_layer2_1_verify")
