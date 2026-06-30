cmake_minimum_required(VERSION 3.20)

# ============================================================
# BFV Core — Layer 1 CUDA math kernels
#
# No dependency on Phantom-FHE, SEAL, or any other FHE library.
# Foundation for a GeminiHE-friendly GPU BFV library.
# ============================================================

project(bfv_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA architecture ───────────────────────────────────────
# Default: sm_80 (A100/A30/A10). Override with:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..   (RTX 3090/A40)
#   cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..   (RTX 4090/H100 NVL)
#   cmake -DCMAKE_CUDA_ARCHITECTURES=90 ..   (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
message(STATUS "bfv_core: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

# ── Output directories ──────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# ── libbfv_core: static library ────────────────────────────
add_library(bfv_core STATIC
    ntt.cu
    poly_ops.cu
    rns.cu
    ks_ip.cu
    galois.cu
)

target_include_directories(bfv_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(bfv_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # Allow __uint128_t (supported on all 64-bit CUDA targets)
    CUDA_NVCC_FLAGS_RELEASE "-O3;-lineinfo"
    CUDA_NVCC_FLAGS_DEBUG   "-G;-g"
    POSITION_INDEPENDENT_CODE ON
)

# NVCC flags applied to all configurations
target_compile_options(bfv_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
    >
)

# ── bfv_layer1_verify: correctness verifier ────────────────
add_executable(bfv_layer1_verify
    verify/verify_layer1.cu
)

target_link_libraries(bfv_layer1_verify PRIVATE bfv_core)
target_include_directories(bfv_layer1_verify PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(bfv_layer1_verify PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(bfv_layer1_verify PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

# ── Summary ─────────────────────────────────────────────────
message(STATUS "bfv_core targets:")
message(STATUS "  libbfv_core         — static library")
message(STATUS "  bfv_layer1_verify   — correctness verifier")
message(STATUS "")
message(STATUS "Build with:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. [-DCMAKE_CUDA_ARCHITECTURES=<sm>]")
message(STATUS "  make -j")
message(STATUS "Run verifier:")
message(STATUS "  ./bin/bfv_layer1_verify           # small-N tests")
message(STATUS "  ./bin/bfv_layer1_verify --large   # + N=32768 test")
