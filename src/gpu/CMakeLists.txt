# src/gpu/CMakeLists.txt — builds the nshedb_gpu static library.
#
# The Phantom target is named "Phantom" (from phantom-fhe/src/CMakeLists.txt:
#   add_library(Phantom SHARED ...))
# When installed it exports as phantom::Phantom, but when added via
# add_subdirectory it is just "Phantom".

cmake_minimum_required(VERSION 3.20)

# -----------------------------------------------------------------------
# Collect sources
# -----------------------------------------------------------------------
set(NSHEDB_GPU_SOURCES
    core/comparator_gpu.cu
    predicates/predicates_gpu.cu
    utils/math_utils_gpu.cpp       # pure C++ — no CUDA needed
)

add_library(nshedb_gpu STATIC ${NSHEDB_GPU_SOURCES})

# -----------------------------------------------------------------------
# Include paths
# -----------------------------------------------------------------------
target_include_directories(nshedb_gpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------------------------------
# Phantom include path (headers are in phantom-fhe/include/)
# Phantom itself sets this up when added via add_subdirectory, but we
# add it explicitly as a SYSTEM include to suppress warnings from
# Phantom's own CUDA headers.
# -----------------------------------------------------------------------
target_include_directories(nshedb_gpu SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include
)

# -----------------------------------------------------------------------
# GeminiHE (HECache) include path
#
# nshedb_gpu uses hecache::HECache from GeminiHE for ciphertext VRAM
# management (comparator_gpu.cu).  Only header-only parts are needed:
#   hecache/hecache.h, hecache/pinned_buffer.h, hecache/cpu_keyswitch.h
# No GeminiHE library target needs to be linked.
#
# Set GEMINIHE_INCLUDE_DIR on the cmake command line to override:
#   cmake -DGEMINIHE_INCLUDE_DIR=/path/to/geminiHE/include ...
# Default assumes GeminiHE lives adjacent to helios_revision/:
#   <project-root>/geminihe_files/geminiHE/include/
# -----------------------------------------------------------------------
if(NOT DEFINED GEMINIHE_INCLUDE_DIR)
    set(GEMINIHE_INCLUDE_DIR
        "${CMAKE_SOURCE_DIR}/../geminihe_files/geminiHE/include"
        CACHE PATH "Path to GeminiHE include directory (contains hecache/hecache.h)")
endif()

if(NOT EXISTS "${GEMINIHE_INCLUDE_DIR}/hecache/hecache.h")
    message(WARNING
        "GeminiHE HECache header not found at ${GEMINIHE_INCLUDE_DIR}/hecache/hecache.h.\n"
        "HECache integration will fail to compile.  Either:\n"
        "  (a) Set -DGEMINIHE_INCLUDE_DIR=<path> to the correct location, or\n"
        "  (b) Remove #include \"hecache/hecache.h\" from comparator_gpu.cu\n"
        "      and set the HECache runtime guard to always return false.")
endif()

target_include_directories(nshedb_gpu SYSTEM PRIVATE
    ${GEMINIHE_INCLUDE_DIR}
)

# -----------------------------------------------------------------------
# Link Phantom
# The exported target from phantom-fhe is "Phantom" (shared library).
# -----------------------------------------------------------------------
target_link_libraries(nshedb_gpu PUBLIC Phantom)

# -----------------------------------------------------------------------
# CUDA properties
# -----------------------------------------------------------------------
set_target_properties(nshedb_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD              17
    CUDA_STANDARD_REQUIRED     ON
)

target_compile_features(nshedb_gpu PUBLIC cxx_std_17)

# Pass --default-stream per-thread to match how Phantom is compiled
# (see phantom-fhe/src/CMakeLists.txt)
target_compile_options(nshedb_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
)
