cmake_minimum_required(VERSION 3.16)

# ============================================================
# GPU option — must be declared before project() so the CUDA
# language is only enabled when actually needed.
# ============================================================
option(BUILD_GPU "Build GPU backend using Phantom-FHE (requires CUDA)" OFF)

if(BUILD_GPU)
    cmake_minimum_required(VERSION 3.20)   # CUDA language support improved in 3.20
    project(NSHEDB LANGUAGES CXX CUDA)
else()
    project(NSHEDB LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3") # Added -O3 for performance

# --- Output Directories ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# --- Dependencies ---
# 1. OpenMP (Needed for your omp/ folder)
find_package(OpenMP REQUIRED)

# 2. SEAL (Try local thirdparty first, then global)
option(SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT "Throw exception on transparent ciphertext" OFF)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/SEAL/CMakeLists.txt")
    add_subdirectory(thirdparty/SEAL)
else()
    find_package(SEAL REQUIRED)
endif()

# --- Build Targets ---
add_subdirectory(src)
add_subdirectory(app)

# ============================================================
# GPU backend (optional, gated by BUILD_GPU)
# ============================================================
if(BUILD_GPU)
    find_package(CUDAToolkit REQUIRED)

    # Phantom-FHE must be present as a git submodule at thirdparty/phantom-fhe.
    # To add it:
    #   git submodule add https://github.com/encryptorion-lab/phantom-fhe.git \
    #       thirdparty/phantom-fhe
    #   git submodule update --init thirdparty/phantom-fhe
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/phantom-fhe/CMakeLists.txt")
        # Disable Phantom's examples/bench/tests to speed up the build
        set(PHANTOM_ENABLE_EXAMPLE OFF CACHE BOOL "" FORCE)
        set(PHANTOM_ENABLE_BENCH   OFF CACHE BOOL "" FORCE)
        set(PHANTOM_ENABLE_TEST    OFF CACHE BOOL "" FORCE)
        add_subdirectory(thirdparty/phantom-fhe)
    else()
        message(FATAL_ERROR
            "Phantom-FHE not found at thirdparty/phantom-fhe.\n"
            "Run:\n"
            "  git submodule add "
            "https://github.com/encryptorion-lab/phantom-fhe.git "
            "thirdparty/phantom-fhe\n"
            "  git submodule update --init thirdparty/phantom-fhe")
    endif()

    # CUDA architecture: inherit from environment or default to native
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    add_subdirectory(src/gpu)
    add_subdirectory(app/gpu_bench)

    message(STATUS "GPU backend enabled (Phantom-FHE BFV)")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()