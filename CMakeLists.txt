cmake_minimum_required(VERSION 3.16)
project(NSHEDB LANGUAGES CXX)

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