# HELIOS bucket benchmark
# Mirrors the structure of app/tpch/CMakeLists.txt

macro(create_helios_bench NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE nshedb)
endmacro()

create_helios_bench(helios_bucket_bench helios_bucket_bench.cpp)
