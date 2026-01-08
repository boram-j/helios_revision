# Helper macro for OMP targets
macro(create_omp_query NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE nshedb OpenMP::OpenMP_CXX)
endmacro()

create_omp_query(tpch1_omp  tpch1_omp.cpp)
create_omp_query(tpch4_omp  tpch4_omp.cpp)
create_omp_query(tpch6_omp  tpch6_omp.cpp)
create_omp_query(tpch8_omp  tpch8_omp.cpp)
create_omp_query(tpch12_omp tpch12_omp.cpp)
create_omp_query(tpch14_omp tpch14_omp.cpp)
create_omp_query(tpch17_omp tpch17_omp.cpp)
create_omp_query(tpch19_omp tpch19_omp.cpp)