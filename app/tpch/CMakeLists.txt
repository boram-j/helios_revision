# Helper macro to create executables easily
macro(create_tpch_query NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE nshedb)
endmacro()

create_tpch_query(tpch1  tpch1.cpp)
create_tpch_query(tpch4  tpch4.cpp)
create_tpch_query(tpch6  tpch6.cpp)
create_tpch_query(tpch8  tpch8.cpp)
create_tpch_query(tpch12 tpch12.cpp)
create_tpch_query(tpch14 tpch14.cpp)
create_tpch_query(tpch17 tpch17.cpp)
create_tpch_query(tpch19 tpch19.cpp)