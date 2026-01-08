set(NSHEDB_SOURCES
    core/comparator.cpp
    predicates/predicates.cpp
    utils/math_utils.cpp
)

add_library(nshedb STATIC ${NSHEDB_SOURCES})

# --- Include Directories ---
# PUBLIC: Apps using nshedb can include "nshedb/nshedb.h"
# PRIVATE: Internal sources can find headers in src/
target_include_directories(nshedb
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link SEAL to the library
if(TARGET SEAL::seal)
    target_link_libraries(nshedb PUBLIC SEAL::seal)
else()
    target_link_libraries(nshedb PUBLIC seal)
endif()