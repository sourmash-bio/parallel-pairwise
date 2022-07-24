# ----------------------------------------------------------------------------
# parallel-pairwise
# ----------------------------------------------------------------------------

# Minimum cmake version
cmake_minimum_required (VERSION 3.14...3.20)
# Name of project and that it is C/C++.
project (
        kSpider
        LANGUAGES CXX C
        VERSION 2.0.0
)

set(default_build_type "Release")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lstdc++fs -fPIC -lgomp -lrt -fopenmp -O3 -Ofast")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/lib/progressbar/include")
include_directories("${PROJECT_SOURCE_DIR}/lib/json_parser")
add_subdirectory("lib/zstr")
include_directories("${PROJECT_SOURCE_DIR}/lib/zstr/src")

set(PHMAP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/lib/parallel-hashmap")
include_directories("${PHMAP_INCLUDE_DIRS}")

add_executable(pairwise main.cpp)
target_link_libraries(pairwise z)
