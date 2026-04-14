cmake_minimum_required( VERSION 3.22 )
project( SLIC_superpixel )

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_CXX_FLAGS "-O3 -fopenmp -ffast-math -fopt-info-vec-optimized" )

add_compile_definitions("PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

find_package( OpenCV REQUIRED )
include_directories(
        ${OpenCV_INCLUDE_DIRS}
)
link_libraries(${OpenCV_LIBS})

add_executable( v1
        v1/main-v1.cpp
)

add_executable( v2
        v2/common.cpp
        v2/common.h
        v2/run-configurations.cpp
        v2/SLIC_sequential.cpp
        v2/SLIC_sequential.h
        v2/SLIC_parallel.cpp
        v2/SLIC_parallel.h
)

add_executable( v3
        v3/common.cpp
        v3/common.h
        v3/run-configurations.cpp
        v3/SLIC_sequential.cpp
        v3/SLIC_sequential.h
        v3/SLIC_parallel.cpp
        v3/SLIC_parallel.h
)

add_executable( v4
        v4/common.cpp
        v4/common.h
        v4/run-configurations.cpp
        v4/SLIC_sequential.cpp
        v4/SLIC_sequential.h
        v4/SLIC_parallel.cpp
        v4/SLIC_parallel.h
        v4/SLIC_tile.cpp
        v4/SLIC_tile.h
)
add_executable(output_check
        v4/output_checker.cpp
        v4/common.cpp
        v4/common.h
        v4/SLIC_sequential.cpp
        v4/SLIC_sequential.h
        v4/SLIC_parallel.cpp
        v4/SLIC_parallel.h
        v4/SLIC_tile.cpp
        v4/SLIC_tile.h
)