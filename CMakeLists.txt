cmake_minimum_required( VERSION 3.22 )
project( SLIC_superpixel_seq )

set( CMAKE_CXX_STANDARD 17 )
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fno-omit-frame-pointer -march=native -ffast-math " )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -march=native -ffast-math " )

find_package( OpenCV REQUIRED )
include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable( SLIC_superpixel_seq common.cpp run-configurations.cpp SLIC_sequential.cpp SLIC_parallel.cpp
        SLIC_parallel.h
        SLIC_sequential.h
        common.h
)
#add_executable( SLIC_superpixel_seq SLIC_sequential.cpp )
#add_executable( SLIC_superpixel_seq main-v1.cpp )

target_link_libraries( SLIC_superpixel_seq ${OpenCV_LIBS} )

target_compile_features( SLIC_superpixel_seq PRIVATE cxx_std_17 )
target_compile_definitions( SLIC_superpixel_seq PRIVATE
        "PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\""
)
