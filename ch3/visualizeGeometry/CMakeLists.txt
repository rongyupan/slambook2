cmake_minimum_required( VERSION 3.2 )
project( visualizeGeometry )

# set(CMAKE_CXX_FLAGS "-std=c++14")

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

# 添加Pangolin依赖
set( Pangolin_DIR "/home/rypan/repos/pangolin/install/lib/cmake/Pangolin" )
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable( visualizeGeometry visualizeGeometry.cpp )
target_link_libraries( visualizeGeometry ${Pangolin_LIBRARIES} )
