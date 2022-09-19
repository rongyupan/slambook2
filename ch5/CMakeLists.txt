cmake_minimum_required(VERSION 3.2)

project(ch5)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 使用了 Pangolin 库，所以要么为 c++14 要么不指定
# set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
set( OpenCV_DIR "/home/rypan/repos/opencv/install/lib/cmake/opencv4" )
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)
