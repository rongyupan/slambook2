cmake_minimum_required(VERSION 3.2)
project(ch8)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 ${SSE_FLAGS} -g -O3 -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OpenCV_DIR "/home/rypan/repos/opencv/install/lib/cmake/opencv4")
set(Sophus_DIR "/home/rypan/repos/sophus/install/share/sophus/cmake")
set(Pangolin_DIR "/home/rypan/repos/pangolin/install/lib/cmake/Pangolin")
find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS} "/usr/include/eigen3/")

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
                      Sophus::Sophus)
