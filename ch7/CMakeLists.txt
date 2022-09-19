cmake_minimum_required(VERSION 3.2)
project(vo1)

# 解决 OPENGL 警告
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif(POLICY CMP0072)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#（1）C++>=14，或者用默认的17；
#（2）-msse4 是必须的；
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")

set(OpenCV_DIR "/home/rypan/repos/opencv/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(g2o_DIR "/home/rypan/repos/g2o/install/lib/cmake/g2o")
find_package(g2o REQUIRED)

set(Sophus_DIR "/home/rypan/repos/sophus/install/share/sophus/cmake")
find_package(Sophus REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS} "/usr/include/eigen3/")

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d g2o::core g2o::stuff ${OpenCV_LIBS}
                      Sophus::Sophus)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d g2o::core g2o::stuff ${OpenCV_LIBS}
                      Sophus::Sophus)
