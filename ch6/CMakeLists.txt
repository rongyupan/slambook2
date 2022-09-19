cmake_minimum_required(VERSION 3.2)
project(ch6)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# 修复 OPENGL 在 cmake 时的警告
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif(POLICY CMP0072)

# OpenCV
set( OpenCV_DIR "/home/rypan/repos/opencv/install/lib/cmake/opencv4" )
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
set( Ceres_DIR "/home/rypan/repos/ceres-solver/install/lib/cmake/Ceres" )
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
set( g2o_DIR "/home/rypan/repos/g2o/install/lib/cmake/g2o" )
find_package(g2o REQUIRED)
# include_directories(${g2o_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} g2o::core g2o::stuff)
