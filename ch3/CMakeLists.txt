cmake_minimum_required(VERSION 3.2)
project(chapter3)

set(EXPORT_COMPILE_COMMANDS ON)
# 新版本的 Pangolin 不能使用 c++11，最低也要14；
# 所以要么如下所示换成c++14，要么就不主动指定版本
# set(CMAKE_CXX_FLAGS "-std=c++14")

# 加入各子工程
add_subdirectory(useEigen)
add_subdirectory(useGeometry)
add_subdirectory(visualizeGeometry)
add_subdirectory(examples)
