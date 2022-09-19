# 声明要求的 cmake 最低版本
cmake_minimum_required(VERSION 3.2)

# 声明一个 cmake 工程
project(HelloSLAM)

# 设置编译模式
# 不调试的话可以设置为 Relase 模式进行提速
set(CMAKE_BUILD_TYPE "Debug")

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
add_executable(helloSLAM helloSLAM.cpp)

# 添加hello库：libhello.a
add_library(hello libHelloSLAM.cpp)
# 共享库：libhello_shared.so
add_library(hello_shared SHARED libHelloSLAM.cpp)

# 添加可执行程序调用hello库中函数
add_executable(useHello useHello.cpp)
# 将库文件链接到可执行程序上
target_link_libraries(useHello hello_shared)
