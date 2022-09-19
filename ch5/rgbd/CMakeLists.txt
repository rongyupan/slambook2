set(Sophus_DIR "/home/rypan/repos/sophus/install/share/sophus/cmake")
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set( Pangolin_DIR "/home/rypan/repos/pangolin/install/lib/cmake/Pangolin" )
find_package(Pangolin REQUIRED)

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Sophus::Sophus)