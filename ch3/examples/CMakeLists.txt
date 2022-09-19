include_directories("/usr/include/eigen3")
add_executable(coordinateTransform coordinateTransform.cpp)

set( Pangolin_DIR "/home/rypan/repos/pangolin/install/lib/cmake/Pangolin" )
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
