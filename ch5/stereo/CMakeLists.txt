set( Pangolin_DIR "/home/rypan/repos/pangolin/install/lib/cmake/Pangolin" )
find_package(Pangolin REQUIRED)

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})