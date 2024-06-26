cmake_minimum_required(VERSION 2.8.3)
project(gvins_camera_model)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    )

# find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
# ubuntu20.04安装ros会安装boost1.71.0版本
# 由于我机器上手动安装了boost1.70.0, 导致如下警告：
# warning: libboost_filesystem.so.1.71.0, needed by /opt/ros/noetic/lib/libroscpp.so, may conflict with libboost_filesystem.so.1.70.0
# 因此这里指定boost具体版本号为1.71.0,警告就解决了。
find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

# find_package(OpenCV REQUIRED)
find_package(OpenCV 4 REQUIRED) # modified by wxliu on 2024-6-17. cv_bridge默认连接了4.2版本的opencv进行编译

# set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


catkin_package(
    INCLUDE_DIRS include
    LIBRARIES gvins_camera_model
    CATKIN_DEPENDS roscpp std_msgs
#    DEPENDS system_lib
    )

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

include_directories("include")


add_executable(gvins_Calibration 
    src/intrinsic_calib.cc
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

add_library(gvins_camera_model
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(gvins_Calibration ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(gvins_camera_model ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
