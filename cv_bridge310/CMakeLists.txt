cmake_minimum_required(VERSION 3.0.2)
project(cv_bridge310)

find_package(catkin REQUIRED COMPONENTS rosconsole sensor_msgs)

if(NOT ANDROID)
  find_package(PythonLibs)

  # if(PYTHONLIBS_VERSION_STRING VERSION_LESS "3.8")
  #   # Debian Buster
  #   find_package(Boost REQUIRED python37)
  # else()
  #   # Ubuntu Focal
    # find_package(Boost REQUIRED python)
    # ros noetic 需要的boost是1.71.0版本，而我自己之前又在机器上安装了boost1.70.0
    # 因此这里指定了boost的具体版本为1.71.0
    find_package(Boost 1.71.0 REQUIRED python)
  # endif()
else()
find_package(Boost REQUIRED)
endif()

# added by wxliu on 2024-6-17.
# set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
# set(OpenCV_DIR /root/opencv4_install_path/lib/cmake/opencv4)
# the end.

# comment opencv4 on 2024-6-17
# set(_opencv_version 4)
# find_package(OpenCV 4 QUIET)
# if(NOT OpenCV_FOUND)
#   message(STATUS "Did not find OpenCV 4, trying OpenCV 3")
#   set(_opencv_version 3)
# endif()
# the end.

# 2024-6-17
# set(OpenCV_DIR /root/opencv310_install_path/lib/cmake/opencv) # find opencv3
# the end.

# find_package(OpenCV ${_opencv_version} REQUIRED
# change this line on 2024-6-17
find_package(OpenCV 3.1.0 REQUIRED
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
  CONFIG
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS rosconsole sensor_msgs
  DEPENDS OpenCV
  CFG_EXTRAS cv_bridge-extras.cmake
)

catkin_python_setup()

include_directories(include ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if(NOT ANDROID)
add_subdirectory(python)
endif()
add_subdirectory(src)
if(CATKIN_ENABLE_TESTING)
  add_subdirectory(test)
endif()

# install the include folder
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
