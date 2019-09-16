#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsensePclSegmentation)

find_package( PCL 1.9 REQUIRED )
 
if (PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  link_directories(${PCL_LIBRARY_DIRS})
endif()

set(DEPENDENCIES realsense2 ${PCL_LIBRARIES})

find_package(OpenGL)
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "\n\n OpenGL package is missing!\n\n")
endif()

list(APPEND DEPENDENCIES ${OPENGL_LIBRARIES})

if(WIN32)
    list(APPEND DEPENDENCIES glfw3)
else()
    find_package(glfw3 REQUIRED)
    list(APPEND DEPENDENCIES glfw)
endif()

add_executable(rs_pcl_segmentation rs_pcl_segmentation.cpp)
target_link_libraries(rs_pcl_segmentation ${DEPENDENCIES})