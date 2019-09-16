cmake_minimum_required(VERSION 3.1.0)

project(pcl_tuts)

find_package(PCL 1.8 REQUIRED)

if (PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  message("<<${PCL_LIBRARY_DIRS}>>")
endif()

find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "\n\n OpenGL package is missing!\n\n")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(INCLUDES 
    ${PCL_INCLUDE_DIRS} 
)
	
set(DEPENDENCIES 
    ${PCL_LIBRARIES}
)

# Create a list of 
set(FILES 
	global_hypothesis_verification
)

foreach(FILE ${FILES})
#	message("<<${FILE}>>")
	add_executable(${FILE} ${FILE}.cpp)
	target_link_libraries(${FILE} ${DEPENDENCIES})
endforeach() 