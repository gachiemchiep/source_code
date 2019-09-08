cmake_minimum_required(VERSION 3.1.0)
project(cpp)

# Select c++ 14 for pcl
set (CMAKE_CXX_STANDARD 14)
# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# View the makefile commands during build
#set(CMAKE_VERBOSE_MAKEFILE on)

add_subdirectory(pcl)
