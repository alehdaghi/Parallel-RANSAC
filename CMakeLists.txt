cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(Parallel)

find_package(PCL 1.7 REQUIRED COMPONENTS common io)
find_package(OpenCV REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_subdirectory("${PROJECT_SOURCE_DIR}/egbis")
add_executable(PRANSAC src/normal.cpp)
target_link_libraries(PRANSAC ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OpenCV_LIBS})
