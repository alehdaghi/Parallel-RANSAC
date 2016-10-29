
project(Parallel)


find_package(CUDA REQUIRED)
add_subdirectory("${PROJECT_SOURCE_DIR}/src/myRansac")
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} /usr/local/cuda/lib64/libcudart.so /usr/lib/libcuda.so /usr/local/cuda/lib64/libcublas.so)
MESSAGE( STATUS "SOURCE: " ${CUDA_LIBRARIES} )
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_VERBOSE_BUILD ON)
SET(CUDA_HOST_COMPILER g++)




find_package(PCL 1.7 REQUIRED COMPONENTS common io search)
find_package(Boost 1.5 REQUIRED COMPONENTS system filesystem regex chrono timer )
find_package(OpenCV REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})



#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
#if (UNIX OR APPLE)
#        set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler -fPIC;")
#endif()


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/egbis")

add_executable(PRANSAC ${SOURCE} )
#cuda_add_executable(PRANSAC ${SOURCE} ${CUDA_SOURCES})
target_link_libraries(PRANSAC   ${PCL_COMMON_LIBRARIES}
                                ${PCL_IO_LIBRARIES}
                                ${PCL_SEARCH_LIBRARIES}
                                ${OpenCV_LIBS}
                                ${Boost_SYSTEM_LIBRARIES}
                                ${Boost_TIMER_LIBRARIES}
                                ${kernelRANSAC}
)
