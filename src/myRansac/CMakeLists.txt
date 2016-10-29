set(SOURCE
    ${SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/ParallelRansac.cpp
    PARENT_SCOPE
)

FIND_PACKAGE(CUDA REQUIRED)
set(CUDA_SOURCES
    ${CUDA_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
    PARENT_SCOPE
)
SET(CUDA_SOURCES  kernel.cu    )


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
SET(CUDA_VERBOSE_BUILD ON)
#SET(CUDA_INCLUDE_DIRS "/usr/local/cuda/include" ${CUDA_INCLUDE_DIRS})
#SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} "/usr/local/cuda/lib64/libcudart.so" "/usr/lib64/libcuda.so" "/usr/local/cuda/lib64/libcublas.so")

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

SET(LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
#CUDA_COMPILE(cuda_objs ${CUDA_SOURCES})
cuda_add_library(myLib ${CUDA_SOURCES} )
set(kernelRANSAC
    myLib
    PARENT_SCOPE
)
set(HEADERS
    ${HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ParallelRansac.h
    ${CMAKE_CURRENT_SOURCE_DIR}/typesCuda.h
    PARENT_SCOPE
)
