CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(yolov5_trt)

SET(CMAKE_CXX_STANDARD 14)
OPTION(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
SET(CMAKE_BUILD_TYPE Debug)

ADD_DEFINITIONS(-O2 -pthread)

# TORCH
set(Torch_DIR /home/ljh/Documents/action_detect/ASGCN_Cpp/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
IF (CUDA_FOUND)
    MESSAGE(STATUS "CUDA library status:")
    MESSAGE(STATUS "    version: ${CUDA_VERSION}")
    MESSAGE(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
ELSE ()
    MESSAGE(FATAL_ERROR "Could not locate CUDA")
ENDIF ()
SET(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++14;-g;-G;-gencode;arch=compute_61;code=sm_61)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
LINK_DIRECTORIES(/usr/local/cuda/lib64)

# 添加tensorrt
include_directories(/home/ljh/Downloads/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6/TensorRT-7.0.0.11/include/)
link_directories(/home/ljh/Downloads/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6/TensorRT-7.0.0.11/lib/)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# Eigen
SET(Eigen3_DIR /home/bruce/Programs/eigen-3.3.8/build/install/share/eigen3/cmake)
FIND_PACKAGE(Eigen3 REQUIRED)
IF (Eigen3_FOUND)
    MESSAGE(STATUS "Eigen3 library status:")
    MESSAGE(STATUS "    version: ${Eigen3_VERSION}")
    MESSAGE(STATUS "    include path: ${EIGEN3_INCLUDE_DIRS}")
ELSE ()
    MESSAGE(FATAL_ERROR "Could not locate Eigen3")
ENDIF ()
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})

# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    MESSAGE(STATUS "OpenCV library status:")
    MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
    MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
ELSE ()
    MESSAGE(FATAL_ERROR "Could not locate OpenCV")
ENDIF ()
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorflow/contrib/makefile/downloads/flatbuffers/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/ActRecognize
        )

link_directories(${OpenCV_LIBRARIES_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR}/lib)


INCLUDE_DIRECTORIES(include
                    include/detect
                    include/track
                    include/svip
                    include/md5
                    include/pose_est
                    include/utils
                    include/ActRecognize)

FILE(GLOB LIB_SOURCE_FILES
        src/detect/*.cpp
        src/detect/*.cu
        src/track/*.cpp
        src/svip/*.cpp
#        src/extract/*.cpp
        src/md5/*.cpp
        src/pose_est/*.cpp
        src/utils/*.cpp
        src/ActRecognize/*.cpp)

# yolov5 + sort + pose_extra
# asgcn

CUDA_ADD_EXECUTABLE(yolov5_trt
                    src/demo.cpp
                    ${LIB_SOURCE_FILES})


TARGET_LINK_LIBRARIES(yolov5_trt nvinfer cudart ${OpenCV_LIBS}
                 ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtensorflowlite.so
                 ${CMAKE_CURRENT_SOURCE_DIR}/lib/libInferAsgcn.a
                 ${TORCH_LIBRARIES}
                    )
