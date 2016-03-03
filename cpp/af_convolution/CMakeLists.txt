cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(ArrayFire)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

#... [gather source files, etc.]

MESSAGE(STATUS "CPU: ${ArrayFire_CPU_FOUND}")
MESSAGE(STATUS "CUDA: ${ArrayFire_CUDA_FOUND}")
MESSAGE(STATUS "CUDA LIBRARIES: ${ArrayFire_CUDA_LIBRARIES}")
MESSAGE(STATUS "OpenCL: ${ArrayFire_OpenCL_FOUND}")
MESSAGE(STATUS "Libraries: ${ArrayFire_LIBRARIES}")

# If you intend to use OpenCL, you need to find it
#FIND_PACKAGE(OpenCL)
#SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARIES})

# Or if you intend to use CUDA, you need it as well as NVVM:
#FIND_PACKAGE(CUDA)
#FIND_PACKAGE(NVVM) # this FIND script can be found in the ArrayFire CMake example repository
#SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB})

ADD_EXECUTABLE(my_executable main.cpp)
TARGET_LINK_LIBRARIES(my_executable ${ArrayFire_LIBRARIES}) # ${ArrayFire_CUDA_LIBRARIES})
