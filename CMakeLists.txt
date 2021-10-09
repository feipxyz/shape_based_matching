cmake_minimum_required(VERSION 3.0)
# set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)


# debug or release
SET(CMAKE_BUILD_TYPE "Release")


# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
#    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
ENDIF()

# SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD
# SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
SET(COMMON_COMPILE_FLAGS "-Wall -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")


# opencv
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/ros/kinetic)
find_package(OpenCV 4 REQUIRED)
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/MIPP/)


# link_directories(
#     ${PROTOBUF_LIB_DIR}
# )


# test exe
# add_executable(${PROJECT_NAME}_test ${PROJECT_SOURCE_DIR}/proto/utils.pb.cc ${PROJECT_SOURCE_DIR}/line2Dup.cpp ${PROJECT_SOURCE_DIR}/test.cpp)
# target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})
add_executable(${PROJECT_NAME}_my_test ${PROJECT_SOURCE_DIR}/line2Dup.cpp ${PROJECT_SOURCE_DIR}/my_test.cpp)
add_executable(${PROJECT_NAME}_test ${PROJECT_SOURCE_DIR}/line2Dup.cpp ${PROJECT_SOURCE_DIR}/test.cpp)
target_link_libraries(${PROJECT_NAME}_my_test ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS})
