cmake_minimum_required(VERSION 3.16)
project(safesite_object_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0>=1.16 gstreamer-app-1.0>=1.16)
find_package(OpenCV REQUIRED core imgproc highgui dnn)

add_executable(object_detection
    src/object_detection.cpp
)

target_include_directories(object_detection PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(object_detection PRIVATE
    ${GSTREAMER_LIBRARY_DIRS}
)

target_link_libraries(object_detection PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${OpenCV_LIBS}
)

target_compile_options(object_detection PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
)

