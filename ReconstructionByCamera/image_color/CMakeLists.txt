cmake_minimum_required(VERSION 3.0)
project(color)
set(CXX_STANDARD_REQUIRED 11)
set(Torch_DIR /home/lihao/Documents/cpp_project/deploy_with_libtorch/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(image_color source/image_color.cpp)
include_directories(include)
target_link_libraries(image_color "${TORCH_LIBRARIES}" ${OpenCV_LIBS})


