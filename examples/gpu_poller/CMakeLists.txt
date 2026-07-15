cmake_minimum_required(VERSION 3.16)
project(gpu_poller LANGUAGES CXX CUDA)

# Find CUDA - use traditional variables for compatibility
find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gpu_poller gpu_poller.cpp)

# Use traditional CUDA variables for maximum compatibility
target_include_directories(gpu_poller PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu_poller PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_LIBRARY})

# Install
install(TARGETS gpu_poller DESTINATION bin)