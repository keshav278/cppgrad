cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)

project(cppgrad)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.1.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

add_executable(mnist mnist.cpp)
add_executable(mnist_gui mnist_gui.cpp)

target_link_libraries(mnist_gui PRIVATE SFML::Graphics)