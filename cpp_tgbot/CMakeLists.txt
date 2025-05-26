cmake_minimum_required(VERSION 3.15)
project(tg-humanbot)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS system thread)
find_package(OpenSSL REQUIRED)

add_executable(bot main.cpp)

target_link_libraries(bot PRIVATE Boost::system Boost::thread OpenSSL::SSL OpenSSL::Crypto curl)