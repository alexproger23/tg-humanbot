cmake_minimum_required(VERSION 3.10)
project(telegram_bot)

set(CMAKE_CXX_STANDARD 17)

find_package(CURL REQUIRED)
find_package(Boost COMPONENTS system thread asio REQUIRED)
find_package(tgbot-cpp REQUIRED)

add_executable(bot main.cpp)

target_link_libraries(bot
    tgbot-cpp
    CURL::libcurl
    Boost::system
    Boost::thread
    Boost::asio
    pthread
    ssl
    crypto
)