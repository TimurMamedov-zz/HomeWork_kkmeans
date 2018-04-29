cmake_minimum_required(VERSION 3.5)

if(DEFINED ENV{TRAVIS_BUILD_NUMBER})
    project(kkmeans VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})
else()
    project(kkmeans VERSION 0.0.1)
endif()

find_package(Boost REQUIRED)
find_package(dlib REQUIRED)

aux_source_directory(. SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST})

set_target_properties(kkmeans PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    COMPILE_OPTIONS -Wpedantic -Wall -Wextra
)

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    dlib::dlib)
