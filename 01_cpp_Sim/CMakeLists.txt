cmake_minimum_required(VERSION 3.16)
project(spde_pde CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# -------------------------------------------------------------------
# Find dependencies
# -------------------------------------------------------------------
find_package(ZLIB REQUIRED)

# -------------------------------------------------------------------
# Library
# -------------------------------------------------------------------
add_library(spde STATIC
    src/simulate_pde.cpp
)
target_include_directories(spde PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -------------------------------------------------------------------
# Executable
# -------------------------------------------------------------------
add_executable(spde_run
    src/main.cpp
)
target_link_libraries(spde_run PRIVATE spde ZLIB::ZLIB)

# -------------------------------------------------------------------
# Fast math / optimization flags
# -------------------------------------------------------------------
if (MSVC)
    target_compile_options(spde PRIVATE /O2)
    target_compile_options(spde_run PRIVATE /O2)
else()
    target_compile_options(spde PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(spde_run PRIVATE -O3 -march=native -ffast-math)
endif()

add_library(spde_capi SHARED src/spde_capi.cpp)
target_link_libraries(spde_capi PRIVATE spde ZLIB::ZLIB)
target_include_directories(spde_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(spde_capi PROPERTIES OUTPUT_NAME "spde_capi")
