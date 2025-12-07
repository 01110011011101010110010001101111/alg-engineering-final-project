cmake_minimum_required(VERSION 3.10)
project(GCS_Parallel_CPP)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find libraries (drake + eigen)
find_package(drake CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(classic_solver 
    src/classic_solver.cpp
    src/utils.cpp
)

# include directories (drake, eigen, local)
target_include_directories(classic_solver PRIVATE
    ${DRAKE_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    "include"
)

# link drake + eigen libraries
target_link_libraries(classic_solver
    drake::drake
    Eigen3::Eigen
)

# add o3 optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
