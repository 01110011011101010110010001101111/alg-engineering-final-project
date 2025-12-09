cmake_minimum_required(VERSION 3.10)
project(ConvexHull)

# Find CGAL
find_package(CGAL REQUIRED)
include_directories(${CGAL_INCLUDE_DIRS})

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# Create the executable
add_executable(convex_hull main.cpp)
add_executable(generate_bottom_square generate_bottom_square.cpp)

# Link libraries
target_link_libraries(convex_hull ${CGAL_LIBRARIES})
target_link_libraries(generate_bottom_square Eigen3::Eigen)

