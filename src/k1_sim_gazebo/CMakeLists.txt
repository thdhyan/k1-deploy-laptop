cmake_minimum_required(VERSION 3.16)
project(k1_sim_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz_bridge REQUIRED)

# Install launch files
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

# Install world files
install(DIRECTORY worlds/ DESTINATION share/${PROJECT_NAME}/worlds)

# Install config files
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# Install mesh files (if any)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/meshes")
  install(DIRECTORY meshes/ DESTINATION share/${PROJECT_NAME}/meshes)
endif()

ament_package()
