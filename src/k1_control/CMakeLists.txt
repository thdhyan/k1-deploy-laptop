cmake_minimum_required(VERSION 3.8)
project(k1_control LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(k1_interfaces REQUIRED)

# Booster Robotics SDK - include headers and link static library directly
# SDK is at ../../sdk/booster_robotics_sdk relative to this package
set(BOOSTER_SDK_ROOT "${CMAKE_SOURCE_DIR}/../../sdk/booster_robotics_sdk")
set(BOOSTER_SDK_INCLUDE "${BOOSTER_SDK_ROOT}/include")

# Determine architecture
if(NOT CMAKE_SYSTEM_PROCESSOR)
  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
  set(BOOSTER_SDK_ARCH "x86_64")
else()
  set(BOOSTER_SDK_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(BOOSTER_SDK_LIB "${BOOSTER_SDK_ROOT}/lib/${BOOSTER_SDK_ARCH}/libbooster_robotics_sdk.a")

if(NOT EXISTS "${BOOSTER_SDK_LIB}")
  message(FATAL_ERROR "Booster Robotics SDK library not found: ${BOOSTER_SDK_LIB}")
endif()

include_directories(BEFORE ${BOOSTER_SDK_INCLUDE})

# Include directories
include_directories(include)

# SDK Bridge Node (C++ component)
add_library(sdk_bridge_node SHARED
  k1_control/sdk_bridge_node.cpp
)

# Link SDK library directly (plain signature - before ament_target_dependencies)
target_link_libraries(sdk_bridge_node ${BOOSTER_SDK_LIB} ${CMAKE_DL_LIBS} rt pthread)

ament_target_dependencies(sdk_bridge_node
  rclcpp
  rclcpp_components
  sensor_msgs
  k1_interfaces
)

rclcpp_components_register_nodes(sdk_bridge_node "k1_control::SdkBridgeNode")

# Install C++ library
install(TARGETS sdk_bridge_node
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install Python scripts (manually since we're using ament_cmake)
install(PROGRAMS
  k1_control/sim_bridge_node.py
  k1_control/joint_state_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install resource marker
install(DIRECTORY resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()