cmake_minimum_required(VERSION 2.8.3)
project(computer_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

## Declare the service files to be processed
add_service_files(
  FILES
  LaneBehaviorCMD.srv  # Your service file
)

## Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs
)
