#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun computer_vision lane_following.py

#rosrun computer_vision lane_following_controller.py

# wait for app to end
dt-launchfile-join