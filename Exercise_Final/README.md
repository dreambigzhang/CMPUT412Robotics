# Exercise 4: Apriltag Detection & Safety on Robots

This repository is built on the following template: [Duckietown Template-ROS](https://github.com/duckietown/template-ros/).

In this exercise, we programmed our duckiebot to detect, read, and react to Apriltags scattered through the environment using the Apriltag library.
Then the duckiebot detects crosswalks on the road, and wait for some unknown amount of time until they are empty to continue driving.
Finally, we implemented an autonomous safe navigation behavior on our Duckiebot. Specifically, while driving we must be able to detect a broken-down Duckiebot, pause, and maneuver around it

---

## Part 1: AprilTag Detection🛑

### AprilTag detection
**File:** `packages/ex4/src/apriltag_detection.py`  
**Description:** Performs image preprocessing, detects AprilTags, performs contouring and numbering augmentation. Publishes to processed_image and detected_tag_id

### AprilTag Behavior
**File:** `packages/ex4/src/apriltag_behaviour.py`  
**Description:** Lane-follows, detects redline and stops for varying amount of time based on the last apriltag detected (subscribe to detected_tag_id)

▶ **Launch Command:**
```bash
dts devel run -H ROBOT_NAME -L apriltag-behaviour
```

---

## Part 2: PeDuckstrian Crosswalks🚸

### Crosswalk and PeDuckstrian Detection
**File:** `packages/ex4/src/crosswalk.py`  
**Description:** Detects crosswalks and peDcukstrians. Waits until crosswalk is safe before signaling to the navigation node to continue driving.

### Crosswalk Navigation
**File:** `packages/ex4/src/crosswalk_navigation.py`  
**Description:** A lane following node adapted to receive processed images and navigation commands from the Crosswalk and PeDuckstrian node.

▶ **Launch Command:**
```bash
dts devel run -H ROBOT_NAME -L crosswalk
```

---

## Part 3: Safe Navigation🦺

**File:** `packages/ex4/src/safe_driving.py`  
**Description:** Detects the back of a broken down duckiebot and moves around it

▶ **Launch Command:**
```bash
dts devel run -H ROBOT_NAME -L safe-driving
```

---

## MiscellaneousControl Node

**File:** `packages/ex4/src/misc_control.py`  
**Description:** All of the above tasks uses this node to control the led and set the camera framerate

---
