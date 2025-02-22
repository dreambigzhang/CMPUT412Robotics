#!/usr/bin/env python3

# potentially useful for question - 1.1 - 1.4 and 2.1

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from duckietown.dtros import DTROS, NodeType
# from srv import SetLed

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Load camera intrinsic parameters from the .yaml file
        self.camera_matrix = np.array([
            [317.4313105701715, 0.0, 276.84824596067926],
            [0.0, 315.7945432243382, 217.9579475772363],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([-0.31269757464446846, 0.07532824868914906, 0.0014825740242535802, 0.0028896360103573197, 0.0])
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribe to the distorted image topic
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.callback, queue_size=1)
        
        # Publisher for undistorted images
        self.undistorted_pub = rospy.Publisher('/camera/image_undistorted', Image, queue_size=1)
        
        # Set frame rate to 3-5 frames per second
        self.frame_rate = 5  # Process 5 frames per second
        self.last_time = rospy.get_time()

        self.srv_leds = rospy.ServiceProxy('set_led', SetLed)


    def undistort_image(self, image):
        """
        Undistort the image using camera calibration parameters.
        """
        h, w = image.shape[:2]
        new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return undistorted

    def preprocess_image(self, image):
        """
        Resize and blur the image for better processing.
        """
        resized_image = cv2.resize(image, (320, 240))  # Resize to smaller resolution
        blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # Apply Gaussian blur
        return blurred_image

    def detect_lane_color(self, image):
        """
        Detect lane colors (blue, red, green) using HSV thresholds.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for each color
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(red_mask, green_mask))
        return combined_mask

    def detect_lane(self, image):
        """
        Detect lanes using contour detection and ROI.
        """
        # Apply ROI
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked_image = cv2.bitwise_and(image, mask)
        
        # Find contours
        contours, _ = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding rectangles around detected lanes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return image
    
    def set_led(self, color):
        """Publishes LED color changes to LEDNode."""
        self.srv_leds(color)
        rospy.loginfo(f"Requested LED color change to {color}")

    def callback(self, msg):
        """
        Callback function for processing camera images.
        """
        current_time = rospy.get_time()
        if current_time - self.last_time < 1.0 / self.frame_rate:
            return  # Skip processing to maintain frame rate
        self.last_time = current_time
        
        # Convert compressed image to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Undistort the image
        undistorted_image = self.undistort_image(cv_image)
        
        # Publish the undistorted image
        undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_image, encoding="bgr8")
        self.undistorted_pub.publish(undistorted_msg)
        
        # Preprocess image
        preprocessed_image = self.preprocess_image(undistorted_image)
        
        # # Detect lanes
        # lane_image = self.detect_lane(preprocessed_image)
        
        # # Publish lane detection results
        # lane_msg = self.bridge.cv2_to_imgmsg(lane_image, encoding="bgr8")
        # self.lane_pub.publish(lane_msg)
        
        # # Detect lanes and colors
        # color_mask = self.detect_lane_color(preprocessed_image)
        
        # # Control LEDs based on detected colors
        # if np.any(color_mask == 255):  # If any color is detected
        #     self.set_led("blue")  # Example: Turn on blue LED

if __name__ == '__main__':
    # Initialize the node
    node = LaneDetectionNode(node_name='lane_detection_node')
    rospy.spin()