#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
import time

import cv2
import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, PointStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
import urllib.request

import math
import tensorflow as tf
import speech_recognition as sr
#import pyttsx3


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)

        self.pose_frame_id = 'map'

        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.rings_found = []

        self.camera_image = None
        self.processing_position = False
        self.parked = False
        self.img_updated = False
        self.current_face = None # Used to store the img of face. With this we check if face is Mona Lisa or normal face
        self.list_of_suggested_rings_1 = set() # Used for storing info of face we 1st visited-->2 colors saved
        self.list_of_suggested_rings_2 = set() # Used for storing info of face we 2nd visited-->2 colors saved
        self.ring_to_visit = None # Used for storing the ring we have to visit-->ring that is in both lists

        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        # self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)

        self.people_marker_sub = self.create_subscription(Marker, 'people_marker',
                                                          self._peopleMarkerCallback,
                                                          QoSReliabilityPolicy.BEST_EFFORT)
        self.camera_sub = self.create_subscription(Image,
                                                   '/top_camera/rgb/preview/image_raw',
                                                   self.camera_callback,
                                                   qos_profile_sensor_data)
                                                   
        
        self.face_img_sub = self.create_subscription(Image, '/detected_face', self.save_face_callback, qos_profile_sensor_data)

        self.ring_sub = self.create_subscription(Marker, "/breadcrumbs", self.breadcrumbs_callback, QoSReliabilityPolicy.BEST_EFFORT)
        self.detected_face_sub = self.create_subscription(Marker, "/detected_face_coord", self.detected_face_callback, QoSReliabilityPolicy.BEST_EFFORT)

        self.qr_detect_sub = self.create_subscription(String, '/qr_info', self.qr_callback, 10)
        self.ring_color_sub = self.create_subscription(String, "/ring_color", self.ring_color_callback, 10)

        self.vel_pub = self.create_publisher(Twist,
                                             '/cmd_vel_nav',
                                             10)
        
        # Arm Publisher
        self.arm_pub = self.create_publisher(String, '/arm_command', 10)

        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.face_pub = self.create_publisher(PointStamped, 'face', qos_profile)

        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.latest_people_marker_pose = None
        self.current_pose = None
        self.hellos_said = 0
        self.rings_detected = 0
        self.bridge = CvBridge()
        self.parking_initiated = False
        self.breadcrumbs_face = None
        self.detected_face_list = []


        # if this is not None, then image of Mona is downloaded
        self.mona_link = None

        #self.audio_engine = pyttsx3.init()

        self.get_logger().info(f"Robot commander has been initialized!")

    def detected_face_callback(self, msg):
    face = []
    face[0] = msg.pose.position.x
    face[1] = msg.pose.position.y
    for detected_face in self.detected_face_list:
        if abs(face[0] - detected_face[0]) < 0.5 and abs(face[1] - detected_face[1]) < 0.5:
            return

    self.detected_face_list.append(face)
    self.latest_people_marker_pose = msg.pose.position
    
    # saving the face detected and the using it for model to make a prediction
    def save_face_callback(self, msg):
        # self.get_logger().info("Saving face image...")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.current_face = cv_image

        #cv2.imshow("Detected Face", cv_image)
        #cv2.waitKey(1)

    def detected_face_callback(self, msg):
        self.latest_people_marker_pose = msg.pose.position

    # Speech recognition function. Saves colors in sets list_of_suggested_rings_1 and list_of_suggested_rings_2 if there are any!
    def recognize_colors(self):
        """recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak the sentence:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            sentence = recognizer.recognize_google(audio)
            print("You said:", sentence)

            colors = []
            for word in sentence.split():
                if word.lower() in ['green', 'red', 'blue', 'yellow', 'black', 'pink', 'orange', 'brown', 'purple']:  
                    colors.append(word.lower())

            print("Recognized colors:", colors)
            if colors:
                if not self.list_of_suggested_rings_1:
                    self.list_of_suggested_rings_1 = set(colors)
                else:
                    self.list_of_suggested_rings_2 = set(colors)    

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))"""
        try:
            sentence = input("Type the sentence: ")
            print("You typed:", sentence)

            colors = []
            for word in sentence.split():
                if word.lower() in ['green', 'red', 'blue', 'yellow', 'black', 'pink', 'orange', 'brown', 'purple']:  
                    colors.append(word.lower())

            print("Recognized colors:", colors)
            if colors:
                if not self.list_of_suggested_rings_1:
                    self.list_of_suggested_rings_1 = set(colors)
                else:
                    self.list_of_suggested_rings_2 = set(colors)    

        except Exception as e:
            print("An error occurred: {0}".format(e))


    # Not that much importan. Used for MODEL to detect Mona Lisa and paintings
    def calculate_reconstruction_error(self, original, reconstructed):
        return np.mean((original - reconstructed) ** 2, axis=-1)

    def breadcrumbs_callback(self, msg):
        print("I AM GETTING CALLED!!!")
        self.breadcrumbs_face = msg.pose.position
        print(f"x {self.breadcrumbs_face.x} y {self.breadcrumbs_face.y}")
        # self.rings_found.append(msg.pose.position, "COLOR")
        self.rings_detected += 1

    def qr_callback(self, msg):
        self.get_logger().info(f"QR code detected: {msg}")
        if "vicos" in msg.data:
            self.mona_link = msg.data
            self.get_logger().info(f"Starting to download Mona Lisa image from {self.mona_link}")
            urllib.request.urlretrieve(self.mona_link, "mona_lisa.jpg")
            time.sleep(1)
   
    def ring_color_callback(self, msg):
        for ring in self.rings_found:
                if ring[1] == "COLOR":
                    ring[1] = msg.data

    def greet_face(self, msg):
        #self.audio_engine.say(msg)
        #self.audio_engine.runAndWait()
        # self.get_logger().info(msg)
        pass

    def camera_callback(self, msg):
        try:
            #this line should convert img to cv-format but it it not working :(
            #Maybe I have to install cv_bridge lib but i cannot cause i am not sudo
            #So i am not sure how this works (cannot check)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.camera_image = cv_image

            qr_detector = cv2.QRCodeDetector()
            val, _, _ = qr_detector.detectAndDecode(cv_image)
            # print(val)

        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
        return
    

    def park(self):
        self.get_logger().info("PARKINGGGGGGGGGGGGGGGG")
        rclpy.spin_once(self)
        while not self.camera_image.any():
            self.debug('Waiting for camera image...')
            time.sleep(0.1)

        try:
            cv_image = self.camera_image
            height, width, _ = cv_image.shape
            size = min(height, width)
            #cv_image = cv_image[:size, :size]

            half_size = size // 2
            top_left = cv_image[:half_size, :half_size]
            top_right = cv_image[:half_size, half_size:]
            bottom_left = cv_image[half_size:, :half_size]
            bottom_right = cv_image[half_size:, half_size:]

            black_pixels_top_left = np.sum(top_left < [5, 5, 5])
            black_pixels_top_right = np.sum(top_right < [5, 5, 5])
            black_pixels_bottom_left = np.sum(bottom_left < [5, 5, 5])
            black_pixels_bottom_right = np.sum(bottom_right < [5, 5, 5])

            #total_pixels = size * size
            #black_percentage_top = (black_pixels_top_left + black_pixels_top_right) / total_pixels
            #black_percentage_bottom = (black_pixels_bottom_left + black_pixels_bottom_right) / total_pixels

            if (black_pixels_bottom_left + black_pixels_bottom_right == 0 or black_pixels_bottom_left + black_pixels_bottom_right < 6000) and black_pixels_top_right + black_pixels_top_left == 0:  # If black is mostly on the bottom
                self.get_logger().info("Reached parking spot. Stopping.")
                self.get_logger().info(f"NUM OF PIXELS: {black_pixels_bottom_right + black_pixels_bottom_left}")
                self.parked = True
                # rclpy.spin_once(self)
                while not self.camera_image.any():
                    self.debug('Waiting for camera image...')
                    time.sleep(0.1)
                return


            velocity_msg_processing = Twist()
            if (black_pixels_bottom_left > 0 and black_pixels_top_left > 0) and black_pixels_bottom_right + black_pixels_top_right == 0:
                self.get_logger().info("PROCESSING WHERE AM I. LEFT SIDE")
                pixel_sum = black_pixels_bottom_left + black_pixels_top_left
                velocity_msg_processing.angular.z = -0.7
                self.vel_pub.publish(velocity_msg_processing)
                self.get_logger().info("TURNED LEFT")
                rclpy.spin_once(self)
                while not self.camera_image.any():
                    self.debug('Waiting for camera image...')
                    time.sleep(0.1)
                time.sleep(1.5)
                #i m not sure if this is OK?
                cv_image_new = self.camera_image
                height, width, _ = cv_image_new.shape
                size = min(height, width)
                #cv_image_new = cv_image_new[:size, :size]

                half_size = size // 2
                top_left_new = cv_image_new[:half_size, :half_size]
                top_right_new = cv_image_new[:half_size, half_size:]
                bottom_left_new = cv_image_new[half_size:, :half_size]
                bottom_right_new = cv_image_new[half_size:, half_size:]

                black_pixels_top_left_new = np.sum(top_left_new < [5, 5, 5])
                black_pixels_top_right_new = np.sum(top_right_new < [5, 5, 5])
                black_pixels_bottom_left_new = np.sum(bottom_left_new < [5, 5, 5])
                black_pixels_bottom_right_new = np.sum(bottom_right_new < [5, 5, 5])

                if pixel_sum +20> black_pixels_bottom_left_new + black_pixels_top_left_new + black_pixels_top_right_new + black_pixels_bottom_right_new:
                    self.get_logger().info("CHANGING DIRECTION. SEEMS 1ST ONE WAS WRONG!")
                    velocity_msg_processing.angular.z = 1.8
                    self.vel_pub.publish(velocity_msg_processing)
                    self.get_logger().info("TURNED RIGHT")
                    self.processing_position = False
                self.processing_position = False

            if (black_pixels_bottom_right > 0 and black_pixels_top_right > 0) and black_pixels_bottom_left + black_pixels_top_left == 0:
                self.get_logger().info("PROCESSING WHERE AM I. RIGHT SIDE")
                pixel_sum = black_pixels_bottom_right + black_pixels_top_right
                velocity_msg_processing.angular.z = 0.7
                self.vel_pub.publish(velocity_msg_processing)
                self.get_logger().info("TURNED RIGHT")
                rclpy.spin_once(self)
                while not self.camera_image.any():
                    self.debug('Waiting for camera image...')
                    time.sleep(0.1)
                time.sleep(1.5)
                #i m not sure if this is OK?
                cv_image_new = self.camera_image
                height, width, _ = cv_image_new.shape
                size = min(height, width)
                #cv_image_new = cv_image_new[:size, :size]

                half_size = size // 2
                top_left_new = cv_image_new[:half_size, :half_size]
                top_right_new = cv_image_new[:half_size, half_size:]
                bottom_left_new = cv_image_new[half_size:, :half_size]
                bottom_right_new = cv_image_new[half_size:, half_size:]

                black_pixels_top_left_new = np.sum(top_left_new < [5, 5, 5])
                black_pixels_top_right_new = np.sum(top_right_new < [5, 5, 5])
                black_pixels_bottom_left_new = np.sum(bottom_left_new < [5, 5, 5])
                black_pixels_bottom_right_new = np.sum(bottom_right_new < [5, 5, 5])

                if pixel_sum +20> black_pixels_bottom_left_new + black_pixels_top_left_new + black_pixels_top_right_new + black_pixels_bottom_right_new:
                    self.get_logger().info("CHANGING DIRECTION. SEEMS 1ST ONE WAS WRONG!")
                    velocity_msg_processing.angular.z = -1.8
                    self.vel_pub.publish(velocity_msg_processing)
                    self.get_logger().info("TURNED LEFT")
                    self.processing_position = False
                self.processing_position = False



            velocity_msg = Twist()

            if black_pixels_bottom_left + black_pixels_bottom_right == 0 and black_pixels_top_left == 0 and black_pixels_top_right > 0:
                velocity_msg.angular.z = -0.4
                velocity_msg.linear.x = 0.4
                self.get_logger().info("BLACK TOP-RIGHT")
                self.vel_pub.publish(velocity_msg)
                self.debug('Parking the robot...')
                time.sleep(0.1)
                time.sleep(1.5)
                rclpy.spin_once(self)
                while not self.camera_image.any():
                    self.debug('Waiting for camera image...')
                    time.sleep(0.1)
                return
            elif black_pixels_bottom_left + black_pixels_bottom_right == 0 and black_pixels_top_right == 0 and black_pixels_top_left > 0:
                velocity_msg.angular.z = 0.4
                velocity_msg.linear.x = 0.4
                self.get_logger().info("BLACK TOP-LEFT")
                self.vel_pub.publish(velocity_msg)
                self.debug('Parking the robot...')
                time.sleep(0.1)
                time.sleep(1.5)
                rclpy.spin_once(self)
                while not self.camera_image.any():
                    self.debug('Waiting for camera image...')
                    time.sleep(0.1)
                return
            #elif black_pixels_top_left + black_pixels_top_right > 0 and black_pixels_bottom_left + black_pixels_bottom_right == 0:  # If black is mostly on top
            #    velocity_msg.linear.x = 0.4
            #    time.sleep(1.5)
            else:
                most_black_square_index = np.argmax([black_pixels_top_left, black_pixels_top_right, black_pixels_bottom_left, black_pixels_bottom_right])
                if most_black_square_index == 0:
                    velocity_msg.angular.z = 0.1
                    velocity_msg.linear.x = 0.05
                    self.get_logger().info("MOVING INSIDE CIRCLE")
                    time.sleep(3.0)
                elif most_black_square_index == 1:
                    velocity_msg.angular.z = -0.1
                    velocity_msg.linear.x = 0.05
                    self.get_logger().info("MOVING INSIDE CIRCLE")
                    time.sleep(3.0)
                elif most_black_square_index == 2:
                    velocity_msg.angular.z = 0.1
                    velocity_msg.linear.x = 0.05
                    self.get_logger().info("MOVING INSIDE CIRCLE")
                    time.sleep(3.0)
                elif most_black_square_index == 3:
                    velocity_msg.angular.z = -0.1
                    velocity_msg.linear.x = 0.05
                    self.get_logger().info("MOVING INSIDE CIRCLE")
                    time.sleep(3.0)
            if not self.parked:
                self.vel_pub.publish(velocity_msg)
                self.debug('Parking the robot...')
                time.sleep(0.1)
                rclpy.spin_once(self)
                return
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")

    def final_check_left(self):
        rclpy.spin_once(self)
        vel_msg = Twist()
        vel_msg.angular.z = 4.5
        self.vel_pub.publish(vel_msg)
        time.sleep(4.0)
        rclpy.spin_once(self)
        self.get_logger().info(f"FINL CHECK LEFT")
        self.park()
        #rclpy.spin_once(self)

    def final_check_right(self):
        rclpy.spin_once(self)
        vel_msg = Twist()
        vel_msg.angular.z = 4.0
        self.vel_pub.publish(vel_msg)
        time.sleep(4.0)
        rclpy.spin_once(self)
        self.get_logger().info(f"FINL CHECK RIGHT")
        self.park()
        #rclpy.spin_once(self)

    def _peopleMarkerCallback(self, msg):
        """Handle new messages from 'people_marker'."""
        self.debug('Received people marker pose')
        # Store the latest pose for use in the movement loop
        # self.latest_people_marker_pose = msg.pose.position


    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        #self.info("What the fuck")
        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        # self.info("What the fuck")
        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        # self.info("What the fuck")
        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return

    def YawToQuaternion(self, angle_z = 0.1):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return

    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
    def check_ring(self, marked_rings, point):
        # self.get_logger().info(f"IM LOOKING FOR RINGS")
        coord_ring_relative_to_r = self.latest_ring_marker_pose
        if coord_ring_relative_to_r is not None:
            coord_ring = PoseStamped()
            coord_ring.header.frame_id = 'map'
            coord_ring.header.stamp = self.get_clock().now().to_msg()

            x = point[0] + coord_ring_relative_to_r.x
            y = point[1] + coord_ring_relative_to_r.y
            z = coord_ring_relative_to_r.z + point[2]
            #self.get_logger().info(f"----------------------------> {z}")

            x1 = coord_ring_relative_to_r.x
            y1 = coord_ring_relative_to_r.y
            z1 = coord_ring_relative_to_r.z
            coord_ring.pose.position.x = x
            coord_ring.pose.position.y = y

            if math.isinf(z):
                z = 0.0

            coord_ring.pose.orientation = self.YawToQuaternion(z)
            
            if len(marked_rings) != 0:
                for ring in marked_rings:
                    if abs(x - ring.pose.position.x) < 1.5 and abs(y - ring.pose.position.y) < 2.5:
                        self.get_logger().info(f"Face already marked")
                        
                        return False, marked_rings
            
            marked_rings.append(coord_ring)
            point_msg = PointStamped()
            point_msg.header.frame_id = 'base_link'
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.point = coord_ring_relative_to_r
            self.latest_ring_marker_pose = None
            self.face_pub.publish(point_msg)

            self.get_logger().info(f"Number of detected rings so far: {len(marked_rings)}")
            for l in range(len(marked_rings)):
                self.get_logger().info(f"ring {l}: x: {marked_rings[l].pose.position.x}, y: {marked_rings[l].pose.position.y}, z: {marked_rings[l].pose.orientation.z}")

            # MOVE BACK TO THE POINT
            goal_pose = PoseStamped()
            goal_pose.pose.position.x = point[0]
            goal_pose.pose.position.y = point[1]
            ## fix maybe
            goal_pose.pose.orientation = self.YawToQuaternion(point[2])

            return True, marked_rings
        else:
            return False, marked_rings

    def check_approach(self, marked_poses, point):
        self.get_logger().info(f"IM LOOKING FOR FACES")
        #Check if there is a new 'people_marker' pose to go to first
        coord_face_relative_to_r = self.latest_people_marker_pose
        # pcl_face = self.pcl_face_pos
        if coord_face_relative_to_r is not None:
            coord_face = PoseStamped()
            coord_face.header.frame_id = 'map'
            coord_face.header.stamp = self.get_clock().now().to_msg()

            x = point[0] + coord_face_relative_to_r.x
            y = point[1] + coord_face_relative_to_r.y
            z = coord_face_relative_to_r.z + point[2]
            #self.get_logger().info(f"----------------------------> {z}")

            x1 = coord_face_relative_to_r.x
            y1 = coord_face_relative_to_r.y
            z1 = coord_face_relative_to_r.z
            coord_face.pose.position.x = x
            coord_face.pose.position.y = y
            coord_face.pose.orientation = self.YawToQuaternion(z)
            
            for face in marked_poses:
                if abs(x - face.pose.position.x) < 1.5 and abs(y - face.pose.position.y) < 2.5:
                    self.get_logger().info(f"Face already marked")
                    
                    return False, marked_poses
            
            marked_poses.append(coord_face)
            

            # MOVE TOWARDS THE FACE
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            
            distance_from_face = 0.1

            if coord_face_relative_to_r.x > 0:
                goal_pose.pose.position.x = coord_face_relative_to_r.x - distance_from_face
            else:
                goal_pose.pose.position.x = coord_face_relative_to_r.x + distance_from_face

            if coord_face_relative_to_r.y > 0:
                goal_pose.pose.position.y = coord_face_relative_to_r.y - distance_from_face
            else:
                goal_pose.pose.position.y = coord_face_relative_to_r.y + distance_from_face

            goal_pose.pose.orientation = self.YawToQuaternion(z1)



            self.get_logger().info(f"going to face on x: {goal_pose.pose.position.x} and y {goal_pose.pose.position.y}")
            self.goToPose(goal_pose)
            while not self.isTaskComplete():
                self.info("Moving towards the face...")
                time.sleep(1)

            self.latest_people_marker_pose = None
            spin_dist = 0.2 * math.pi
            n = 0
            approached = False
            while n < 10:
                self.spin(spin_dist)
                n += 1
                while not self.isTaskComplete():
                   
                    if(self.latest_people_marker_pose is not None):
                        time.sleep(3)
                        self.greet_face("Hi there")
                        #self.get_logger().info(f"{self.hellos_said}")
                        #self.hellos_said += 1
                        #self.get_logger().info(f"Hello there!")

                        # NOTE: when we aproach the face we should say colors of rings. I think this should be done very faast but we can change that easy.
                        #       For now leave it as it is. We can change it later.
                        self.recognize_colors()
                        time.sleep(5)
                        n = 10
                        break
                    # self.check_approach(marked_poses, rc.current_pose)
                    time.sleep(1)
            time.sleep(1)

            # MOVE BACK TO THE POINT
            goal_pose.pose.position.x = point[0]
            goal_pose.pose.position.y = point[1]
            ## fix maybe
            goal_pose.pose.orientation = self.YawToQuaternion(point[2])

            self.goToPose(goal_pose)
            while not self.isTaskComplete():
                self.info("Moving back to the point...")
                time.sleep(1)
            time.sleep(1)

            #goal_pose.pose.orientation = self.YawToQuaternion(0.5)
            # Reset the latest people marker pose to ensure it's only used once
            self.latest_people_marker_pose = None
            return True, marked_poses
        else:
            return False, marked_poses

def main(args=None):

    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()

    # Finally send it a goal to reach
    #               1                2                  3                   4                 5                  6                7                   8                  9                 10                11              12               13                  14
    points = [[-0.9, -0.4, 0,00],[-1.6, 1.22, 0.0],[-1.41, 4.36,-0.165],[-1.35, 3.17,-0.568],[1.9,3.04,0.57],[2.48,1.81,0.00247],[0.39,1.87,-0.207],[1.34,0.308,0.0582],[2.23,-1.78,-1],[3.27,-1.4,0.961],[1.14,-1.8,-1.0],[-0.16,-1.33,0.832]]
    # ,[-0.7, 1.42,-0.584] 3
    # [-0.464, 0.18, 0,00]
    # [1.5,-0.4,-0.069] --> 10
    # [2.23,-1.78,-1] --> 11
    # [0.63,-0.76,0.458],[1.5,-0.4,-0.069]   9 and 10 possitions!!!


    arm_msg = String()
    arm_msg.data = "look_for_qr"
    rc.arm_pub.publish(arm_msg)
    time.sleep(2)
    marked_rings = []
    marked_poses = []
    model = tf.keras.models.load_model('/home/kappa/task2_robot/src/dis_tutorial3/scripts/anomaly_detection_model.h5')
    approached_face = None
    i = 0
    while len(points) > i:
        try:
            # NOTE: This part has to change. We should run it only after we visit every point.
            #       Afer we visit every point then we park.
            #       Then we check for cylinder.
            #       Then we go to real Mona Lisa.
            if rc.list_of_suggested_rings_1 and rc.list_of_suggested_rings_2:
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = 'map'
                # goal_pose.header.stamp = self.get_clock().now().to_msg()
                goal_pose.pose.position.x = 2.45
                goal_pose.pose.position.y = -1.6
                # goal_pose.pose.position.z = 0.0
                goal_pose.pose.orientation = rc.YawToQuaternion(-1.0)
                rc.goToPose(goal_pose)

                while not rc.isTaskComplete():
                    rc.info("Moving to the green ring.")
                    time.sleep(1)


                rclpy.spin_once(rc)
                rc.info("Starting to Park outside the while loop")
                while not rc.parked:
                    rc.info("Starting to Park")
                    rc.park()
                    if rc.parked:
                        break
                    rclpy.spin_once(rc)

                rc.parked = False
                rc.final_check_left()
                while not rc.isTaskComplete():
                    rc.info("Waiting for the task to complete...")
                    rclpy.spin_once(rc)
                    time.sleep(1)
                time.sleep(4.0)    
                rclpy.spin_once(rc)
                #rc.park()
                rc.parked = False
                #rclpy.spin_once(rc)


                # This is still not checked if it works. Old code is below it!
                for i in range(10):
                    rc.final_check_left()
                    while not rc.isTaskComplete():
                        rc.info("Waiting for the task to complete...")
                        rclpy.spin_once(rc)
                        time.sleep(1)
                    time.sleep(2.0)
                    #rclpy.spin_once(rc)
                    rc.parked = False    
                # This should break out of whole loop. 
                # After this we should look around to detect cylinder. 
                # After that publish to arm_controller and move arm to read QR Code on top of cylinder.
                # With taht we teach model to detect real Mona Lisa. 
                # Use that model and go around map and find it!
                break

            point = points[i]
            # If no new 'people_marker' pose, proceed with the next point in the list
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = rc.get_clock().now().to_msg()
            goal_pose.pose.position.x = point[0]
            goal_pose.pose.position.y = point[1]
            goal_pose.pose.orientation = rc.YawToQuaternion(point[2])

            rc.goToPose(goal_pose)

            while not rc.isTaskComplete():
                time.sleep(1)


            # Here I put both ring detection and face approach in the same while loop.
            # This way we only have to do everything onece.
            # Also removed stoping logic and destroying the node since we are not stoping the robot at all.
            rc.latest_ring_marker_pose = None
            rc.latest_people_marker_pose = None
            spin_dist = 0.5 * math.pi
            n = 0
            while n < 4:
                rc.spin(spin_dist)
                n+=1
                while not rc.isTaskComplete():

             
                    
                    rc.get_logger().info(f"curr pose x: {rc.current_pose.pose.position.x} y: {rc.current_pose.pose.position.y} z: {rc.current_pose.pose.orientation.z}")
                    approached_ring, marked_rings = rc.check_ring(marked_rings, point)

                    # Loading model and checking if we need to approach the face
                    error_map = None
                    mean_error = None
                    threshold = 0.1
                    img = None
                    # Here, we transform the face image to the same size as the model was trained on
                    if rc.current_face is not None and rc.current_face.size > 0:
                        print("Calling the check approach function")
                        approached_face, marked_poses = rc.check_approach(marked_poses, point)
                        cv2.imshow("Detected Face", rc.current_face)

                    if approached_ring or approached_face:
                        n = 0
                    # rc.check_approach(marked_poses, rc.current_pose)
                    time.sleep(1)
            i+=1
        except IndexError:
            print(f"Error: Attempted to access index {i} in points list, which has {len(points)} elements.")
            break

    rc.destroyNode()
    # And a simple example
if __name__=="__main__":
    main()
