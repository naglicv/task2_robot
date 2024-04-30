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

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, PointStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import math
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
        self.camera_image = None

        self.bridge = CvBridge()


        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)

        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)

        self.people_marker_sub = self.create_subscription(Marker,
                                                          'people_marker',
                                                          self._peopleMarkerCallback,
                                                          QoSReliabilityPolicy.BEST_EFFORT)

        #for now best coords are [0.,0.9,1.3,1.0] for look_for_parking? (better possible?)
        self.camera_sub = self.create_subscription(Image,
                                                   '/top_camera/rgb/preview/img_raw',
                                                   self.camera_callback,
                                                   qos_profile_sensor_data)

        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose',
                                                      10)
        self.face_pub = self.create_publisher(PointStamped,
                                                      'face',
                                                      qos_profile)

        self.vel_pub = self.create_publisher(Twist,
                                             'cmd_vel', 
                                             10)

        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.latest_people_marker_pose = None
        self.current_pose = None
        self.hellos_said = 0
        self.rings_detected = 0

        #self.audio_engine = pyttsx3.init()

        self.get_logger().info(f"Robot commander has been initialized!")

    def greet_face(self, msg):
        #self.audio_engine.say(msg)
        #self.audio_engine.runAndWait()
        # self.get_logger().info(msg)
        pass

    def _peopleMarkerCallback(self, msg):
        """Handle new messages from 'people_marker'."""
        self.debug('Received people marker pose')
        # Store the latest pose for use in the movement loop
        self.latest_people_marker_pose = msg.pose.position


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

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

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

    def YawToQuaternion(self, angle_z = 0.):
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

    """def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            height, width, _ = cv.image.shape
            size = min(height, width)
            cv_image = cv_image[:size, :size]

            half_size = size // 2
            top_left = cv_image[:half_size, :half_size]
            top_right = cv_image[:half_size, half_size:]
            bottom_left = cv_image[half_size:, :half_size]
            bottom_right = cv_image[half_size:, half_size:]

            # Calculate the amount of black color in each square
            black_pixels_top_left = np.sum(top_left < [5, 5, 5])  
            black_pixels_top_right = np.sum(top_right < [5, 5, 5])
            black_pixels_bottom_left = np.sum(bottom_left < [5, 5, 5])
            black_pixels_bottom_right = np.sum(bottom_right < [5, 5, 5])

            # Determine which square has the most black color
            black_counts = [black_pixels_top_left, black_pixels_top_right, black_pixels_bottom_left, black_pixels_bottom_right]
            most_black_square_index = np.argmax(black_counts)

            velocity_msg = Twist()
            if most_black_square_index == 0:  # Top Left
                # Turn left and then move forward
                velocity_msg.angular.z = 0.3  # Angular speed (turn left)
                velocity_msg.linear.x = 0.1  # Forward speed
            elif most_black_square_index == 1:  # Top Right
                # Turn right and then move forward
                velocity_msg.angular.z = -0.3  # Angular speed (turn right)
                velocity_msg.linear.x = 0.1  # Forward speed
            elif most_black_square_index == 2:  # Bottom Left
                # Turn left and then move forward
                velocity_msg.angular.z = 0.3  # Angular speed (turn left)
                velocity_msg.linear.x = 0.1  # Forward speed
            elif most_black_square_index == 3:  # Bottom Right
                # Turn right and then move forward
                velocity_msg.angular.z = -0.3  # Angular speed (turn right)
                velocity_msg.linear.x = 0.1  # Forward speed


            # Publish the velocity command
            self.velocity_publisher.publish(velocity_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")"""


    def camera_callback(self, msg):
        try:
            #this line should convert img to cv-format but it it not working :(
            #Maybe I have to install cv_bridge lib but i cannot cause i am not sudo 
            #So i am not sure how this works (cannot check)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") 
            self.camera_image = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            
    def park(self):
        while not self.camera_image:
            self.debug('Waiting for camera image...')
            time.sleep(0.1) 
        #Code from now does next:
        #   Takes img and divides it in 4 equal parts.
        #   Looks which part contains most black color (since parking spot is circle with black edge)
        #   In this code I am considering that the robot is in front of box so in every iteration he moves only forward and a little to the side based on logic implemented
        #   It publishes to the topic cmd_vel and moves robot! (not sure if it works since I cannot check)
        #For example:
        #   If most black is on top left corner, robot moves a little forward and little to the right (oposite of LEFT)
        #TODO:
        #   Put this code in while loop and make some logic for it to stop at some condition (if there is no black in img???)
        #   To do this we should check how camera is positioned and try to figure out how the img loos like from camera view

        try:
            #we take img 
            cv_image = self.camera_image
            #divide it in 4 equal parts
            height, width, _ = cv.image.shape
            size = min(height, width)
            cv_image = cv_image[:size, :size]

            half_size = size // 2
            top_left = cv_image[:half_size, :half_size]
            top_right = cv_image[:half_size, half_size:]
            bottom_left = cv_image[half_size:, :half_size]
            bottom_right = cv_image[half_size:, half_size:]

            #calculate the amount of black color in each square
            black_pixels_top_left = np.sum(top_left < [5, 5, 5])  
            black_pixels_top_right = np.sum(top_right < [5, 5, 5])
            black_pixels_bottom_left = np.sum(bottom_left < [5, 5, 5])
            black_pixels_bottom_right = np.sum(bottom_right < [5, 5, 5])

            #determine which square has the most black color
            black_counts = [black_pixels_top_left, black_pixels_top_right, black_pixels_bottom_left, black_pixels_bottom_right]
            most_black_square_index = np.argmax(black_counts)

            velocity_msg = Twist()
            if most_black_square_index == 0:  #top Left
                #turn left and then move forward
                velocity_msg.angular.z = 0.3  #angular speed (turn left)
                velocity_msg.linear.x = 0.1  #forward speed
            elif most_black_square_index == 1:  #top Right
                #turn right and then move forward
                velocity_msg.angular.z = -0.3  #angular speed (turn right)
                velocity_msg.linear.x = 0.1  #forward speed
            elif most_black_square_index == 2:  #bottom Left
                #turn left and then move forward
                velocity_msg.angular.z = 0.3  #angular speed (turn left)
                velocity_msg.linear.x = 0.1  #forward speed
            elif most_black_square_index == 3:  #bottom Right
                #turn right and then move forward
                velocity_msg.angular.z = -0.3  #angular speed (turn right)
                velocity_msg.linear.x = 0.1  #aorward speed

            #publish the velocity command
            self.velocity_publisher.publish(velocity_msg)
            self.debug('Parking the robot...')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")"""
           

    def check_approach(self, marked_poses, point):
        self.get_logger().info(f"IM LOOKING FOR FACES")
        #Check if there is a new 'people_marker' pose to go to first
        coord_face_relative_to_r = self.latest_people_marker_pose
        if coord_face_relative_to_r is not None:
            coord_face = PoseStamped()
            coord_face.header.frame_id = 'map'
            coord_face.header.stamp = self.get_clock().now().to_msg()

            x = point[0] + coord_face_relative_to_r.x
            y = point[1] + coord_face_relative_to_r.y
            z = coord_face_relative_to_r.z + point[2]
            self.get_logger().info(f"----------------------------> {z}")

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
            point_msg = PointStamped()
            point_msg.header.frame_id = 'base_link'
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.point = coord_face_relative_to_r

            self.face_pub.publish(point_msg)

            self.get_logger().info(f"Number of detected faces so far: {len(marked_poses)}")
            for l in range(len(marked_poses)):
                self.get_logger().info(f"face {l}: x: {marked_poses[l].pose.position.x}, y: {marked_poses[l].pose.position.y}, z: {marked_poses[l].pose.orientation.z}")

            # MOVE TOWARDS THE FACE
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()

            goal_pose.pose.orientation = self.YawToQuaternion(z1)

            while abs(x1) >= 1.0 or abs(y1) >= 1.0:
                curr_pose = self.current_pose.pose.position
                if x1 < 0.3:
                    goal_pose.pose.position.x = curr_pose.x
                    goal_pose.pose.position.y = curr_pose.y + y1/2
                elif y1 < 0.3:
                    goal_pose.pose.position.x = curr_pose.x + x1/2
                    goal_pose.pose.position.y = curr_pose.y
                else:  
                    goal_pose.pose.position.x = curr_pose.x + x1/2
                    goal_pose.pose.position.y = curr_pose.y + y1/2
                x1 = goal_pose.pose.position.x
                y1 = goal_pose.pose.position.y
                goal_pose.pose.orientation = self.YawToQuaternion(math.atan2(y1, x1))
                self.goToPose(goal_pose)
                while not self.isTaskComplete():
                    self.info("Moving towards the face...")
                    time.sleep(1)

            
            # while not self.isTaskComplete():
            #     self.info("Moving towards the face...")
            #     time.sleep(1)

            self.latest_people_marker_pose = None
            spin_dist = 0.2 * math.pi
            n = 0
            approached = False
            while n < 10:
                self.spin(spin_dist)
                n += 1
                while not self.isTaskComplete():
                    self.info("Waiting for the task to complete...")
                    if(self.latest_people_marker_pose is not None):
                        time.sleep(3)
                        self.greet_face("Hi there")
                        self.get_logger().info(f"{self.hellos_said}")
                        self.hellos_said += 1
                        self.get_logger().info(f"Hello there!")
                        time.sleep(3)
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
    ############points = [[-0.9, -0.4, 0,00],[-1.6, 1.22, 0.0],[-1.41, 4.36,-0.165],[-1.35, 3.17,-0.568],[1.9,3.04,0.57],[2.48,1.81,0.00247],[0.39,1.87,-0.207],[1.34,0.308,0.0582],[2.23,-1.78,-1],[3.27,-1.4,0.961],[1.14,-1.8,-1.0],[-0.16,-1.33,0.832]]
    # ,[-0.7, 1.42,-0.584] 3
    # [-0.464, 0.18, 0,00]
    # [1.5,-0.4,-0.069] --> 10
    # [2.23,-1.78,-1] --> 11
    # [0.63,-0.76,0.458],[1.5,-0.4,-0.069]   9 and 10 possitions!!!

    #
    point = [-1.07,1.27,0.481]
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = rc.get_clock().now().to_msg()
    goal_pose.pose.position.x = point[0]
    goal_pose.pose.position.y = point[1]
    goal_pose.pose.orientation = rc.YawToQuaternion(point[2])
    rc.goToPose(goal_pose)


    # TASK 1                    RETURN TO NORMAL WHEN FINISHED
    # marked_poses = []
    # i = 0
    # while len(points) > i or rc.hellos_said <= 3:
    #     point = points[i]
    #     # If no new 'people_marker' pose, proceed with the next point in the list
    #     goal_pose = PoseStamped()
    #     goal_pose.header.frame_id = 'map'
    #     goal_pose.header.stamp = rc.get_clock().now().to_msg()
    #     goal_pose.pose.position.x = point[0]
    #     goal_pose.pose.position.y = point[1]
    #     goal_pose.pose.orientation = rc.YawToQuaternion(point[2])

    #     rc.goToPose(goal_pose)

    #     while not rc.isTaskComplete():
    #         rc.info("Waiting for the task to complete...")
    #         time.sleep(1)

    #     rc.latest_people_marker_pose = None
    #     spin_dist = 0.5 * math.pi
    #     n = 0
    #     while n < 4:
    #         rc.spin(spin_dist)
    #         n+=1
    #         while not rc.isTaskComplete():
    #             rc.info("Waiting for the task to complete...")
    #             rc.get_logger().info(f"curr pose x: {rc.current_pose.pose.position.x} y: {rc.current_pose.pose.position.y} z: {rc.current_pose.pose.orientation.z}")
    #             approached_face, marked_poses = rc.check_approach(marked_poses, point)
    #             if(rc.hellos_said >= 3):
    #                 time.sleep(2)
    #                 rc.info("I have greeted 3 people, I am done!")
    #                 rc.greet_face("I am done with this shit")
    #                 rc.destroyNode()
    #                 break
    #             if approached_face:
    #                 n = 0
    #             # rc.check_approach(marked_poses, rc.current_pose)
    #             time.sleep(1)
    #     i+=1
    #                          END OF RETURN TO NORMAL WHEN FINISHED
    rc.destroyNode()
    # And a simple example
if __name__=="__main__":
    main()
