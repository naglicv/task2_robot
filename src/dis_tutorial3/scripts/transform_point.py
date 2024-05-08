#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class TranformPoints(Node):
    """Demonstrating some convertions and loading the map as an image"""
    def __init__(self):
        super().__init__('map_goals')

        self.face_point_in_robot_frame = None
        self.ring_point_in_robot_frame = None
        self.cylinder_point_in_robot_frame = None
        self.parking_space_point_in_robot_frame = None
        # Basic ROS stuff
        timer_frequency = 1
        timer_period = 1/timer_frequency

        # Functionality variables
        self.marker_id = 0

        # For listening and loading the 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # For publishing the markers
        self.marker_pub = self.create_publisher(Marker, "/breadcrumbs", QoSReliabilityPolicy.BEST_EFFORT)
        #self.face_sub = self.create_subscription(PointStamped, "/face", self.face_callback, qos_profile)
        self.ring_sub = self.create_subscription(PointStamped, "/ring_robot", self.ring_callback, qos_profile)
        #self.cylinder_sub = self.create_subscription(PointStamped, "/cylinder", self.cylinder_callback, qos_profile)
        #self.cylinder_sub = self.create_subscription(PointStamped, "/detected_cylinder", self.cylinder_callback, qos_profile)
        #self.parking_space_sub = self.create_subscription(PointStamped, "/parking_space", self.parking_space_callback, qos_profile)

        # Create a timer, to do the main work.
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def face_callback(self, msg):
        self.get_logger().info(f"Received face point: {msg.point.x}, {msg.point.y}, {msg.point.z}")
        self.face_point_in_robot_frame = msg

    def ring_callback(self, msg):
        self.get_logger().info(f"Received ring point: {msg.point.x}, {msg.point.y}, {msg.point.z}")
        self.ring_point_in_robot_frame = msg

    def cylinder_callback(self, msg):
        self.get_logger().info(f"Received cylinder point: {msg.point.x}, {msg.point.y}, {msg.point.z}")
        self.cylinder_point_in_robot_frame = msg

    def parking_space_callback(self, msg):
        self.get_logger().info(f"Received parking space point: {msg.point.x}, {msg.point.y}, {msg.point.z}")
        self.parking_space_point_in_robot_frame = msg

    def timer_callback(self):
        points_to_transform = [
            #(self.face_point_in_robot_frame, "face", [1.0, 0.0, 0.0]), # Red for face
            (self.ring_point_in_robot_frame, "ring", [0.0, 1.0, 0.0]), # Green for ring
            #(self.cylinder_point_in_robot_frame, "cylinder", [0.0, 0.0, 1.0]), # Blue for cylinder
            #(self.parking_space_point_in_robot_frame, "parking_space", [1.0, 1.0, 0.0]) # Yellow for parking space
        ]

        #for point, label, color in points_to_transform:
        color = [0.0, 1.0, 0.0]
        label = "ring"
        point = self.ring_point_in_robot_frame
        if point is not None:
            self.get_logger().info(f"--> Transforming {label}")

            time_now = rclpy.time.Time()
            timeout = Duration(seconds=0.1)
            try:
                camera_frame = point.header.frame_id
                map_frame = "map" 
                transform = self.tf_buffer.lookup_transform(map_frame, camera_frame, time_now, timeout)
                self.get_logger().info(f"Looks like the transform is available for {label}.")

                point_in_map_frame = tfg.do_transform_point(point, transform)
                self.get_logger().info(f"We transformed a PointStamped for {label}!")

                marker_in_map_frame = self.create_marker(point_in_map_frame, self.marker_id, label, color)

                self.marker_pub.publish(marker_in_map_frame)
                self.get_logger().info(f"The marker for {label} has been published to /breadcrumbs. You are able to visualize it in RViz")

                self.marker_id += 1
                if label == "face":
                    self.face_point_in_robot_frame = None
                elif label == "ring":
                    self.ring_point_in_robot_frame = None
                elif label == "cylinder":
                    self.cylinder_point_in_robot_frame = None
                elif label == "parking_space":
                    self.parking_space_point_in_robot_frame = None

            except TransformException as te:
                self.get_logger().info(f"Could not get the transform for {label}: {te}")

    def create_marker(self, point_stamped, marker_id, label, color):
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.id = marker_id

        # Set the scale of the marker
        scale = 0.3
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y
        marker.pose.position.z = point_stamped.point.z

        return marker

def main():
    rclpy.init(args=None)
    node = TranformPoints()
    node.get_logger().info("The node has been initialized.")
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()