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


import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class QRCodeReader(Node):
    def __init__(self, node_name='qr_code_reader', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        self.bridge = CvBridge()
        self.camera_image = None
        self.camera_sub = self.create_subscription(Image,
                                                   '/top_camera/rgb/preview/image_raw',
                                                   self.camera_callback,
                                                   10)  # Adjust QoS as needed
        
        self.qr_info_pub = self.create_publisher(String, '/qr_info', 10)

        self.qr_detector = cv2.QRCodeDetector()

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.camera_image = cv_image
            val, _, _ = self.qr_detector.detectAndDecode(cv_image)
            if val != '':
                print(val)
                qr_msg = String()
                qr_msg.data = val
                self.qr_info_pub.publish(qr_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")

def main(args=None):
    rclpy.init(args=args)
    qr_code_reader = QRCodeReader()
    rclpy.spin(qr_code_reader)
    qr_code_reader.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()