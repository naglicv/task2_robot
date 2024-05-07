#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        # self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

        #self.segmented_cloud_sub = self.create_subscription(PointCloud2, "/segmented_cloud", self.segmented_cloud_callback, 1)

        # Publiser for the visualization markers
        self.ring_pub = self.create_publisher(Marker, "/breadcrumbs", QoSReliabilityPolicy.BEST_EFFORT)
        self.ring_pub_point = self.create_publisher(PointStamped, "/ring", qos_profile)
        

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

        self.depth_info = None        

    def image_callback(self, data):
        # self.get_logger().info(f"I got a new image! Will try to find rings...")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        blue = cv_image[:,:,0]
        green = cv_image[:,:,1]
        red = cv_image[:,:,2]

        # Tranform image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # gray = red

        # Apply Gaussian Blur
        gray = cv2.GaussianBlur(gray,(3,3),0)

        # Do histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)
        cv2.imshow("Binary Image", thresh)
        cv2.waitKey(1)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        # cv2.drawContours(gray, contours, -1, (255, 0, 0), 3)
        # cv2.imshow("Detected contours", gray)
        # cv2.waitKey(1)

        # Fit ellipses to contours and check aspect ratio, circularity, and radius consistency
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)
            

        # Find two elipses with same centers
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                depth_info_local = self.depth_info
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                angle_diff = np.abs(e1[2] - e2[2])

                # cv2.ellipse(cv_image, e1, (0, 0, 255), 2)
                # cv2.imshow("All fitted ellipeses",cv_image)    

                # this check seems weird but it works for distinguising rings on the ground vs high up (the ones we want)
                if e1[0][1] > 80 or e2[0][1] > 80:
                    continue

                # cv2.ellipse(cv_image, e1, (0, 0, 255), 2)
                # cv2.imshow("All higher ellipses",cv_image)    

                # The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
                if dist >= 25:
                    continue

                # cv2.ellipse(cv_image, e1, (0, 0, 255), 2)
                # cv2.imshow("All higher with small distance ellipses",cv_image)

                # cv2.ellipse(cv_image, e1, (0, 0, 255), 2)
                # cv2.imshow("Non Depth verified rings",cv_image)

                # The rotation of the elipses should be whitin 4 degrees of eachother
                if angle_diff>15:
                    continue

                # cv2.ellipse(cv_image, e1, (0, 0, 255), 2)
                # cv2.imshow("All lower diff ells",cv_image)

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring
                
                
                
                # Verify detection with depth sensors to avoid cubes, 2d rings on the wall
                if self.detect_depth_ring(depth_info_local, e1[0][0], e1[0][1], cv_image) is not True:
                    continue
              
                color = self.detect_ring_color(cv_image, (e1[0][0], e1[0][1]), e1[1][0], e1[1][1])
                self.get_logger().info(f"e1 color {color}")
                color = self.detect_ring_color(cv_image, (e2[0][0], e2[0][1]), e2[1][0], e2[1][1])
                self.get_logger().info(f"e2 color {color}")


                ## --------------------------------------------------------------
                ## At this point we have a correct ring detected and the color of it. Need to mark this on the map.

                    
                candidates.append((e1,e2))

        # print("Processing is done! found", len(candidates), "candidates for rings")

        # Plot the rings on the image
        for c in candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

            # Assuming e1 and e2 are the ellipses you want to publish
            center_e1 = e1[0]
            center_e2 = e2[0]

            # Calculate the average center of e1 and e2
            average_center = ((center_e1[0] + center_e2[0]) / 2, (center_e1[1] + center_e2[1]) / 2)

            # Create a new PointStamped message
            point = PointStamped()

            # Set the frame ID to the frame of the image
            point.header.frame_id = "/base_link"

            # Set the point to the average center of the detected rings
            point.point.x = center[0]
            point.point.y = center[1]
            point.point.z = 0.0  # Assuming the ring is in the plane of the image

            # Publish the point
            self.ring_pub_point.publish(point)


            marker = Marker()

            marker.header.frame_id = "/map"
            marker.header.stamp = data.header.stamp

            marker.type = 2
            marker.id = 0

            # Set the scale of the marker
            scale = 0.1
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Set the color
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Set the pose of the marker
            marker.pose.position.x = average_center[0]
            marker.pose.position.y = average_center[1]
            marker.pose.position.z = 0.0
            self.ring_pub.publish(marker)

        if len(candidates)>0:
                cv2.imshow("Detected rings",cv_image)
                cv2.waitKey(1)


    def detect_depth_ring(self, depth_info, center_x, center_y, cv_image):
        #np.savetxt("depth_info", depth_info, fmt='%f', delimiter=', ')
        min_radius = 1
        max_radius = 300
        min_circularity = 0.6
        max_hole_area_ratio = 0.5

        # Thresholding: Find pixels within the specified depth range
        mask = np.logical_and(depth_info >= min_radius, depth_info <= max_radius)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

         
        # Loop through each contour
        for contour in contours:
            # Fit a circle to the contour
            (x,y), radius = cv2.minEnclosingCircle(contour)

            if radius > 30:
                continue

            # Check if the center of the circle is within a certain distance from the specified center
            if abs(abs(x) - abs(center_x)) >= 20.0 or abs(abs(y) - abs(center_y)) >= 20.0:
                continue

            # Draw the circle on the image for visualization
            # cv2.circle(cv_image, (int(x), int(y)), int(radius), (0, 0, 255))
            # # cv2.circle(cv_image, (int(x),int(y)), int(radius), (0, 0, 255))
            # cv2.imshow("Depth Detected rings",cv_image)

            area = cv2.contourArea(contour)

            if area <= 4.0:
                continue

            depth_center = depth_info[int(center_y), int(center_x)]

            if depth_center < 0.3:
                # Ring detected
                return True
            else:
                # No ring detected
                continue


        return False  # No ring detected


    def detect_ring_color(self, cv_image, center, x_axis_length, y_axis_length):
        x_center, y_center = center
        half_x_length = x_axis_length // 2
        half_y_length = y_axis_length // 2
        x_min = int(max(0, x_center - half_x_length))
        x_max = int(min(cv_image.shape[1], x_center + half_x_length))
        y_min = int(max(0, y_center - half_y_length))
        y_max = int(min(cv_image.shape[0], y_center + half_y_length))

        # Extract the region of interest (ROI) from the image
        roi = cv_image[y_min:y_max, x_min:x_max]
        # cv2.ellipse(cv_image, e1, (0, 0, 255), 2)
        cv2.imshow("cropped image",roi)

        gray_threshold = 1
        gray_color = [178, 178, 178]
        roi_filtered = roi[~np.all(np.abs(roi - gray_color) <= gray_threshold, axis=-1)]

       
        # Convert the filtered ROI to HSV color space
        # Ensure that roi_filtered has 3 channels (RGB/BGR image)
        if len(roi_filtered.shape) == 2:
            roi_filtered = cv2.cvtColor(roi_filtered, cv2.COLOR_GRAY2BGR)

        # Convert the filtered ROI to HSV color space
        hsv_roi = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2HSV)


        # Count the number of pixels for each color
        blue_pixels = np.sum(hsv_roi[:,:,0] > 170)  # Assuming blue color has high blue channel intensity
        green_pixels = np.sum(hsv_roi[:,:,1] > 170)  # Assuming green color has high green channel intensity
        red_pixels = np.sum(hsv_roi[:,:,2] > 185)  # Assuming red color has high red channel intensity

        self.get_logger().info(f"blue px {blue_pixels}")
        self.get_logger().info(f"green px {green_pixels}")
        self.get_logger().info(f"red px  {red_pixels}")

        if red_pixels > 100:
            return "red"
        elif red_pixels != 0:
            return "green"
        
        return "blue"

        # Determine the color with the most pixels
        max_pixels_color = max(blue_pixels, green_pixels, red_pixels)

        # # Return the color with the most pixels
        # if max_pixels_color == blue_pixels:
        #     return "blue"
        # elif max_pixels_color == green_pixels:
        #     return "green"
        # else:
        #     return "red"


    def depth_callback(self,data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        depth_image[depth_image == np.inf] = 0

        # Convert depth image to visualization format
        image_1 = depth_image / 65536.0 * 255
        max_val = np.max(image_1)
        if max_val > 0: # Check to avoid division by zero
            image_1 = image_1 / max_val * 255
        else:
            image_1 = np.zeros_like(image_1) # Or handle this case as needed

        # Clip values to the range of np.uint8 and cast
        image_viz = np.clip(image_1, 0, 255).astype(np.uint8)

        self.depth_info = depth_image

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)




def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
