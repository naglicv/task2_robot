#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import tensorflow as tf

from ultralytics import YOLO

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult
import matplotlib.pyplot as plt

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		marker_topic = "/people_marker"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
		self.breadcrumbs_sub = self.create_subscription(Marker, "/breadcrumbs", self.breadcrumbs_callback, QoSReliabilityPolicy.BEST_EFFORT)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.face_img_pub = self.create_publisher(Image, "/detected_face", QoSReliabilityPolicy.BEST_EFFORT)
		self.detected_face_pub = self.create_publisher(Marker, "/detected_face_coord", QoSReliabilityPolicy.BEST_EFFORT)
		self.face_pub = self.create_publisher(PointStamped, 'face', qos_profile)

		self.model = YOLO("yolov8n.pt")

		self.face = None
		self.marked_faces = []

		self.model_mona_lisa = tf.keras.models.load_model('/home/kappa/Desktop/task2_robot-anomaly_detection/src/dis_tutorial3/scripts/anomaly.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError})
		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def breadcrumbs_callback(self, msg):
		self.detected_face_pub.publish(msg)

	def calculate_reconstruction_error(self, original, reconstructed):
		return np.mean((original - reconstructed) ** 2, axis=-1)

	def load_and_preprocess_image(self, image):
		if isinstance(image, str):
			img = cv2.imread(image)
		else:
			img = image
		img = cv2.resize(img, (224, 224))
		img = img.astype("float32") / 255.0
		return img


	def load_images(self, image_paths):
		images = [self.load_and_preprocess_image(path) for path in image_paths]
		return np.array(images)

	def rgb_callback(self, data):

		# self.face = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			real_mona_lisa_image_paths = ["src/mona_lisa.jpg"]
			real_images = self.load_images(real_mona_lisa_image_paths)


			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device, conf=0.75)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue



				# self.get_logger().info(f"Person has been detected!")

				bbox = bbox[0]

				# draw rectangle
				#cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)

				# Extract the region of interest (ROI) containing the detected face
				# I need this so I can publish the face as an image message to robot_commander and use it in model
				roi_width = int(bbox[2] - bbox[0])
				roi_height = int(bbox[3] - bbox[1])
				roi = cv_image[int(bbox[1]):int(bbox[1] + roi_height), int(bbox[0]):int(bbox[0] + roi_width)]


				# Convert ROI to a ROS Image message
				roi_msg = self.bridge.cv2_to_imgmsg(roi, "bgr8")
				bruh = self.bridge.imgmsg_to_cv2(roi_msg, desired_encoding="bgr8")

				"""img = cv2.resize(bruh, (224, 224))
				img = img.astype('float32') / 255
				img = np.expand_dims(img, axis=0)

				reconstructed_img = self.mona_model.predict(img)[0]

				# cv2.imshow()

				error_map = self.calculate_reconstruction_error(img[0], reconstructed_img)
				mean_error = np.mean(error_map)
				threshold = 0.038"""

				"""if mean_error and mean_error < threshold:
					self.get_logger().info(f"Model has rejected an image that potentially was a face")
					continue

				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)
				self.get_logger().info(f"Face Accepted")
				cv2.imshow("Should be a face", bruh)"""

				original_img = self.load_and_preprocess_image(bruh)
				original_img_batch = np.expand_dims(original_img, axis=0)
				reconstructed_img = self.model_mona_lisa(original_img_batch)

				error_map = self.calculate_reconstruction_error(original_img, reconstructed_img)
				mean_error = np.mean(error_map)

				threshold = 1.5 * np.mean(self.calculate_reconstruction_error(real_images, self.model_mona_lisa.predict(real_images)))
				print(threshold)
				print(mean_error)
				if mean_error and mean_error > threshold:
					print("NOT REAL MONA LISA")
					cv2.imshow("FAKE MONA LISA", bruh)
					if mean_error > threshold:
						result = "Altered Mona Lisa"
					else:
						result = "Real Mona Lisa"

					plt.figure(figsize=(10, 5))  # Adjusted figsize for the new subplot

					plt.subplot(1, 5, 1)  # Original Image
					plt.title('Original Image')
					plt.imshow(cv2.cvtColor((original_img * 255).astype('uint8'), cv2.COLOR_BGR2RGB))

					plt.subplot(1, 5, 2)  # Reconstructed Image
					plt.title('Reconstructed Image')
					plt.imshow(cv2.cvtColor((reconstructed_img.numpy() * 255).astype('uint8'), cv2.COLOR_BGR2RGB))  # Convert Tensor to NumPy array before calling astype

					plt.subplot(1, 5, 3)  # Reconstruction Error
					plt.title('Reconstruction Error')
					plt.imshow(np.squeeze(error_map), cmap='inferno')  # Use np.squeeze to remove singleton dimensions
					plt.colorbar()

					plt.subplot(1, 5, 4)  # Result
					plt.title(result)
					plt.text(0.5, 0.5, result, fontsize=15, ha='center')
					plt.axis('off')

					plt.show()

					continue

				print("WE FOUND TRUE MONA LISA!!!!!!!!")
				cv2.imshow("TRUE MONA LISA", bruh)

				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)



				#rc.info("RESIZING IMG")
				#img = cv2.resize(roi_msg, (224, 224))
				#img = img.astype('float32') / 255
				#img = np.expand_dims(img, axis=0)

				#reconstructed_img = model.predict(img)[0]

				#error_map = rc.calculate_reconstruction_error(img[0], reconstructed_img)
				#mean_error = np.mean(error_map)
				# Publish the ROI as an image message
				self.face_img_pub.publish(roi_msg)

				# self.get_logger().info(f"rawposition x {cx}")
				# self.get_logger().info(f"rawposition y {cy}")

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)

				self.face = (cx,cy)



			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()

		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step

		# iterate over face coordinates
		# for x, y in self.faces:
		if self.face == None:
			return

		# get 3-channel representation of the point cloud in numpy format
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))
		# read center coordinates
		x = int(self.face[0])
		y = int(self.face[1])
		self.get_logger().info(f"raw position x {x}")
		self.get_logger().info(f"raw position y {y}")

		d = a[y, x, :]
		# create marker
		marker = Marker()
		marker.header.frame_id = "/base_link"
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
		marker.color.g = 1.0
		marker.color.b = 1.0
		marker.color.a = 1.0
		# Set the pose of the marker
		marker.pose.position.x = float(d[0])
		marker.pose.position.y = float(d[1])
		marker.pose.position.z = float(d[2])
		# Check if coordinates are already in marked_faces within the threshold
		# already_marked = False
		# threshold = 0.1
		# for face in self.marked_faces:
		# 	if abs(face[0] - d[0]) < threshold and abs(face[1] - d[1]) < threshold:
		# 		already_marked = True
		# 		break
		# if not already_marked:
			# self.get_logger().info(f"published position x {d[0]}")
			# self.get_logger().info(f"published position y {d[1]}")
			# self.get_logger().info("Im smarter now and do not publish the same face twice")
		self.get_logger().info(f"published position x {d[0]}")
		self.get_logger().info(f"published position y {d[1]}")
		self.face = None
		self.marker_pub.publish(marker)

		point_msg = PointStamped()
		point_msg.header.frame_id = 'base_link'
		point_msg.header.stamp = self.get_clock().now().to_msg()
		point_msg.point = marker.pose.position
		self.face_pub.publish(point_msg)

			# self.marked_faces.append((d[0], d[1]))


def main():
	print('Face detection node starting.')
	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
