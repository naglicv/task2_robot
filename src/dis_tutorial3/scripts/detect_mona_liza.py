 
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO

import tensorflow as tf
import matplotlib.pyplot as plt

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

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

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.face_img_pub = self.create_publisher(Image, "/detected_face", QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

		self.model_mona_lisa = tf.keras.models.load_model('/home/kappa/Desktop/task2_robot-anomaly_detection/src/dis_tutorial3/scripts/anomaly.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError})


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

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			real_mona_lisa_image_paths = ["src/mona_lisa.jpg"]
			real_images = self.load_images(real_mona_lisa_image_paths)


			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device, conf=0.7)

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
				img = img.astype("float32") / 255.0
				img = np.expand_dims(img, axis=0)

				reconstructed_img = self.model_mona_lisa.predict(img)[0]

				error_map = self.calculate_reconstruction_error(img[0], reconstructed_img)
				mean_error = np.mean(error_map)


				threshold = 0.03"""
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

				self.get_logger().info(f"rawposition x {cx}")
				self.get_logger().info(f"rawposition y {cy}")

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)



				self.faces.append((cx,cy))



			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()

		except CvBridgeError as e:
			print(e)

	#def calculate_reconstruction_error(self, original, reconstructed):
    #	return np.mean((original - reconstructed) ** 2, axis=-1)
	def calculate_reconstruction_error(self, original, reconstructed):
		return np.mean((original - reconstructed)**2, axis=-1)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step

		# iterate over face coordinates
		for x,y in self.faces:

			# get 3-channel representation of the poitn cloud in numpy format
			a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
			a = a.reshape((height,width,3))

			# read center coordinates
			d = a[y,x,:]

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

			self.get_logger().info(f"position x {d[0]}")
			self.get_logger().info(f"position y {d[1]}")

			self.marker_pub.publish(marker)

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
