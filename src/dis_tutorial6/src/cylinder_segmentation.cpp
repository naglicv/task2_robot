#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/visualization_msgs/msg/marker.hpp"

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr planes_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;

std::shared_ptr<rclcpp::Node> node;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

typedef pcl::PointXYZRGB PointT;

int marker_id = 0;
float error_margin = 0.02;  // 2 cm margin for error
float target_radius = 0.11;
bool verbose = false;

std::string target_color = "unknown";
std::vector<std::string> colorNames = {"red", "green", "blue", "black", "yellow"};
std::vector<std_msgs::msg::ColorRGBA> colorList;

void initializeColorList() {
    // Red
    std_msgs::msg::ColorRGBA red;
    red.r = 1.0f;
    red.g = 0.0f;
    red.b = 0.0f;
    red.a = 1.0f;
    colorList.push_back(red);

    // Green
    std_msgs::msg::ColorRGBA green;
    green.r = 0.0f;
    green.g = 1.0f;
    green.b = 0.0f;
    green.a = 1.0f;
    colorList.push_back(green);

    // Blue
    std_msgs::msg::ColorRGBA blue;
    blue.r = 0.0f;
    blue.g = 0.0f;
    blue.b = 1.0f;
    blue.a = 1.0f;
    colorList.push_back(blue);

    // Black
    std_msgs::msg::ColorRGBA black;
    black.r = 0.0f;
    black.g = 0.0f;
    black.b = 0.0f;
    black.a = 1.0f;
    colorList.push_back(black);

    // Yellow
    std_msgs::msg::ColorRGBA yellow;
    yellow.r = 1.0f;
    yellow.g = 1.0f;
    yellow.b = 0.0f;
    yellow.a = 1.0f;
    colorList.push_back(yellow);
}


std::string getColor(int r, int g, int b) {
    // Define a threshold for determining if a color component is "significant"
    float threshold = 70;

    // Check for black
    if (r < threshold && g < threshold && b < threshold) {
        return "black";
    }

    // Check for red
    if (r > threshold && g < threshold && b < threshold) {
        return "red";
    }

    // Check for green
    if (r < threshold && g > threshold && b < threshold) {
        return "green";
    }

    // Check for blue
    if (r < threshold && g < threshold && b > threshold) {
        return "blue";
    }

    // Check for yellow (red + green)
    if (r > threshold && g > threshold && b < threshold) {
        return "yellow";
    }

    // If none of the above conditions are met, return "unknown"
    return "unknown";
}


bool filterByColor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    // Calculate the average color of the point cloud
    int total_r = 0, total_g = 0, total_b = 0;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        pcl::PointXYZRGB& point = cloud->points[i];
        total_r += point.r;
        total_g += point.g;
        total_b += point.b;
    }
    int avg_r = total_r / cloud->points.size();
    int avg_g = total_g / cloud->points.size();
    int avg_b = total_b / cloud->points.size();

    // Define the threshold for the color difference
    int threshold = 40;  // Adjust this value as needed

    // std::cout << "Red: " << avg_r << " Green: " << avg_g << " Blue: " << avg_b << std::endl;
    target_color = getColor(avg_r, avg_g, avg_b);

    // If the average color components are close to each other, return false
    if ((std::abs(avg_r - avg_g) <= threshold && std::abs(avg_g - avg_b) <= threshold && std::abs(avg_r - avg_b) <= threshold && !(avg_r < 20 && avg_g < 20 && avg_b < 20)) || target_color == "unknown") {
        //std::cout << "Discard" << std::endl;
        return false;
    }


    // If the average color components are not close to each other, return true
    return true;
}

void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // save timestamp from message
    rclcpp::Time now = (*msg).header.stamp;

    // set up PCL objects
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::PCDWriter writer;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    Eigen::Vector4f centroid;

    // set up pointers
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PCLPointCloud2::Ptr pcl_pc(new pcl::PCLPointCloud2);
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

    // convert ROS msg to PointCloud2
    pcl_conversions::toPCL(*msg, *pcl_pc);

    // convert PointCloud2 to templated PointCloud
    pcl::fromPCLPointCloud2(*pcl_pc, *cloud);

    if (verbose) {
        std::cerr << "PointCloud has: " << cloud->points.size() << " data points." << std::endl;
    }

    // Build a passthrough filter to remove spurious NaNs
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0, 10);
    pass.filter(*cloud_filtered);
    if (verbose) {
        std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;
    }

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_filtered);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);

    // Create the segmentation object for the planar model and set all the
    // parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);

    seg.segment(*inliers_plane, *coefficients_plane);
    if (verbose) {
        std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
    }

    // Extract the planar inliers from the input cloud
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);
    extract.filter(*cloud_plane);

    // Remove the planar inliers, extract the rest
    extract.setNegative(true);
    extract.filter(*cloud_filtered2);
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(cloud_normals);
    extract_normals.setIndices(inliers_plane);
    extract_normals.filter(*cloud_normals2);

    // Create the segmentation object for cylinder segmentation and set all the
    // parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setRadiusLimits(0.06, 0.15);
    seg.setInputCloud(cloud_filtered2);
    seg.setInputNormals(cloud_normals2);

    // Obtain the cylinder inliers and coefficients
    seg.segment(*inliers_cylinder, *coefficients_cylinder);

    // return if no cylinder was detected
    int coef_size = (*coefficients_cylinder).values.size();
    if (coef_size == 0) {
        return;
    }

    if (verbose) {
        std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;
    }

    float detected_radius = (*coefficients_cylinder).values[6];

    if (std::abs(detected_radius - target_radius) > error_margin) {
        return;
    }

    // extract cylinder
    extract.setInputCloud(cloud_filtered2);
    extract.setIndices(inliers_cylinder);
    extract.setNegative(false);
    pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_cylinder);

    // std::cout << "Red: " << int(cloud_cylinder->points[0].r) << " Green: " << int(cloud_cylinder->points[0].g) << " Blue: " << int(cloud_cylinder->points[0].b) << std::endl;
    if (!filterByColor(cloud_cylinder)) {
        return;
    }

    // calculate marker
    pcl::compute3DCentroid(*cloud_cylinder, centroid);
    if (verbose) {
        std::cerr << "centroid of the cylindrical component: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3] << std::endl;
    }

    geometry_msgs::msg::PointStamped point_camera;
    geometry_msgs::msg::PointStamped point_map;
    visualization_msgs::msg::Marker marker;
    geometry_msgs::msg::TransformStamped tss;

    // set up marker messages
    std::string toFrameRel = "map";
    std::string fromFrameRel = (*msg).header.frame_id;
    point_camera.header.frame_id = fromFrameRel;

    point_camera.header.stamp = now;
    point_camera.point.x = centroid[0];
    point_camera.point.y = centroid[1];
    point_camera.point.z = centroid[2];

    try {
        tss = tf_buffer_->lookupTransform(toFrameRel, fromFrameRel, now);
        tf2::doTransform(point_camera, point_map, tss);
    } catch (tf2::TransformException& ex) {
        std::cout << ex.what() << std::endl;
    }

    if (verbose) {
        std::cerr << "point_camera: " << point_camera.point.x << " " << point_camera.point.y << " " << point_camera.point.z << std::endl;
        std::cerr << "point_map: " << point_map.point.x << " " << point_map.point.y << " " << point_map.point.z << std::endl;
    }

    // publish marker
    marker.header.frame_id = "map";
    marker.header.stamp = now;

    marker.ns = "cylinder";
    // marker.id = 0; // only latest marker
    marker.id = marker_id++;  // generate new markers

    marker.type = visualization_msgs::msg::Marker::CYLINDER;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = point_map.point.x;
    marker.pose.position.y = point_map.point.y;
    marker.pose.position.z = point_map.point.z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    std::cout << "Detected " << target_color << " cylinder" << std::endl;
    int index = -1;
    for (size_t i = 0; i < colorNames.size(); ++i) {
        if (colorNames[i] == target_color) {
            index = i;
            break;
        }
    }
    if (index == -1) {
        std::cerr << "Color not found" << std::endl;
        marker.color = colorList[0];
    } else {
        marker.color = colorList[index];
    }
    // marker.color.r = 0.0f;
    // marker.color.g = 1.0f;
    // marker.color.b = 0.0f;
    // marker.color.a = 1.0f;
    

    // marker.lifetime = rclcpp::Duration(1,0);
    marker.lifetime = rclcpp::Duration(0,0);

    marker_pub->publish(marker);

    //////////////////////////// publish result point clouds /////////////////////////////////

    // convert to pointcloud2, then to ROS2 message
    sensor_msgs::msg::PointCloud2 plane_out_msg;
    pcl::PCLPointCloud2::Ptr outcloud_plane(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*cloud_plane, *outcloud_plane);
    pcl_conversions::fromPCL(*outcloud_plane, plane_out_msg);
    planes_pub->publish(plane_out_msg);

    // publish cylinder
    sensor_msgs::msg::PointCloud2 cylinder_out_msg;
    pcl::PCLPointCloud2::Ptr outcloud_cylinder(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*cloud_cylinder, *outcloud_cylinder);
    pcl_conversions::fromPCL(*outcloud_cylinder, cylinder_out_msg);
    cylinder_pub->publish(cylinder_out_msg);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "cylinder_segmentation" << std::endl;

    node = rclcpp::Node::make_shared("cylinder_segmentation");

    // create subscriber
    node->declare_parameter<std::string>("topic_pointcloud_in", "/oakd/rgb/preview/depth/points");
    std::string param_topic_pointcloud_in = node->get_parameter("topic_pointcloud_in").as_string();
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription = node->create_subscription<sensor_msgs::msg::PointCloud2>(param_topic_pointcloud_in, 10, &cloud_cb);

    // setup tf listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // create publishers
    planes_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("planes", 1);
    cylinder_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("cylinder", 1);
    marker_pub = node->create_publisher<visualization_msgs::msg::Marker>("detected_cylinder", 1);
    initializeColorList();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
