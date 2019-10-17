/*
 * rs_to_pcl.cpp
 *
 *  Created on: Sep 15, 2019
 *      Author: gachiemchiep
 */

#include <librealsense2/rs.hpp>
#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include "c44_namespace.hpp"

int main(int argc, char *argv[]) {

	rs2::log_to_console(RS2_LOG_SEVERITY_WARN);

	// Check whether device is connected
	rs2::context ctx;

	if (ctx.query_devices().size() == 0) {
		std::cerr << "NO DEVICE FOUND" << std::endl;
		return EXIT_FAILURE;
	}

	// Create a Pipeline - this serves as a top-level API for streaming and processing frames
	rs2::pipeline pipe(ctx);
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_DEPTH);
	cfg.enable_stream(RS2_STREAM_COLOR);

	// Configure and start the pipeline
	pipe.start(cfg);

	pcl::PointCloud<pcl::PointXYZ>::Ptr originalCloud;

	// setup viewer?
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	c44::AddCloudToVisualizer(originalCloud, viewer, 255, 255, 255, "sample cloud1");

	// loop that stays active as long as the viewer is running
	while (!viewer->wasStopped()) {
		viewer->removeAllPointClouds();
		//viewer->spinOnce(100);
		// grab a frame from realsense and display it

		// filter frame (reduce point count, remove obvious outliers, etc)

		//c44::DownsampleCloud(originalCloud);
		c44::RemoveNansFromCloud (originalCloud);
		// remove old cloud from display and add new filtered cloud to display
		// c44::overwriteVis(originalCloud, viewer, 255, 255, 255);
		// viewer->spinOnce(100);

		// segment largest plane from image
		// estimate cloud normals
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
		c44::EstimateCloudNormals(originalCloud, cloud_normals);

		// extract plane and display the segmented plane in a specific color (e.g. white)
		// remove plane from cloud
		pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPlane = c44::SegmentPlane(originalCloud, cloud_normals,
				coefficients_plane);
		c44::AddCloudToVisualizer(cloudPlane, viewer, 255, 255, 255, "plane");
		//viewer->spinOnce(100);

		double green = 0;
		double blue = 0;
		bool success = true;
		int i = 0;
		while (success && !cloudPlane->points.empty() && i < 5) {
			// loop that segments objects out until there are no more.

			// segment extract and remove ith object
			pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCylinder = c44::ExtractObjects(originalCloud, cloud_normals,
					coefficients_cylinder);

			// extract object and display in a specific color (changes somehow for each object)
			c44::AddCloudToVisualizer(cloudCylinder, viewer, 255, green, blue, "object_" + std::to_string(i));

			// check if a cylinder was found
			if (cloudCylinder->points.empty()) {
				std::cerr << "Can't find the cylindrical component." << std::endl;
				success = false;
			} else {
				viewer->spinOnce(100);
				std::cerr << std::to_string(i) + " Total PointClouds representing cylindrical components: "
						<< cloudCylinder->points.size() << " data points." << std::endl;
				success = true;
			}

			// get ready for next loop
			i++;

			// object clouds will be colored from red to green to white and then back to red
			if (green < 255)
				green += 51;
			else if (blue < 255)
				blue += 51;
			else
				blue = green = 0;

			// end object segmenatation loop
		}

		// wait for a second?
		boost::this_thread::sleep(boost::posix_time::microseconds(1000000));

		// clear the display so it is ready for the next frame?
		//originalCloud.reset();
		c44::GrabRealsenseFrame(dev, 1.5, originalCloud);
		//c44::overwriteVis(originalCloud, viewer, 255, 255, 255);
		//pcl::io::savePCDFile("in_driver.pcd", *originalCloud);

		//viewer.reset();
		// end viewer loop
	}

	return EXIT_SUCCESS;

}

