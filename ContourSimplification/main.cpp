#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include "ContourUtils.h"

/**
* Convert a slice image to a footprint image.
*/
void convert(cv::Mat img, cv::Mat& result, float eps) {
	// initialize result
	result = cv::Mat(img.size(), CV_8U, cv::Scalar(255));

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	// dilate the image to the right and below
	/*
	cv::Mat_<uchar> kernel = (cv::Mat_<uchar>(3, 3) << 1, 1, 0, 1, 1, 0, 0, 0, 0);
	cv::Mat inflated;
	cv::dilate(img, inflated, kernel);
	*/

	// extract contours
	cv::findContours(img.clone(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	for (int i = 0; i < contours.size(); i++) {
		std::vector<cv::Point> approx_contour;
		contour::simplify(contours[i], approx_contour);

		if (approx_contour.size() >= 3) {
			cv::polylines(result, approx_contour, true, cv::Scalar(0), 1);
		}
	}
}

int main() {
	std::string dir_name("data");
	std::string output_name("results");

	for (auto it = boost::filesystem::directory_iterator(dir_name); it != boost::filesystem::directory_iterator(); it++) {
		if (!boost::filesystem::is_directory(it->path())) {
			std::cout << it->path().filename().string() << std::endl;

			// read an image
			cv::Mat img = cv::imread(dir_name + "/" + it->path().filename().string(), cv::IMREAD_GRAYSCALE);

			// generate a footprint image
			cv::Mat result;
			convert(img, result, 1);

			// merge two images
			cv::Mat merged(img.rows, img.cols + result.cols, CV_8U);
			img.copyTo(merged(cv::Rect(0, 0, img.cols, img.rows)));
			result.copyTo(merged(cv::Rect(img.cols, 0, result.cols, result.rows)));

			cv::imwrite(output_name + "/" + it->path().filename().string(), merged);
		}
	}

	return 0;
}