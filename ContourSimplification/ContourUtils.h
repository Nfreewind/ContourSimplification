#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace contour {

	void simplify(std::vector<cv::Point> contour, std::vector<cv::Point>& result, float eps = 1.0f);
	void regularizePolygon(std::vector<cv::Point> contour, std::vector<cv::Point>& result);
	void findContour(const cv::Mat& img, std::vector<cv::Point>& contour);

}
