#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

/**
* My custom contour extraction
* Currently, only the right-angle corner is supported.
*
* @param img		input single-channel image (0 - background, 255 - footprint)
* @param contour	output contour polygon
*/
void findContour(const cv::Mat& img, std::vector<cv::Point>& contour) {
	contour.clear();

	// find the start point
	cv::Point start;
	bool found = false;
	for (int r = 0; r < img.rows && !found; r++) {
		for (int c = 0; c < img.cols; c++) {
			if (img.at<uchar>(r, c) == 255) {
				start = cv::Point(c, r);
				found = true;
				break;
			}
		}
	}

	cv::Point prev_dir(1, 0);
	cv::Point cur = start;
	contour.push_back(cur);
	int cnt = 0;
	do {
		cnt++;
		if (cnt > 10000) {
			break;
		}

		cv::Point left_dir(prev_dir.y, -prev_dir.x);
		cv::Point next = cur + left_dir;
		if (img.at<uchar>(next.y, next.x) == 255) {
			if (contour.size() > 0 && contour.back() != cur) contour.push_back(cur);
			cur = next;
			prev_dir = left_dir;
			continue;
		}

		cv::Point left_fore_dir = prev_dir + left_dir;
		if (std::abs(left_fore_dir.x) > 1) left_fore_dir.x /= std::abs(left_fore_dir.x);
		if (std::abs(left_fore_dir.y) > 1) left_fore_dir.y /= std::abs(left_fore_dir.y);
		next = cur + left_fore_dir;
		if (img.at<uchar>(next.y, next.x) == 255) {
			if (contour.size() > 0 && contour.back() != cur) contour.push_back(cur);
			cur = next;
			prev_dir = left_fore_dir;
			continue;
		}

		next = cur + prev_dir;
		if (img.at<uchar>(next.y, next.x) == 255) {
			cur = next;
			continue;
		}

		cv::Point right_fore_dir = prev_dir - left_dir;
		if (std::abs(right_fore_dir.x) > 1) right_fore_dir.x /= std::abs(right_fore_dir.x);
		if (std::abs(right_fore_dir.y) > 1) right_fore_dir.y /= std::abs(right_fore_dir.y);
		next = cur + right_fore_dir;
		if (img.at<uchar>(next.y, next.x) == 255) {
			if (contour.size() > 0 && contour.back() != cur) contour.push_back(cur);
			cur = next;
			prev_dir = right_fore_dir;
			continue;
		}

		cv::Point right_dir(-prev_dir.y, prev_dir.x);
		next = cur + right_dir;
		if (img.at<uchar>(next.y, next.x) == 255) {
			if (contour.size() > 0 && contour.back() != cur) contour.push_back(cur);
			cur = next;
			prev_dir = right_dir;
			continue;
		}

		cv::Point back_dir = -prev_dir;
		next = cur + back_dir;
		if (img.at<uchar>(next.y, next.x) == 255) {
			//contour.push_back(cur);
			cur = next;
			prev_dir = back_dir;
			continue;
		}

		break;
	} while (cur != start);
}

/**
* Regularize a polygon
*
* @param contour	input contour polygon
* @param result	output regularized polygon
*/
void regularizePolygon(std::vector<cv::Point> contour, std::vector<cv::Point>& result) {
	result.clear();

	float resolution = 5.0f;
	float area = cv::contourArea(contour);

	float min_cost = std::numeric_limits<float>::max();

	for (float angle = 0; angle < 180; angle += 10) {
		float theta = angle / 180 * CV_PI;
		for (int dx = 0; dx < resolution; dx++) {
			for (int dy = 0; dy < resolution; dy++) {
				// create a transformation matrix
				cv::Mat_<float> M = (cv::Mat_<float>(3, 3) << cos(theta) / resolution, -sin(theta) / resolution, dx / resolution, sin(theta) / resolution, cos(theta) / resolution, dy / resolution, 0, 0, 1);

				// create inverse transformation matrix
				cv::Mat_<float> invM = M.inv();

				// transform the polygon
				std::vector<cv::Point> polygon(contour.size());
				for (int i = 0; i < contour.size(); i++) {
					cv::Mat_<float> p = (cv::Mat_<float>(3, 1) << contour[i].x, contour[i].y, 1);
					cv::Mat_<float> p2 = M * p;
					polygon[i] = cv::Point(p2(0, 0), p2(1, 0));
				}

				// calculate the bounding box
				cv::Point min_pt(INT_MAX, INT_MAX);
				cv::Point max_pt(INT_MIN, INT_MIN);
				for (int i = 0; i < contour.size(); i++) {
					min_pt.x = std::min(min_pt.x, polygon[i].x - 3);
					min_pt.y = std::min(min_pt.y, polygon[i].y - 3);
					max_pt.x = std::max(max_pt.x, polygon[i].x + 3);
					max_pt.y = std::max(max_pt.y, polygon[i].y + 3);
				}

				// offset the polygon
				for (int i = 0; i < polygon.size(); i++) {
					polygon[i].x -= min_pt.x;
					polygon[i].y -= min_pt.y;
				}

				cv::Mat img(max_pt.y - min_pt.y + 1, max_pt.x - min_pt.x + 1, CV_8U, cv::Scalar(0));

				// draw a polygon
				std::vector<std::vector<cv::Point>> polygons;
				polygons.push_back(polygon);
				cv::fillPoly(img, polygons, cv::Scalar(255), cv::LINE_8);

				//cv::imwrite("test.png", img);

				// extract a contour (my custom version)
				std::vector<cv::Point> new_contour;
				findContour(img, new_contour);

				float a = cv::contourArea(new_contour) * resolution * resolution;
				if (new_contour.size() >= 3 && a > 0) {
					// convert the polygon back to the original coordinates
					for (int i = 0; i < new_contour.size(); i++) {
						cv::Mat_<float> p = (cv::Mat_<float>(3, 1) << new_contour[i].x + min_pt.x, new_contour[i].y + min_pt.y, 1);
						cv::Mat_<float> p2 = invM * p;
						new_contour[i] = cv::Point(p2(0, 0), p2(1, 0));
					}

					// calculate cost
					float cost = 0.0f;
					for (int i = 0; i < new_contour.size(); i++) {
						float min_dist = std::numeric_limits<float>::max();
						for (int j = 0; j < contour.size(); j++) {
							float dist = cv::norm(new_contour[i] - contour[j]);
							min_dist = std::min(min_dist, dist);
						}
						cost += min_dist;
					}
					cost /= new_contour.size();

					// calculate the cost
					cost += new_contour.size() * 0.3;

					if (cost < min_cost) {
						min_cost = cost;

						result = new_contour;
					}
				}
			}
		}
	}
}

/**
* Convert a slice image to a footprint image.
*/
void convert(cv::Mat img, cv::Mat& result, float eps) {
	// initialize result
	result = cv::Mat(img.size(), CV_8U, cv::Scalar(255));

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	// extract contours
	cv::findContours(img.clone(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	for (int i = 0; i < contours.size(); i++) {
		// simplify contours
		std::vector<cv::Point> approx_contour;
		cv::approxPolyDP(contours[i], approx_contour, eps, true);

		if (approx_contour.size() >= 3) {
			// regularize a contour
			regularizePolygon(approx_contour, approx_contour);
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