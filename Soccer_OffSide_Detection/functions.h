
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <opencv2/opencv.hpp>
#include <string>


using namespace cv;

// Function declarations
int addNumbers(int a, int b);

void segmentation_kmeans(const std::string& path);

void get_pixels_position(Mat segmented, int number_clusters, Mat originalImage);

void Line_Detection(Mat img);

void Line_Detection_Track(Mat img);

void person_detection(const std::string& path);

void person_detection_lowerbody(const std::string& path);

void segmentation_with_ranges(Mat img);

void segmentation(Mat img);

void checkOffside(const std::vector<std::string>& stringList, int id);	

//void onTrackbarChange(int, void*);
#endif