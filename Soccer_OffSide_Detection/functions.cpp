#include "functions.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

struct ImagePair {
    cv::Mat image1;
    cv::Mat image2;
};

// Function implementations
int addNumbers(int a, int b) {
    return a + b;
}

void checkOffside(const std::vector<std::string>& stringList, int id) {
    std::string input;
    std::string color_of_closest_player;
    //std::cout << id << std::endl;

    //std::cout << stringList.size() << std::endl;    // number of players
    color_of_closest_player = stringList[id];
    std::cout << "Color of closest player: " << color_of_closest_player << std::endl;

    //print all colors
    //for (const auto& str : stringList) {
    //    std::cout << str << std::endl;
    //}
     
    std::cout << "-------------------------------- " << std::endl;
    // Prompt the user for input
    std::cout << "Is the attacking team red? (yes/no): " << std::endl;
    std::cin >> input;

    // Convert the input to lowercase for case-insensitive comparison
    for (char& c : input) {
        c = std::tolower(c);
    }

    // Check if the input is "yes"
    if (input == "yes" && color_of_closest_player == "Red") {
        std::cout << "OffSide detected! Ilegal Maneuver" << std::endl;
    }

    else {
        std::cout << "There is no OffSide, Legal Maneuver!" << std::endl;

    }
}


//Segmentation with threshold
void segmentation(Mat img) {
    int ksize = 5;
    cv::Mat gray, temp, mask;
    //convert input image to grayscale
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    //apply blur filter
    cv::blur(gray, temp, cv::Size(ksize, ksize));
    //Otsu optimal threshold to output image
    double value = cv::threshold(temp, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::printf("Otsu threshold: %f\n", value);

    // Display the image with bounding boxes
    imshow("Image work", mask);
    waitKey(0);

}


//segmentation with ranges function
void segmentation_with_ranges(Mat img) {

    // Convert to HSV color space
    Mat hsv_image, gray_image, blur_image, segmented_image2;
    cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);    //COLOR_BGR2HSV  hsv_image
    bilateralFilter(gray_image, blur_image, 9, 75, 75);

    //Apply a color threshold to segment the image based on the colors of the players' jerseys
    Mat mask1, mask2, segmented_image;
    int b1 = 0, g1 = 0, r1 = 50; // lower bounds for team 1 red
    int B1 = 110, G1 = 170, R1 = 255; // upper bounds for team 1 red
    int b2 = 10, g2 = 0, r2 = 0; // lower bounds for team 2 blue
    int B2 = 255, G2 = 120, R2 = 120; // upper bounds for team 2 blue

    inRange(img, Scalar(b1, g1, r1), Scalar(B1, G1, R1), mask1); // team 1 color range
    inRange(img, Scalar(b2, g2, r2), Scalar(B2, G2, R2), mask2); // team 2 color range
    bitwise_or(mask1, mask2, segmented_image); // combine the two masks

    //Apply a morphological opening to remove small objects and fill gaps in the segmented image
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(segmented_image, segmented_image2, MORPH_OPEN, kernel);

    // Display the image with bounding boxes
    imshow("Image gray", gray_image);
    imshow("Image Original", img);
    imshow("Image Blur", blur_image);
    imshow("Image segmented", segmented_image);
    imshow("Image segmented2", segmented_image2);
    waitKey(0);
}

//Person Detection with Haar Cascade
void person_detection(const std::string & path) {

    // Load image
    Mat image = imread(path);
    // Load the pre-trained Haar Cascade classifier for person detection
    cv::CascadeClassifier cascade;
    cascade.load(cv::samples::findFile("haarcascade_fullbody.xml"));

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Detect persons in the image
    std::vector<cv::Rect> persons;
    cascade.detectMultiScale(gray, persons, 1.1, 1, 0, cv::Size(10, 10), cv::Size(200, 200));

    // Draw bounding boxes around the detected persons
    for (size_t i = 0; i < persons.size(); ++i)
    {
        cv::rectangle(image, persons[i], cv::Scalar(0, 255, 0), 2);
    }

    // Display the image with detections
    cv::imshow("Person Detection", image);
    cv::waitKey(0);

}

//Person Detection with Haar Cascade upperbody
void person_detection_lowerbody(const std::string& path) {

    // Load image
    Mat image = imread(path);
    // Load the pre-trained Haar Cascade classifier for person detection
    cv::CascadeClassifier cascade;
    cascade.load(cv::samples::findFile("haarcascade_upperbody.xml")); //haarcascade_lowerbody.xml

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect persons in the image
    std::vector<cv::Rect> persons;
    cascade.detectMultiScale(gray, persons, 1.1, 3, 0, cv::Size(30, 30));

    // Draw bounding boxes around the detected persons
    for (size_t i = 0; i < persons.size(); ++i)
    {
        cv::rectangle(image, persons[i], cv::Scalar(0, 255, 0), 2);
    }

    // Display the image with detections
    cv::imshow("Person Detection", image);
    cv::waitKey(0);

}


// Trackbar callback function
void onTrackbarChange(int value, void* userData)
{
    
    // Access the additional image passed as an argument
    ImagePair* imagesPtr = static_cast<ImagePair*>(userData);
    cv::Mat& image1 = imagesPtr->image1;  //edges
    cv::Mat& image2 = imagesPtr->image2;  //original
    
    int thresholdValue_ = value;
    // Perform Hough line detection with the updated threshold value
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(image1, lines, 1, CV_PI / 150, thresholdValue_);

    std::cout << "number of lines: " << lines.size() << std::endl;

    cv::Mat image_lines = image2.clone();
    // Draw detected lines on image
    // Filter out horizontal and similar lines
    for (size_t i = 0; i < lines.size(); i++)            //size_t i = 0; i < lines.size()
    {
        float rho1 = lines[i][0], theta1 = lines[i][1];

        // Ignore horizontal lines if (theta1 < CV_PI / 4 || theta1 > 3 * CV_PI / 4)
        if (theta1 < CV_PI / 180 * 50 || theta1 > CV_PI / 180 * 130)
        {
            continue;
        }

        bool duplicate = false;
        for (size_t j = 0; j < i; j++)
        {
            float rho2 = lines[j][0], theta2 = lines[j][1];

            // Ignore horizontal lines
            if (theta2 < CV_PI / 180 * 50 || theta2 > CV_PI / 180 * 130)
            {
                continue;
            }

            // Check if the two lines are very similar
            if (abs(rho1 - rho2) < 10 && abs(theta1 - theta2) < CV_PI / 36)  // 20 36
            {
                duplicate = true;
                continue;   //break
            }
        }

        // If the line is not a duplicate, draw it on the output image
        if (!duplicate)
        {
            Point pt1, pt2;
            double a = cos(theta1), b = sin(theta1);
            double x0 = a * rho1, y0 = b * rho1;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            line(image_lines, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
            std::cout << "Line " << i + 1 << ": rho = " << rho1 << ", theta = " << theta1 << std::endl;
            std::cout << "pt1: (" << pt1.x << ", " << pt1.y << ")" << std::endl;
            std::cout << "pt2: (" << pt2.x << ", " << pt2.y << ")" << std::endl;

        
        }
    }
    // Show the result image
    cv::imshow("Window Name", image_lines);
}


void Line_Detection_Track(Mat img) {

    // Global variables for trackbar callback
    int thresholdValue = 150;

    // Load the image and perform edge detection
    std::cout << "Enter Line Detection" << std::endl;

    // Convert image to grayscale
    Mat gray, gray2;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Apply Canny edge detection generate_edge_map
    Mat edges;
    Canny(gray, edges, 50, 150);

    // Create a window to display the image
    cv::namedWindow("Hough Lines");

    
    ImagePair images;
    images.image1 = edges;
    images.image2 = img.clone();

    // Create a trackbar to adjust the threshold value  and call the callback function
    Mat ogimage = img.clone();
    cv::createTrackbar("Threshold", "Hough Lines", &thresholdValue, 255, onTrackbarChange, &images);

    // Display the initial image
    cv::imshow("Hough Lines", edges);

    // Wait for key press
    cv::waitKey(0);



}



void Line_Detection(Mat img)
{
    std::cout << "Enter Line Detection" << std::endl;
    // Global variables for trackbar callback
    int thresholdValue = 150;

    // Convert image to grayscale
    Mat gray, gray2;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Apply Canny edge detection generate_edge_map
    Mat edges;
    Canny(gray, edges, 30, 40);

    //hough_transforms(edges);
    std::vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 50, 100);

    //Mat houghImg = Mat::zeros(img.size(), CV_8UC3);
    //Mat houghImg;
    //cvtColor(edges, houghImg, COLOR_GRAY2BGR);
    Mat houghImg = img.clone(); // Create a copy of the original image



     std::cout << "number of lines: " << lines.size();

    // Draw detected lines on image
    // Filter out horizontal and similar lines
    for (size_t i = 0; i < lines.size(); i++)            //size_t i = 0; i < lines.size()
    {
        float rho1 = lines[i][0], theta1 = lines[i][1];

        // Ignore horizontal lines
        if (theta1 < CV_PI / 4 || theta1 > 3 * CV_PI / 4)
        {
            continue;
        }

        bool duplicate = false;
        for (size_t j = 0; j < i; j++)
        {
            float rho2 = lines[j][0], theta2 = lines[j][1];

            // Ignore horizontal lines
            if (theta2 < CV_PI / 4 || theta2 > 3 * CV_PI / 4)
            {
                continue;
            }

            // Check if the two lines are very similar
            if (abs(rho1 - rho2) < 20 && abs(theta1 - theta2) < CV_PI / 36)
            {
                duplicate = true;
                break;
            }
        }

        // If the line is not a duplicate, draw it on the output image
        if (!duplicate)
        {
            Point pt1, pt2;
            double a = cos(theta1), b = sin(theta1);
            double x0 = a * rho1, y0 = b * rho1;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            line(houghImg, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
        }
    }



    // Display image
    imshow("OG Image", img);
    imshow("grayed", gray);
    imshow("Edges", edges);
    //namedWindow("Detected Lines", WINDOW_NORMAL);
    imshow("houghlines", houghImg);
    waitKey(0);
}



void segmentation_kmeans(const std::string& path){
    // Load image
    Mat image = imread(path);

    // Convert image to float data type
    Mat samples;
    image.convertTo(samples, CV_32F);

    // Reshape image to a 2D matrix of pixels
    Mat samples_2d = samples.reshape(1, samples.rows * samples.cols);

    // Perform K-means clustering
    int k = 5;  // Number of clusters // 5 and 6 works for image 3
    Mat labels, centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.5);
    double compacted =kmeans(samples_2d, k, labels, criteria, 3, KMEANS_RANDOM_CENTERS, centers);

    std::cout << "fine til here" << std::endl;


   
    // Assign colors to each pixel based on the cluster label
    Mat segmented_image = Mat::zeros(samples_2d.rows, 1, CV_8UC3);
    Vec3b* segmented_pixels = segmented_image.ptr<Vec3b>();
    std::cout << "fine til here" << std::endl;

    cv::Mat coloredImage = cv::Mat::zeros(image.size(), CV_8UC3);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int label = labels.at<int>(y * image.cols + x); // Get the cluster label for the pixel

            // Assign colors based on the cluster label
            if (label == 0) {
                coloredImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0); // Red color
            }
            else if (label == 1) {
                coloredImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Blue color
            }
            else if (label == 2) {
                coloredImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Green color
            }
            else if (label == 3) {
                coloredImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 255); // Yellow color
            }
            else if (label == 4) {
                coloredImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 0); // Cyan color
            }
            else if (label == 5) {
                coloredImage.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 0, 128); // Purple color
            }
            // Add more conditions for additional cluster labels and corresponding colors if needed
        }
    }

    //get_pixels_position(coloredImage, k, image);

    imshow("Original Image", image);
    imshow("Segmented Image", coloredImage);
    waitKey(0);


    /*
    for (int i = 0; i < samples_2d.rows; ++i) {
        int cluster_label = labels.at<int>(i);
        segmented_pixels[i] = centers.at<Vec3f>(cluster_label);
    }
    */

    /*

    
    // Reshape segmented image back to the original size
    segmented_image = segmented_image.reshape(3, image.rows);

    // Display the original and segmented images
    imshow("Original Image", image);
    imshow("Segmented Image", segmented_image);
    waitKey(0);

    */



}


//doesnt work well
void get_pixels_position(Mat segmentedImage, int number_clusters, Mat originalImage) {
    std::cout << "fine til here, enter pixel positions" << std::endl;
    int k = number_clusters;
    // Calculate the percentage of yellow and blue pixels in each cluster
    std::vector<int> yellowPixelCounts(k, 0);
    std::vector<int> bluePixelCounts(k, 0);
    for (int y = 0; y < segmentedImage.rows; y++) {
        for (int x = 0; x < segmentedImage.cols; x++) {
            int label = segmentedImage.at<int>(y, x);
            cv::Vec3b pixel = originalImage.at<cv::Vec3b>(y, x); // Original image

            if (label == 0) {
                // Check if pixel is yellow
                if (pixel[2] > pixel[0] && pixel[2] > pixel[1]) {
                    yellowPixelCounts[label]++;
                }
            }
            else if (label == 1) {
                // Check if pixel is blue
                if (pixel[0] > pixel[1] && pixel[0] > pixel[2]) {
                    bluePixelCounts[label]++;
                }
            }
            else if (label == 2) {
                // Check if pixel is blue
                if (pixel[0] > pixel[1] && pixel[0] > pixel[2]) {
                    bluePixelCounts[label]++;
                }

            }
            else if (label == 3) {
                // Check if pixel is yellow
                if (pixel[2] > pixel[0] && pixel[2] > pixel[1]) {
                    yellowPixelCounts[label]++;
                }
            }
            else if (label == 4) {
                // Check if pixel is yellow
                if (pixel[2] > pixel[0] && pixel[2] > pixel[1]) {
                    yellowPixelCounts[label]++;
                }
            }
            else if (label == 5) {
                // Check if pixel is blue
                if (pixel[0] > pixel[1] && pixel[0] > pixel[2]) {
                    bluePixelCounts[label]++;
                }
            }
        }
    }

    // Define the threshold for mostly yellow and mostly blue regions
    double yellowThreshold = 0.5; // Adjust as needed
    double blueThreshold = 0.5; // Adjust as needed

    // Get the pixel positions for mostly yellow regions
    std::vector<cv::Point> yellowPixelPositions;
    for (int i = 0; i < k; i++) {
        double yellowPercentage = static_cast<double>(yellowPixelCounts[i]) / (segmentedImage.rows * segmentedImage.cols);
        if (yellowPercentage > yellowThreshold) {
            // Add pixel positions of the current cluster to the yellowPixelPositions vector
            for (int y = 0; y < segmentedImage.rows; y++) {
                for (int x = 0; x < segmentedImage.cols; x++) {
                    int label = segmentedImage.at<int>(y, x);
                    if (label == i) {
                        yellowPixelPositions.push_back(cv::Point(x, y));
                    }
                }
            }
        }
    }

    // Get the pixel positions for mostly blue regions
    std::vector<cv::Point> bluePixelPositions;
    for (int i = 0; i < k; i++) {
        double bluePercentage = static_cast<double>(bluePixelCounts[i]) / (segmentedImage.rows * segmentedImage.cols);
        if (bluePercentage > blueThreshold) {
            // Add pixel positions of the current cluster to the bluePixelPositions vector
            for (int y = 0; y < segmentedImage.rows; y++) {
                for (int x = 0; x < segmentedImage.cols; x++) {
                    int label = segmentedImage.at<int>(y, x);
                    if (label == i) {
                        bluePixelPositions.push_back(cv::Point(x, y));
                    }
                }
            }
        }
    }

    // Print the pixel positions for mostly yellow regions
    std::cout << "Pixel positions for mostly yellow regions:" << std::endl;
    for (const cv::Point& position : yellowPixelPositions) {
        std::cout << "x: " << position.x << ", y: " << position.y << std::endl;
    }

    // Print the pixel positions for mostly blue regions
    std::cout << "Pixel positions for mostly blue regions:" << std::endl;
    for (const cv::Point& position : bluePixelPositions) {
        std::cout << "x: " << position.x << ", y: " << position.y << std::endl;
    }

}
