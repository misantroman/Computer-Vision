///Victor Velazquez 2043179

#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

//Declaration of functions
void Line_Detection(Mat img);
Mat hough_transforms(Mat img);


int main() {
    // Load image
    //string path = "C:Users\velaz\OneDrive\Documents\Control Systems Master\4th Semester\Computer Vision\Lab3\road1.png";
    //C:\Users\velaz\source\repos\opencvtest\opencvtest
    string path = "road8.jpg";
    Mat img = imread(path);
    // Check if image is loaded successfully
    if (img.empty()) {
        std::cout << "Failed to load image!" << std::endl;
        return -1;
    }

    Line_Detection(img);

    return 0;
}




// Definition of functions


void Line_Detection(Mat img)
{
    // Convert image to grayscale
    Mat gray, gray2;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Apply Canny edge detection generate_edge_map
    Mat edges;
    Canny(gray, edges, 50, 200);

    //hough_transforms(edges);
    std::vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 150);


    medianBlur(gray, gray2, 5);
    vector<Vec3f> circles;
    HoughCircles(gray2, circles, HOUGH_GRADIENT, 6, img.rows / 10, 150, 150, 0, 45);

    //Mat houghImg = Mat::zeros(img.size(), CV_8UC3);
    //Mat houghImg;
    //cvtColor(edges, houghImg, COLOR_GRAY2BGR);
    Mat houghImg = img.clone(); // Create a copy of the original image

    //Draw detected Circles
    for (size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(houghImg, center, radius, Scalar(0, 255, 0), 3, LINE_AA);
    }
    cout << "number of circles: " << circles.size();
    cout << "number of lines: " << lines.size();
    // Draw detected lines on image
    // Filter out horizontal and similar lines
    for (size_t i = 0; i < lines.size(); i++)
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