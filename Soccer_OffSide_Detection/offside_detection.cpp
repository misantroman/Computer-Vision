///Victor Velazquez 2043179

#include <opencv2/opencv.hpp>
#include<iostream>
#include "functions.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


using namespace std;
using namespace cv;

//Declaration of functions

void read_boxes(Mat img);

std::vector<float> extractFloats(const std::string& line) {
    std::vector<float> floats;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ' ')) {
        // Remove unnecessary characters
        value.erase(std::remove(value.begin(), value.end(), '['), value.end());
        value.erase(std::remove(value.begin(), value.end(), ']'), value.end());

        // Convert to float and add to the vector
        std::stringstream converter(value);
        float variable;
        if (converter >> variable)
            floats.push_back(variable);
    }
    return floats;
}

cv::Scalar calculateAverageColor(const cv::Mat& image, const cv::Rect& roi) {
    cv::Mat roiImage = image(roi);  // Extract the ROI
    cv::Scalar averageColor = cv::mean(roiImage);  // Calculate the average color
    return averageColor;
}

double calculateDistance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

double calculateDistance_toline(cv::Point2f pt1, cv::Point2f pt2, cv::Point2f point) {
    //std::cout << "Enter calculate distance to line: " << std::endl;
    //pt1 is 1st point of reference line
    //pt2 is 2nd point of reference line
    //point is players point to evaluate distance to line
    double dx = pt2.x - pt1.x;
    double dy = pt2.y - pt1.y;
    double numerator = std::abs(dy * point.x - dx * point.y + pt2.x * pt1.y - pt2.y * pt1.x);
    double denominator = std::sqrt(dx * dx + dy * dy);
    return numerator / denominator;
}

void draw_goal_line(Mat img) {
    // Define the line parameters
    
    cv::Point pt1_l(-905, 497);
    cv::Point pt2_l(1032, -1);
    cv::line(img, pt1_l, pt2_l, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    

}

void draw_offside_lide(float x_gk, float y_gk, const std::vector<std::vector<float>>& boxes, Mat img, const std::vector<std::string>& stringList) {
    std::cout << "Processing Image..." << std::endl;
    std::cout << ". " << std::endl;
    std::cout << ". " << std::endl;
    std::cout << "Identifying players..." << std::endl;
    std::cout << ". " << std::endl;
    std::cout << ". " << std::endl;
    std::cout << "Making calculations..." << std::endl;
    std::cout << ". " << std::endl;
    std::cout << ". " << std::endl;
    std::cout << "Drawing OffSide Line..." << std::endl;
    std::cout << ". " << std::endl;
    std::cout << ". " << std::endl;
    std::cout << "... " << std::endl;
    //points of goalline
    cv::Point pt1_goal(-905, 497);
    cv::Point pt2_goal(1032, -1);

    // Define the point through which the parallel line should pass this is for goalkeeper
    cv::Point2f target_point(x_gk, y_gk); //in case we want to have line reference in goakkeeper position
    cv::Point2f target_point2(461.426, 247.116);

    // Calculate the slope of the original line
    float slope = static_cast<float>(pt2_goal.y - pt1_goal.y) / (pt2_goal.x - pt1_goal.x);
    float target_slope = (pt2_goal.y - pt1_goal.y) / (pt2_goal.x - pt1_goal.x);

    // Calculate the y-intercept of the parallel line
    float y_intercept = target_point.y - slope * target_point.x;

    // Generate the x-coordinates for the parallel line
    std::vector<double> x_coords;
    for (double x = pt1_goal.x; x <= pt2_goal.x; x += 1.0) {
        x_coords.push_back(x);
    }

    // Calculate the corresponding y-coordinates for the parallel line
    std::vector<double> y_coords;
    for (const auto& x : x_coords) {
        double y = slope * x + y_intercept;
        y_coords.push_back(y);
    }

    
    cv::Point first_point;
    cv::Point last_point;
    //std::cout << "Draw goalkeeperline " << std::endl;
    for (int i = 0; i < x_coords.size() - 1; ++i) {
        cv::Point parallel_pt11(x_coords[i], y_coords[i]);
        cv::Point parallel_pt22(x_coords[i + 1], y_coords[i + 1]);
        //cv::line(img, parallel_pt11, parallel_pt22, cv::Scalar(0, 0, 255), 2);
        // Store the first and last points
        if (i == 0) {
            first_point = parallel_pt11;
        }
        if (i == x_coords.size() - 2) {
            last_point = parallel_pt22;
        }
    }

    draw_goal_line(img);

    // Find the point with the minimum distance to the line segment
    double min_distance = 1000000;
    cv::Point2f closest_point;
    cv::Point2f point_c;
    int id_close = 0;
    int id_closest_player = 0;

    for(const auto & box : boxes) {
        
        // Access the box variables
        float x1 = box[0];
        float y1 = box[1];
        float x2 = box[2];  
        float y2 = box[3];
        y2 = y2 - ((y2 - y1) / 2.8); //focus on the tshirt color
        y1 = y1 + ((y2 - y1) / 6.3); //avoid the head

        if (id_close == 0) {
            //std::cout << "entered distance iterations" << x1 << y1 << first_point << last_point << std::endl;
        }

        cv::Point2f point_c(x1, y1);
        //cv::Point2f pt2(1137, 188);
        double distance = calculateDistance_toline(pt1_goal, pt2_goal, point_c);
        //std::cout << "Distance:  " <<  distance << std::endl;
        if (distance < min_distance && id_close != 1) {
            min_distance = distance;
            closest_point = point_c;
            //std::cout << "new id of closest player: " << id_close << std::endl;
            id_closest_player = id_close;

        }
        id_close += 1;
    }

    // Calculate the y-intercept of the parallel line in closest point
    float y_intercept_2 = closest_point.y - slope * closest_point.x;
    //std::cout << "coordinates closest player: " << closest_point << std::endl;
    std::cout << "Distance to closest player player: " << min_distance << std::endl;
    std::cout << "ID of closest player player: " << id_closest_player << std::endl;

    // Generate the x-coordinates for the parallel line
    std::vector<double> x_coords_2;
    for (double x = pt1_goal.x; x <= pt2_goal.x; x += 1.0) {
        x_coords_2.push_back(x);
    }

    // Calculate the corresponding y-coordinates for the parallel line
    std::vector<double> y_coords_2;
    for (const auto& x : x_coords_2) {
        double y = slope * x + y_intercept_2;
        y_coords_2.push_back(y);
    }

   
    // Draw the parallel line
    
    //std::cout << "Draw OFFSIDE LINE! " << std::endl;

    for (int i = 0; i < x_coords_2.size() - 1; ++i) {
        cv::Point parallel_pt1(x_coords_2[i], y_coords_2[i]);
        cv::Point parallel_pt2(x_coords_2[i + 1], y_coords_2[i + 1]);
        cv::line(img, parallel_pt1, parallel_pt2, cv::Scalar(0, 0, 255), 2);
    }
    
    //final output
    checkOffside(stringList, id_closest_player);

}


int main() {
    // Load image
    //string path = "C:Users\velaz\OneDrive\Documents\Control Systems Master\4th Semester\Computer Vision\Lab3\road1.png";
    //C:\Users\velaz\source\repos\opencvtest\opencvtest
    string path = "offside3.jpg"; 


    //segmentation_kmeans(path); //works good for offside3

    //person_detection(path);
    // 
    //person_detection_lowerbody(path);
    Mat image = imread(path);

    // Check if image is loaded successfully
    if (image.empty()) {
        std::cout << "Failed to load image!" << std::endl;
        return -1;
    }

    read_boxes(image);
    //detect line and get the pendiente
    //Line_Detection(image);
    //Line_Detection_Track(image);  //threshold of 230 or plus works for image3


    //down of here is another segmentation from ranges make it a function
    //segmentation_with_ranges(image);
    
    //segmentation with threshold
    //segmentation(image);



    return 0;
}


// Definition of functions

void read_boxes(Mat img) {
    std::ifstream input_file("bounding_boxes.txt");
    std::string line;
    std::vector<std::vector<float>> boxes;

    while (std::getline(input_file, line)) {
        std::vector<float> box = extractFloats(line);
        boxes.push_back(box);
    }

    int blue = 0, red = 0, goalkepeerreferee = 0, id_player = 0;

    std::string red_name = "red";
    std::string blue_name = "blue";
    std::string extra_name = "extra";

    int goal_keeper_id = -1;
    int id_closest_player = 0;
    float ref_x1 = 0.0f;
    float ref_y1 = 0.0f;
    std::string str;
    // Initialize an empty vector of strings
    std::vector<std::string> colors_list;

    // Iterate over the boxes
    for (const auto& box : boxes) {
        
        // Access the box variables
        float x1 = box[0];
        float y1 = box[1];
        float x2 = box[2];
        float y2 = box[3];
        y2 = y2 - ((y2 - y1) / 2.8); //focus on the tshirt color
        y1 = y1 + ((y2 - y1) / 6.3); //avoid the head
        // Print the box variables
        //std::cout << "Box: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << std::endl;

        // Draw the rectangle on the image
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

        // Calculate the average color within the bounding box
        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Scalar averageColor = calculateAverageColor(img, roi);

        // Calculate the square root of the sum of squares of BGR components of the average color
        double sumOfSquares = averageColor[0] * averageColor[0] + averageColor[1] * averageColor[1] + averageColor[2] * averageColor[2];
        double sqrtSumOfSquares = std::sqrt(sumOfSquares);
        
        //variables to write
        cv::Point location(x1, y1);  // Coordinates of the text location
        cv::Point location2(x2, y2);  // Coordinates of the text location
        cv::Scalar color(0, 255, 0);  // Text color (BGR format)
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;  // Font type
        double fontScale = 1.0;  // Font scale
        int thickness = 1;  // Text thickness

        if (sqrtSumOfSquares < 196) {
            // Value is less than 200
            //std::cout << "Value is less than 200 BLUE" << std::endl;
            //std::cout << "id player: " << id_player << std::endl;
            std::string str = "Blue";
            colors_list.push_back(str);

            blue += 1;
            std::string numberString = std::to_string(id_player);
            cv::putText(img, blue_name, location, fontFace, fontScale, color, thickness);
            cv::putText(img, numberString, location2, fontFace, fontScale, color, thickness);

        }
        else if (sqrtSumOfSquares >= 196 && sqrtSumOfSquares < 256) {
            // Value is between 200 and 229
            //std::cout << "Value is between 200 and 229 RED" << std::endl;
            //std::cout << "id player: " << id_player << std::endl;
            std::string str = "Red";
            colors_list.push_back(str);


            red += 1;
            std::string numberString = std::to_string(id_player);
            cv::putText(img, red_name, location, fontFace, fontScale, color, thickness);
            cv::putText(img, numberString, location2, fontFace, fontScale, color, thickness);

        }
        else {
            // Value is 230 or greater
            goalkepeerreferee += 1;
            std::string numberString = std::to_string(id_player);
            //std::cout << "Value is 230 or greater GOAL KEEPER/REFEREE" << std::endl;
            //std::cout << "id player: " << id_player << std::endl;
            std::string str = "Extra";
            colors_list.push_back(str);

            cv::putText(img, extra_name, location, fontFace, fontScale, color, thickness);
            cv::putText(img, numberString, location2, fontFace, fontScale, color, thickness);
            goal_keeper_id = id_player ;
            ref_x1 = x1;
            ref_y1 = y1;

        }
        

        id_player += 1;
        //std::cout << "goal keeper id: " << goal_keeper_id << std::endl;
        // Print the square root of the sum of squares
        //std::cout << "Square Root of Sum of Squares: " << sqrtSumOfSquares << std::endl;

        // Print the average color
        //std::cout << "Average Color: B=" << averageColor[0] << ", G=" << averageColor[1] << ", R=" << averageColor[2] << std::endl;
        
    }
    std::cout << "-------------Start Soccer OffSide Detection Algorithm--------------" << std::endl;
    //draw the line  if we have as reference the goal keeper
    /*
    int i = 0;
    double min_distance = 100000;
    for (const auto& box : boxes) {
        
        if (i == goal_keeper_id) {
            //std::cout << "Entered goal keeper id: " << i << std::endl;
            i += 1;
            continue;
            
        }
        
        else {
            float x2 = box[0];
            float y2 = box[1];

            double distance = calculateDistance(ref_x1, ref_y1, x2, y2);
            if (min_distance > distance) {
                min_distance = distance;
                id_closest_player = i;
            }
            //std::cout << "id's checked " << i << std::endl;
            i += 1;
        }


        
    }
    */
    //std::cout << "ID of the closest player: " << id_closest_player << std::endl;
    //std::cout << "Distance to the closest player: " << min_distance << std::endl;
    
    float x_gk = ref_x1;
    float y_gk = ref_y1;

    draw_offside_lide(x_gk, y_gk, boxes, img, colors_list);
    
    //std::cout << "blue players: " << blue << "red players: " << red << std::endl;

    std::cout << "-------------------End of program------------------" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;


    input_file.close();
    imshow("Image boxes", img);
    waitKey(0);
}


