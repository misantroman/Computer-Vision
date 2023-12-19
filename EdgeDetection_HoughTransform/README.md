# Edge Detection and Hough Transform

his project aims to explore the application of edge detection and Hough transform in image processing, examples of segmenting of street lanes and round street signs
from provided images. 
This work is done with OPENCV library.

## Methodology

1.- Load the input image: The input image was loaded from the file path using the imread() function from OpenCV. The program checks if the image is loaded successfully, and if not, an error message is displayed.
2.- Convert the image to grayscale: The loaded color image was converted to grayscale using the cvtColor() function from OpenCV.
3.- Apply Canny edge detection: The Canny edge detection algorithm was applied to the grayscale image using the Canny() function from OpenCV. This generates an edge map of the image, where the edges are represented as white pixels and the background is represented as black pixels.
4.- Detect circles using Hough transform: The HoughCircles() function from OpenCV was used to detect the circular street signs. This function applies the Hough transform to the grayscale image, which searches for circular shapes in the image. The detected circles were stored in a vector of Vec3f data type.
5.- Detect lines using Hough transform: The HoughLines() function from OpenCV was used to detect the street lanes. This function applies the Hough transform to the edge map generated in step 3, which searches for lines in the image. The detected lines were stored in a vector of Vec2f data type.
6.- Filter out horizontal and similar lines: Horizontal lines and similar lines were filtered out using the detected lines from step 5. Horizontal lines were ignored by checking the angle of the line with respect to the horizontal axis. Similar lines were ignored by comparing the distance and angle between the current line and previously detected lines.
7.- Draw detected circles and lines on the output image: A copy of the original input image was made and the detected circles and filtered lines were drawn on it using
the circle() and line() functions from OpenCV.
8.- Display the output image: The original input image, grayscale image, edge map, and the final output image with detected circles and lines were displayed using the imshow() function from OpenCV. The program waits for a key event before closing
the windows.

