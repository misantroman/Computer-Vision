# Soccer OffSide Detection
(see offside_detection.pdf)

The project is composed of 1 python folder and 3 cpp files, plus the image to process in this case "offside3" -inside the python folder we find the torchdetection.py file which is the model used to find the bounding boxes,
the image, and the bounding_boxes.txt, which is used to pass the bounding boxes to the offside_detection.cpp file to continue processing.
File functions.h contains the declaration of developed functions. File functions.cpp contains the definition of the functions.

In these previous mentioned files we can find all the functions used to experiment with the images. They are leaved there for possible evaluation and evidence of the work. 
In file offside_detection.cpp we find the main() function as well as the important functions used to arrived to the result.

The main pipeline of functions is the following: torchdetection.py --> offside_detection.cpp --> Main() --> imread() --> read_boxes()
--> draw_offside_lide() --> draw_goal_line() calculateDistance_toline() --> functions.cpp --> checkOffside()
All other functions were used for experimentation. In case user wants to try any other
function just has to uncomment the function from the main()
