import torch
import torchvision
import cv2
import numpy as np


# Load the pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
print(model)
# Load the COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
     'cat', 'dog', 'horse', 'sheep', 'cow','book', 'clock','toothbrush'
]
 #C:\Users\velaz\source\repos\opencvtest\opencvtest\python\torchdetection.py
#path = "C:\Users\velaz\source\repos\opencvtest\opencvtest\offside3.jpg"
path = "offside3.jpg"
# Function to perform person detection
def detect_person(image_path):
    print("enter detection")
    # Load the input image
    image = cv2.imread(image_path)
    image_tensor = torchvision.transforms.ToTensor()(image)  #transform image to tensor

    # Run the image through the model
    with torch.no_grad():
        prediction = model([image_tensor])

    # Get the predicted bounding boxes, labels, and scores
    boxes = prediction[0]['boxes'].detach().numpy()
    labels = prediction[0]['labels'].detach().numpy()
    scores = prediction[0]['scores'].detach().numpy()

    # Filter predictions to keep only person detections
    person_indices = labels == 1
    person_boxes = boxes[person_indices]
    person_scores = scores[person_indices]

    num_boxes = len(person_boxes)
    print("Number of boxes detected:", num_boxes)
    output_file = open("bounding_boxes.txt", "w")

    # Display the detected persons
    for box, score in zip(person_boxes, person_scores):
        if score > 0.7:  # Adjust the threshold as needed
            
            # save the bounding boxes
            output_file.write(f" {box}\n")

            x, y, w, h = box
            x1, y1 = int(x), int(y)
            point =(x,y)
            print("boxes corner")
            print(point)
            x2, y2 = int(x + w/8.5), int(y + h/3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Close the file
    output_file.close()

    # Show the image with detections
    #show offside line
    slope = -0.5596284289276808
    point = (461.426, 238.85843)
    #rho = 573, theta = 1.21475
    theta1 = -59
    rho1 = 1.5708
    a = np.cos(theta1)
    b = np.sin(theta1)
    x0 = a * rho1
    y0 = b * rho1

    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    pt1 = (-738, 886)
    pt2 = (1137, 188)
    #cv2.line(image, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    #get the correct offside line!!

    # Define the points on the original line
    pt1 = (-738, 886)
    pt2 = (1137, 188)

    # Define the point through which the parallel line should pass
    target_point = (461.426, 238.85843)

    # Calculate the slope of the original line
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    # Calculate the y-intercept of the parallel line
    y_intercept = target_point[1] - slope * target_point[0]

    # Generate the x-coordinates for the parallel line
    x_coords = np.linspace(pt1[0], pt2[0], num=100)

    # Calculate the corresponding y-coordinates for the parallel line
    y_coords = slope * x_coords + y_intercept

    # Draw the parallel line
    for i in range(len(x_coords)-1):
        pt1 = (int(x_coords[i]), int(y_coords[i]))
        pt2 = (int(x_coords[i+1]), int(y_coords[i+1]))
        #cv2.line(image, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.imshow('Person Detection', image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Call the detect_person function

detect_person(path)
