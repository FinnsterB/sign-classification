import sys

import cv2
import os
import glob
import imutils

def shape_counter(img_path):
    # Initialize counters for each shape
    total_shapes = 0
    total_circles = 0
    total_squares = 0
    total_triangles = 0
    total_rectangles = 0

    # Load the input image
   # img_path = r'C:\Users\Blast\Desktop\Machine Learning\opecvtutorials\images\100sign.jpg'
    img = cv2.imread(img_path)

    # Convert the image to grayscale and blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Perform edge detection
    edged = cv2.Canny(blurred, 50, 100)

    # Display original and edge-detected images
    #cv2.imshow('Original', img)

    # Find contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res_edged = imutils.resize(img, height=500)
    cv2.imshow('Original', res_edged)

    # Colors for different shapes
    colors = {
        'Triangle': (0, 255, 255),  # Yellow
        'Square': (255, 0, 0),      # Blue
        'Circle': (0, 255, 0),      # Green
        'Rectangle': (255, 0, 255), # Magenta
        'Unknown': (0, 0, 255)      # Red
    }

    # Loop over each contour
    for i, c in enumerate(contours):
        # Ignore small contours (noise)
        if cv2.contourArea(c) < 50:
            continue

        # Approximate the contour to reduce vertices
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.0004 * perimeter, True)

        # Initialize the shape type and contour color
        shape_type = "Unknown"
        color = colors['Unknown']

        # Classify shapes based on number of vertices
        if len(approx) == 3:
            shape_type = "Triangle"
            total_triangles += 1
            color = colors['Triangle']

        elif len(approx) > 4:
            # Check if the contour is a circle
            area = cv2.contourArea(c)
            radius = perimeter / (2 * 3.14159)
            circularity = area / (3.14159 * (radius ** 2))
            if 0.5 <= circularity <= 1.5:
                shape_type = "Circle"
                total_circles += 1
                color = colors['Circle']
        elif len(approx) == 4:
            # Check if the contour is a square or rectangle
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "Square"
                total_squares += 1
                color = colors['Square']
            else:
                shape_type = "Rectangle"
                total_rectangles += 1
                color = colors['Rectangle']


        # Draw the contour and label the shape
        cv2.drawContours(img, [c], -1, color, 2)
        x, y, w, h = cv2.boundingRect(c)  # Get the position to place the label
        cv2.putText(img, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Increment total shapes count
        total_shapes += 1

    # Show the image with contours drawn and labeled
    res_img = imutils.resize(img, height=500)
    cv2.imshow('Classified Shapes', res_img)

    # Print total shapes found

    print()
    print("[INFO] Selected Image: {}".format(img_path))
    print()
    print("[INFO] Total shapes: {}".format(total_shapes))
    print("[INFO] Circles: {}".format(total_circles))
    print("[INFO] Squares: {}".format(total_squares))
    print("[INFO] Triangles: {}".format(total_triangles))
    print("[INFO] Rectangles: {}".format(total_rectangles))

    k = cv2.waitKey(0) & 0xFF
    if k == ord("q") or k == 27:
        sys.exit()

    else:
       # cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    """ Test segmentation functions"""
    data_path = r'C:\Users\Blast\Desktop\Machine Learning\opecvtutorials\images'

    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # loop over the image paths
    for filename in file_list:

        img = cv2.imread(filename)
        shape_counter(filename)

        k = cv2.waitKey(0) & 0xFF
        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break
