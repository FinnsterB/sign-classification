
import glob
import os
import cv2
import numpy as np
import imutils

# Initialize the min and max HSV values
hmin, smin, vmin = 0, 50, 50  # You can adjust these based on your preference
hmax, smax, vmax = 179, 255, 255


def empty(a):
    pass


def color_controls():
    # Create a named window
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 640, 200)

    # Create trackbars for adjusting HSV values
    cv2.createTrackbar("Hue Min", "Trackbars", hmin, 179, empty)
    cv2.createTrackbar("Hue Max", "Trackbars", hmax, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", smin, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", smax, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", vmin, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", vmax, 255, empty)


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: The image '{image_path}' could not be loaded.")
        return
    img_resized = cv2.resize(img, (640, 480))
    imgHSV = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

 #   color_controls()

    for X in range(1):
        hmin = 0
        hmax = 179
        smin = 92
        smax = 255
        vmin = 0
        vmax = 255

        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, smax, vmax])

        mask = cv2.inRange(imgHSV, lower, upper)
        result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            mask_largest = np.zeros_like(img_resized)

            cv2.drawContours(mask_largest, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            img_largest_contour = cv2.bitwise_and(img_resized, mask_largest)

            cv2.imshow("Original Image", img_resized)
            cv2.imshow("Masked Image", result)
            cv2.imshow("Largest Contour Area", img_largest_contour)

            img_resized2 = img_resized

            # Load the input image and convert it to grayscale
            #image = cv2.imread(image_path)

            gray = cv2.cvtColor(img_largest_contour, cv2.COLOR_BGR2GRAY)

            # blur the image (to reduce false-positive detections) and then
            # perform edge detection
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # adjust this to get more accurate results
            edged = cv2.Canny(blurred, 50, 130)

            #cv2.imshow('Original', img)
            # cv2.imshow('Blurred',blurred)
            cv2.imshow('With contours', edged)

            # Find contours and hierarchy
            # Get contours and hierarchy
            edged, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total = 0

            # Loop over each contour
            for i, c in enumerate(edged):
                # Ignore small contours (noise)
                if cv2.contourArea(c) < 25:
                    continue

                # Use hierarchy to check if the contour is inside another contour
                if hierarchy[0][i][3] == -1:  # Check if the contour has no parent (outer contour)
                    # Draw the outer contour in one color
                    cv2.drawContours(img_resized2, [c], -1, (204, 0, 255), 2)
                else:
                    # Draw inner contours in a different color (nested contours)
                    cv2.drawContours(img_resized2, [c], -1, (0, 255, 0), 2)

                total += 1

            # Show the image with contours drawn
            cv2.imshow('Drawn', img_resized2)


            #      cv2.imshow('Drawn', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print("[INFO] found {} shapes".format(total))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#image_path = r'C:\Users\Blast\Desktop\Machine Learning\opecvtutorials\images\100sign.jpg'
#process_image(image_path)

if __name__ == "__main__":
    """ Test segmentation functions"""
    data_path = r'C:\Users\Blast\Desktop\Machine Learning'
    # data_path = r'C:\Users\Blast\Desktop\Machine Learning\Segmentation_results'

    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.jpg'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # loop over the image paths
    for filename in file_list:

        img = cv2.imread(filename)
        process_image(filename)

        k = cv2.waitKey(1000) & 0xFF
        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break
