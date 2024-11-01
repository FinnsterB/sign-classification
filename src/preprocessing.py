import os
import cv2
import numpy as np


def contains_red_circle(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            if len(contour) >= 5:  # Check contour is circular
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis, minor_axis = axes

                # Allow ovals by checking if the major and minor axes are reasonably close
                axis_ratio = 0
                if minor_axis != 0:
                    axis_ratio = major_axis / minor_axis

                # If the shape is oval it should also pass as a circle
                if 0.5 <= axis_ratio <= 2.0:
                    return True

    return False


def remove_bad_data(segmented_data_dir):
    for entry in os.listdir(segmented_data_dir):
        path = os.path.join(segmented_data_dir, entry)
        if os.path.isdir(path):
            remove_bad_data(path)
        else:
            if not contains_red_circle(path):
                os.remove(path)


remove_bad_data("segmented_data")
