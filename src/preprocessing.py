import os
import cv2
import numpy as np
import random


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
                if 0.7 <= axis_ratio <= 1.3:
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


def equalize_image_count(base_dir):
    dir_counts = {}
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            image_files = [f for f in os.listdir(dir_path)]
            dir_counts[dir_path] = len(image_files)

    min_count = min(dir_counts.values())

    for dir_path, count in dir_counts.items():
        if count > min_count:
            image_files = [f for f in os.listdir(dir_path)]
            files_to_remove = random.sample(image_files, count - min_count)
            for file_name in files_to_remove:
                os.remove(os.path.join(dir_path, file_name))
            print(
                f"Reduced {dir_path} to {min_count} images by removing {len(files_to_remove)} images."
            )
    print("All directories have been equalized to the minimum count:", min_count)


remove_bad_data("segmented_data")
equalize_image_count("segmented_data")
