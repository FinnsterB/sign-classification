import cv2
import numpy as np
import os
import sys
import shutil

data_dir = "dataset"
segmented_data_dir = "segmented_data"

# Red HSV values
hmin = 0
hmax = 179
smin = 92
smax = 255
vmin = 0
vmax = 255


def remove_directory(directory_name):
    try:
        shutil.rmtree(directory_name)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: The image '{image_path}' could not be loaded.")
        return
    return img


def save_image(image_path, image):
    new_path = image_path.replace(data_dir, segmented_data_dir)
    new_dir = os.path.dirname(new_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    cv2.imwrite(new_path, image)
    print(new_path)


def create_color_mask(image):
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])

    mask = cv2.inRange(img_HSV, lower, upper)
    return mask


def create_largest_contour_mask(image, color_mask):
    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # IF no contours are found return empty mask
    if len(contours) == 0:
        print("WARNING: No contours found")
        return np.zeros_like(image)

    largest_contour = max(contours, key=cv2.contourArea)

    contour_mask = np.zeros_like(image)

    cv2.drawContours(
        contour_mask,
        [largest_contour],
        -1,
        (255, 255, 255),
        thickness=cv2.FILLED,
    )
    return contour_mask


def process_image(image_path):
    img = load_image(image_path)
    img_resized = cv2.resize(img, (640, 480))
    mask = create_color_mask(img_resized)

    largest_contour_mask = create_largest_contour_mask(img_resized, mask)

    img_largest_contour = cv2.bitwise_and(img_resized, largest_contour_mask)
    return img_largest_contour


def segment_images(image_dir):
    for entry in os.listdir(image_dir):
        path = os.path.join(image_dir, entry)
        if os.path.isdir(path):
            segment_images(path)
        else:
            if os.path.basename(path) != ".gitkeep":
                processed_image = process_image(path)
                save_image(path, processed_image)


if __name__ == "__main__":
    remove_directory(segmented_data_dir)
    segment_images(data_dir)
