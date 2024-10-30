import cv2
import numpy as np
import os
import shutil

data_dir = "dataset"
segmented_data_dir = "segmented_data"

# Green HSV values
hmin = 27
hmax = 80
smin = 20
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


def find_largest_red_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def crop_image(image, largest_contour):
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y : y + h, x : x + w]
    return cropped_image


def process_image(image_path):
    img = load_image(image_path)
    adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
    img_resized = cv2.resize(adjusted, (640, 480))
    mask = create_color_mask(img_resized)

    mask = cv2.bitwise_not(mask)
    kernel = kernel = np.ones((50, 50), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_without_background = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    largest_contour = find_largest_red_contour(image_without_background)
    # If no contour is found return a black image
    if largest_contour is None:
        height, width, channels = img_resized.shape
        return np.zeros((height, width, channels), dtype=np.uint8)
    cropped_result = crop_image(image_without_background, largest_contour)
    cropped_result = cv2.resize(cropped_result, (250, 250))
    return cropped_result


def is_image_blank(image):
    return np.all(image == image[0, 0])


def segment_images(image_dir):
    for entry in os.listdir(image_dir):
        path = os.path.join(image_dir, entry)
        if os.path.isdir(path):
            segment_images(path)
        else:
            if os.path.basename(path) != ".gitkeep":
                processed_image = process_image(path)
                if not is_image_blank(processed_image):
                    save_image(path, processed_image)


if __name__ == "__main__":
    remove_directory(segmented_data_dir)
    segment_images(data_dir)
