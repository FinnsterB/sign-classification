import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import concurrent.futures


def find_circle_mask(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for detecting red and purple hues in HSV space
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])

    # Create masks for red and purple
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    # Combine the red and purple masks
    mask = cv2.add(mask_red1, mask_red2)
    mask = cv2.add(mask, mask_purple)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep the largest one (assuming it's the circle)
    if len(contours) == 0:
        raise ValueError("No red or purple circular outline found.")

    contour = max(contours, key=cv2.contourArea)

    # Get the minimum enclosing circle for the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)

    # Create a circular mask based on the detected circle
    mask_circle = np.zeros_like(image[:, :, 0])
    cv2.circle(mask_circle, (int(x), int(y)), int(radius), 255, -1)

    return mask_circle, (x, y, radius)


def calculate_black_white_ratio(image, mask_circle):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the circular mask
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_circle)

    # Threshold the grayscale image (0: black, 255: white)
    _, thresh = cv2.threshold(masked_gray, 128, 255, cv2.THRESH_BINARY)

    # Calculate the number of black and white pixels inside the circle
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)

    # Compute the black-to-white ratio
    if white_pixels == 0:
        return None  # Avoid division by zero
    ratio = black_pixels / white_pixels

    return ratio


def process_image(image_path):
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image {image_path} is corrupted or unreadable.")

    # Find the circular mask
    mask_circle, circle_info = find_circle_mask(image)

    # Calculate the black-to-white ratio inside the circle
    ratio = calculate_black_white_ratio(image, mask_circle)

    if ratio is None:
        raise ValueError(f"Could not calculate black-white ratio for image {image_path}. No white area detected.")

    return ratio


def load_and_process_images(data_directory, max_retries=5, timeout=1):
    ratios = []
    labels = []

    # Get all subdirectories (these are the labels)
    data_path = Path(data_directory)
    subfolders = [f for f in data_path.iterdir() if f.is_dir()]

    # Set up a ThreadPoolExecutor for managing timeouts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for subfolder in subfolders:
            label = subfolder.name  # The label will be the name of the subfolder

            # Get all .png files in the subfolder
            for image_path in subfolder.glob("*.png"):
                success = False
                attempts = 0

                print(f"[INFO] Processing image: {image_path}")

                while attempts < max_retries:
                    try:
                        # Try processing the image with timeout
                        future = executor.submit(process_image, image_path)
                        ratio = future.result(timeout=timeout)

                        # Add the ratio and label to the dataset
                        ratios.append(ratio)
                        labels.append(label)

                        success = True
                        break  # Exit retry loop if successful
                    except concurrent.futures.TimeoutError:
                        print(f"[WARNING] Image {image_path} took too long to process. Skipping...")
                        break
                    except Exception as e:
                        attempts += 1
                        print(f"[WARNING] Attempt {attempts} failed for image {image_path}: {e}")

                if not success:
                    # If all attempts failed, log the error and move to the next image
                    print(f"[ERROR] Failed to process image {image_path} after {max_retries} attempts. Skipping.")

    return ratios, labels


def remove_outliers(ratios, labels):
    # Convert to numpy arrays
    ratios = np.array(ratios)
    labels = np.array(labels)

    # Calculate the IQR (Interquartile Range)
    Q1 = np.percentile(ratios, 25)
    Q3 = np.percentile(ratios, 75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the ratios and labels to remove outliers
    valid_indices = np.where((ratios >= lower_bound) & (ratios <= upper_bound))
    filtered_ratios = ratios[valid_indices]
    filtered_labels = labels[valid_indices]

    return filtered_ratios, filtered_labels


def plot_results(ratios, labels):
    # Remove outliers before plotting
    ratios, labels = remove_outliers(ratios, labels)

    # Create subplots: one for scatter plot, one for box plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Convert labels to integers for plotting
    label_dict = {label: i for i, label in enumerate(set(labels))}
    numeric_labels = [label_dict[label] for label in labels]

    # Scatter plot of black-to-white ratios
    ax[0].scatter(numeric_labels, ratios, c=numeric_labels, cmap='rainbow', s=100, edgecolor='k', alpha=0.75)
    ax[0].set_title("Black-to-White Area Ratio Inside Circular Signs (Scatter)")
    ax[0].set_xlabel("Labels")
    ax[0].set_ylabel("Black-to-White Ratio")
    ax[0].set_xticks(ticks=range(len(label_dict)))
    ax[0].set_xticklabels(label_dict.keys(), rotation=45)

    # Box plot of black-to-white ratios
    unique_labels = sorted(set(labels))
    data_by_label = [[ratios[i] for i in range(len(labels)) if labels[i] == label] for label in unique_labels]

    ax[1].boxplot(data_by_label, labels=unique_labels)
    ax[1].set_title("Black-to-White Area Ratio Inside Circular Signs (Box Plot)")
    ax[1].set_xlabel("Labels")
    ax[1].set_ylabel("Black-to-White Ratio")

    # Save the plot
    plt.tight_layout()
    plt.savefig("black_white_ratio_plots.png")
    print("Plot saved as 'black_white_ratio_plots.png'")

    # Show the plot
    plt.show()


def main():
    data_directory = r'C:\Users\Blast\Desktop\Machine Learning\segmented_dataset'  # Update this to your folder
    ratios, labels = load_and_process_images(data_directory)

    # Plot the results
    plot_results(ratios, labels)


if __name__ == "__main__":
    main()
