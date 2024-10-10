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


def calculate_black_white_areas(image, mask_circle):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the circular mask
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_circle)

    # Threshold the grayscale image (0: black, 255: white)
    _, thresh = cv2.threshold(masked_gray, 128, 255, cv2.THRESH_BINARY)

    # Calculate the number of black and white pixels inside the circle
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)

    # Return the black and white areas
    return black_pixels, white_pixels


def process_image(image_path):
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image {image_path} is corrupted or unreadable.")

    # Find the circular mask
    mask_circle, circle_info = find_circle_mask(image)

    # Calculate the black and white areas inside the circle
    black_pixels, white_pixels = calculate_black_white_areas(image, mask_circle)

    if white_pixels == 0 and black_pixels == 0:
        raise ValueError(f"Could not calculate black-white areas for image {image_path}. No area detected.")

    return black_pixels, white_pixels


def load_and_process_images(data_directory, max_retries=5, timeout=1):
    black_areas = []
    white_areas = []
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
                        black_pixels, white_pixels = future.result(timeout=timeout)

                        # Add the areas and label to the dataset
                        black_areas.append(black_pixels)
                        white_areas.append(white_pixels)
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

    return black_areas, white_areas, labels


def remove_outliers(black_areas, white_areas, labels, sensitivity=2.5):
    # Convert to numpy arrays
    black_areas = np.array(black_areas)
    white_areas = np.array(white_areas)
    labels = np.array(labels)

    # Calculate the IQR (Interquartile Range)
    Q1_black = np.percentile(black_areas, 25)
    Q3_black = np.percentile(black_areas, 75)
    IQR_black = Q3_black - Q1_black

    Q1_white = np.percentile(white_areas, 25)
    Q3_white = np.percentile(white_areas, 75)
    IQR_white = Q3_white - Q1_white

    # Define bounds for outliers using less sensitivity (2.5 times IQR)
    lower_bound_black = Q1_black - sensitivity * IQR_black
    upper_bound_black = Q3_black + sensitivity * IQR_black

    lower_bound_white = Q1_white - sensitivity * IQR_white
    upper_bound_white = Q3_white + sensitivity * IQR_white

    # Filter the black and white areas to remove outliers
    valid_indices = np.where((black_areas >= lower_bound_black) & (black_areas <= upper_bound_black) &
                             (white_areas >= lower_bound_white) & (white_areas <= upper_bound_white))

    filtered_black_areas = black_areas[valid_indices]
    filtered_white_areas = white_areas[valid_indices]
    filtered_labels = labels[valid_indices]

    return filtered_black_areas, filtered_white_areas, filtered_labels


def plot_results(black_areas, white_areas, labels):
    # Remove outliers before plotting (less sensitive outlier detection)
    black_areas, white_areas, labels = remove_outliers(black_areas, white_areas, labels)

    # Create subplots: one for scatter plot, one for box plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Convert labels to integers for plotting
    label_dict = {label: i for i, label in enumerate(set(labels))}
    numeric_labels = [label_dict[label] for label in labels]

    # Scatter plot (Black on X, White on Y)
    scatter = ax[0].scatter(black_areas, white_areas, c=numeric_labels, cmap='rainbow', s=100, edgecolor='k',
                            alpha=0.75)
    ax[0].set_title("Black vs. White Area Inside Circular Signs (Scatter Plot)")
    ax[0].set_xlabel("Black Area (pixels)")
    ax[0].set_ylabel("White Area (pixels)")
    legend1 = ax[0].legend(*scatter.legend_elements(), title="Labels")
    ax[0].add_artist(legend1)

    # Box plot of black areas
    unique_labels = sorted(set(labels))
    black_data_by_label = [[black_areas[i] for i in range(len(labels)) if labels[i] == label] for label in
                           unique_labels]
    white_data_by_label = [[white_areas[i] for i in range(len(labels)) if labels[i] == label] for label in
                           unique_labels]

    ax[1].boxplot(black_data_by_label, labels=unique_labels)
    ax[1].set_title("Distribution of Black Area by Label (Box Plot)")
    ax[1].set_xlabel("Labels")
    ax[1].set_ylabel("Black Area (pixels)")

    # Save the plot
    plt.tight_layout()
    plt.savefig("black_white_scatter_boxplot.png")
    print("Plot saved as 'black_white_scatter_boxplot.png'")

    # Show the plot
    plt.show()


def main():
    data_directory = r'C:\Users\Blast\Desktop\Machine Learning\segmented_dataset'  # Update this to your folder
    black_areas, white_areas, labels = load_and_process_images(data_directory)

    # Plot the results
    plot_results(black_areas, white_areas, labels)


if __name__ == "__main__":
    main()
