import cv2
import numpy as np
import os

lower = np.array([50, 0, 130])
upper = np.array([164, 255, 245])


def create_digit_mask(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lower, upper)
    kernel = kernel = np.ones((4, 4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def numberOfDigits(img):
    mask = create_digit_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    return len(large_contours)


def calculate_perimeter(img):
    mask = create_digit_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    total_perimeter = 0

    for contour in large_contours:
        total_perimeter += cv2.arcLength(contour, True)

    return total_perimeter


def calculate_area(img):
    mask = create_digit_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    total_area = 0
    for contour in large_contours:
        total_area += cv2.contourArea(contour)

    return total_area


def find_circle(img):
    total_shapes = 0
    total_circles = 0
    total_uknowns = 0
    mask = create_digit_mask(img)

    result = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Colors for different shapes
    colors = {"Circle": (0, 255, 0), "Unknown": (0, 0, 255)}

    for _, c in enumerate(contours):
        if cv2.contourArea(c) < 50:
            continue
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.00004 * perimeter, True)

        shape_type = "Unknown"
        color = colors["Unknown"]

        if len(approx) >= 4:
            area = cv2.contourArea(c)
            radius = perimeter / (2 * 3.14159)
            circularity = area / (3.14159 * (radius**2))
            if 0.5 <= circularity <= 1.5:
                shape_type = "Circle"
                total_circles += 1
                color = colors["Circle"]
            else:
                shape_type = "Unknown"
                total_uknowns += 1
                color = colors["Unknown"]

        cv2.drawContours(result, [c], -1, color, 2)
        x, y, _, _ = cv2.boundingRect(c)
        cv2.putText(
            result, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        total_shapes += 1
    return total_circles, total_uknowns, total_shapes


def get_features(image):
    features = []
    features.append(numberOfDigits(image))
    features.append(calculate_perimeter(image))
    circles, unkowns, total = find_circle(image)
    features.append(circles)
    features.append(unkowns)
    features.append(total)
    features.append(calculate_area(image))
    return features


def get_all_features(image_dir, debug=False):
    x = []
    y = []
    for entry in os.listdir(image_dir):

        path = os.path.join(image_dir, entry)
        if os.path.isdir(path):
            features, labels = get_all_features(path, debug)
            x += features
            y += labels
        else:
            img = cv2.imread(path)
            x.append(get_features(img))
            label = os.path.relpath(image_dir, "segmented_data")
            y.append(int(label))
            if debug:
                if show_debug(path):
                    break
    return x, y


# Returns True if the user wants to skip to the next class
def show_debug(image_path):
    global lower
    global upper
    img = cv2.imread(image_path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Combine the original and masked images horizontally
    combined_img = cv2.hconcat([img, result])
    combined_img = cv2.resize(combined_img, (800, 400))

    # Create a trackbars window
    trackbar_window_name = "Adjust HSV Thresholds"
    cv2.namedWindow(trackbar_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(trackbar_window_name, 800, 200)

    def update_mask():
        mask = cv2.inRange(imgHSV, lower, upper)
        kernel = kernel = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        result = cv2.bitwise_and(img, img, mask=mask)
        combined_img = cv2.hconcat([img, result])
        combined_img = cv2.resize(combined_img, (800, 400))
        cv2.imshow("Original vs masked", combined_img)

    def update_HSV():
        lower[0] = cv2.getTrackbarPos("Hue Min", trackbar_window_name)
        lower[1] = cv2.getTrackbarPos("Sat Min", trackbar_window_name)
        lower[2] = cv2.getTrackbarPos("Val Min", trackbar_window_name)
        upper[0] = cv2.getTrackbarPos("Hue Max", trackbar_window_name)
        upper[1] = cv2.getTrackbarPos("Sat Max", trackbar_window_name)
        upper[2] = cv2.getTrackbarPos("Val Max", trackbar_window_name)
        update_mask()

    def empty(val):
        pass

    # Initialize trackbars with the values from the lower and upper arrays
    cv2.createTrackbar("Hue Min", trackbar_window_name, lower[0], 179, empty)
    cv2.createTrackbar("Hue Max", trackbar_window_name, upper[0], 179, empty)
    cv2.createTrackbar("Sat Min", trackbar_window_name, lower[1], 255, empty)
    cv2.createTrackbar("Sat Max", trackbar_window_name, upper[1], 255, empty)
    cv2.createTrackbar("Val Min", trackbar_window_name, lower[2], 255, empty)
    cv2.createTrackbar("Val Max", trackbar_window_name, upper[2], 255, empty)

    cv2.imshow("Original vs masked", combined_img)

    while True:
        update_HSV()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"Lower HSV values: {lower}")
            print(f"Upper HSV values: {upper}")
            cv2.destroyAllWindows()
            exit()
        elif key == ord("s"):
            cv2.destroyAllWindows()
            return True
        elif key == 13:  # Enter key to proceed
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    print(get_all_features("segmented_data", debug=True))
