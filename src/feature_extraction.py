import cv2
import numpy as np
import os

LOWER = np.array([67, 0, 140])
UPPER = np.array([130, 255, 255])


def showImg(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def numberOfDigits(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, LOWER, UPPER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    return len(large_contours)


def calculate_perimeter(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, LOWER, UPPER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    total_perimeter = 0

    for contour in large_contours:
        total_perimeter += cv2.arcLength(contour, True)

    return total_perimeter


def calculate_area(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(imgHSV, LOWER, UPPER)
    result = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    total_area = 0

    for contour in large_contours:
        total_area += cv2.contourArea(contour)

    return total_area


def harrisCornerDetection(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([55, 0, 0])
    upper = np.array([126, 255, 180])
    mask = cv2.inRange(imgHSV, lower, upper)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.18)

    corners = cv2.dilate(corners, None)

    img[corners > 0.01 * corners.max()] = [0, 0, 255]

    showImg("Harris Corners", img)

    return corners


def houghLines(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([55, 0, 0])
    upper = np.array([126, 255, 180])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=28, minLineLength=10, maxLineGap=50
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # showImg("Hough Lines on Original Image", img)

    return lines


def houghCircles(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 136])
    upper = np.array([147, 45, 216])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=40,
        minRadius=10,
        maxRadius=50,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

    showImg("Hough Circles on Original Image", img)

    return circles


def empty(a):
    pass


def find_circle(img):
    total_shapes = 0
    total_circles = 0
    total_uknowns = 0
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(imgHSV, LOWER, UPPER)

    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(thresholded, 50, 130)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Colors for different shapes
    colors = {"Circle": (0, 255, 0), "Unknown": (0, 0, 255)}

    for i, c in enumerate(contours):
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


def get_features(image_path):
    features = []
    img = cv2.imread(image_path)
    features.append(numberOfDigits(img))
    # features.append(harrisCornerDetection(image_path))
    features.append(calculate_perimeter(img))
    circles, unkowns, total = find_circle(img)
    features.append(circles)
    features.append(unkowns)
    features.append(total)
    features.append(calculate_area(img))
    # features.append(houghLines(image_path))
    return features


def get_all_features(image_dir):
    x = []
    y = []
    for entry in os.listdir(image_dir):
        path = os.path.join(image_dir, entry)
        if os.path.isdir(path):
            features, labels = get_all_features(path)
            x += features
            y += labels
        else:
            x.append(get_features(path))
            label = image_dir.replace("segmented_data/", "")
            y.append(int(label))
    return x, y
