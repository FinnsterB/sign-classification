import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# Initialize the min and max HSV values
hmin, smin, vmin = 55, 0, 0  # You can adjust these based on your preference
hmax, smax, vmax = 126, 255, 180


def getContourProximityRatio(image_path, reference_width):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)

        x, y, width, height = cv2.boundingRect(biggest_contour)

        ratio = width / reference_width

        print(f"The ratio of proximity (width/reference width) is: {ratio:.2f}")

        cv2.drawContours(img, [biggest_contour], -1, (0, 255, 0), 3)
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)

        cv2.imshow("Biggest Contour with Bounding Box", img)
    else:
        print("No contours found!")

    # Step 9: Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showImg(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convexHull(image_path):
    img = cv2.imread(image_path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([55, 0, 0])
    upper = np.array([126, 255, 180])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    convex_hull_points = []

    for contour in large_contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(result, [hull], -1, (0, 255, 0), 1)
        convex_hull_points.append(hull)

    num_hull_contours = len(convex_hull_points)
    print(f"Number of convex hull contours detected: {num_hull_contours}")

    showImg("Convex Hull", result)

    return convex_hull_points

def calculate_perimeter(image_path):
    img = cv2.imread(image_path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([55, 0, 0])
    upper = np.array([126, 255, 180])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    total_perimeter = 0

    for contour in large_contours:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 1)
        total_perimeter += cv2.arcLength(contour, True)

    num_contours = len(large_contours)
    print(f"Number of contours detected: {num_contours}")
    print(f"Total perimeter of all contours: {total_perimeter}")

    showImg("Contours", result)

    return total_perimeter

def harrisCornerDetection(image_path):
    img = cv2.imread(image_path)
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



def houghLines(image_path):
    img = cv2.imread(image_path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([55, 0, 0])
    upper = np.array([126, 255, 180])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=28, minLineLength=10, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    

    # showImg("Hough Lines on Original Image", img)

    return lines

def houghCircles(image_path):
    img = cv2.imread(image_path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 136])
    upper = np.array([147, 45, 216])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=40, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

    # showImg("Hough Circles on Original Image", img)

    return circles

def empty(a):
    pass

def color_controls():
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 640, 200)

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
    
    color_controls()

    while True:
        hmin = cv2.getTrackbarPos("Hue Min", "Trackbars")
        hmax = cv2.getTrackbarPos("Hue Max", "Trackbars")
        smin = cv2.getTrackbarPos("Sat Min", "Trackbars")
        smax = cv2.getTrackbarPos("Sat Max", "Trackbars")
        vmin = cv2.getTrackbarPos("Val Min", "Trackbars")
        vmax = cv2.getTrackbarPos("Val Max", "Trackbars")

        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, smax, vmax])

        mask = cv2.inRange(imgHSV, lower, upper)
        result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("Masked Image", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def find_circle(img_path):
    total_shapes = 0
    total_circles = 0
    total_uknowns = 0
    img = cv2.imread(img_path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([67, 0, 140])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 50, 130)

    res_edged = imutils.resize(img, height=800)
    # cv2.imshow('With contours', res_edged)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Colors for different shapes
    colors = {
        'Circle': (0, 255, 0),
        'Unknown': (0, 0, 255)
    }

    for i, c in enumerate(contours):
        if cv2.contourArea(c) < 50:
            continue
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.00004 * perimeter, True)

        shape_type = "Unknown"
        color = colors['Unknown']

        if len(approx) > 4:
            area = cv2.contourArea(c)
            radius = perimeter / (2 * 3.14159)
            circularity = area / (3.14159 * (radius ** 2))
            if 0.5 <= circularity <= 1.5:
                shape_type = "Circle"
                total_circles += 1
                color = colors['Circle']
            else:
                shape_type = "Unknown"
                total_uknowns += 1
                color = colors['Unknown']

        cv2.drawContours(result, [c], -1, color, 2)
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(result, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        total_shapes += 1

    res_img = imutils.resize(result, height=800)
    cv2.imshow('Classified Shapes', res_img)

    print("[INFO] Total shapes: {}".format(total_shapes))
    print("[INFO] Circles: {}".format(total_circles / 2))
    print("[INFO] Uknowns: {}".format(total_uknowns / 2))

    k = cv2.waitKey(0) & 0xFF
    if k == ord("q") or k == 27:
        cv2.destroyAllWindows()

    array_circles_and_unknown = [total_circles, total_uknowns]
    return array_circles_and_unknown



#There has to be images from the segmented data folder, you can get them through running the segmentation.py.
image_path = 'segmented_data/120/1727344223625093954.png'
find_circle(image_path)
