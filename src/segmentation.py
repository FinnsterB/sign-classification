import cv2
import numpy as np

# Initialize the min and max HSV values
hmin, smin, vmin = 0, 65, 0
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
    # Load an image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: The image '{image_path}' could not be loaded.")
        return
    img_resized = cv2.resize(img, (640, 480))  # Resize to 640x480 pixels
    imgHSV = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Run the color control trackbars
    color_controls()

    while True:
        # Get trackbar positions
        hmin = cv2.getTrackbarPos("Hue Min", "Trackbars")
        hmax = cv2.getTrackbarPos("Hue Max", "Trackbars")
        smin = cv2.getTrackbarPos("Sat Min", "Trackbars")
        smax = cv2.getTrackbarPos("Sat Max", "Trackbars")
        vmin = cv2.getTrackbarPos("Val Min", "Trackbars")
        vmax = cv2.getTrackbarPos("Val Max", "Trackbars")

        # Define HSV lower and upper bounds based on trackbar positions
        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, smax, vmax])

        # Apply the mask to get only the colors within the defined HSV range
        mask = cv2.inRange(imgHSV, lower, upper)
        result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw filtered contours on
        img_filtered_contours = img_resized.copy()

        # Minimum and maximum area thresholds
        min_area = 500  # Minimum contour area
        max_area = 5000  # Maximum contour area

        # Filter contours based on area
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:  # Keep only contours within the area range
                x, y, w, h = cv2.boundingRect(contour)  # Get bounding rectangle
                cv2.rectangle(img_filtered_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

        # Display the original, masked, and filtered contour images
        cv2.imshow("Original Image", img_resized)
        cv2.imshow("Masked Image", result)
        cv2.imshow("Filtered Contours", img_filtered_contours)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all windows
    cv2.destroyAllWindows()

image_path = 'images/50/20240905_113847_098.jpg'
process_image(image_path)

