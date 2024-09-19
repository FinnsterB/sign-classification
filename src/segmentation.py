import cv2
import numpy as np

# Initialize the min and max HSV values
hmin, smin, vmin = 0, 92, 0  # You can adjust these based on your preference
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
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: The image '{image_path}' could not be loaded.")
        return
    img_resized = cv2.resize(img, (640, 480))
    imgHSV = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    color_controls()

    while True:
        hmin = 0
        hmax = 179
        smin = 92
        smax = 255
        vmin = 0
        vmax = 255

        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, smax, vmax])

        mask = cv2.inRange(imgHSV, lower, upper)
        result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            mask_largest = np.zeros_like(img_resized)

            cv2.drawContours(mask_largest, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            img_largest_contour = cv2.bitwise_and(img_resized, mask_largest)

            cv2.imshow("Original Image", img_resized)
            cv2.imshow("Masked Image", result)
            cv2.imshow("Largest Contour Area", img_largest_contour)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

image_path = 'street_sign.jpg'
process_image(image_path)
