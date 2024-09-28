import cv2
import numpy as np

# 27 80 0 255 0 255

# Initialize the min and max HSV values
hmin, smin, vmin = 0, 57, 60  # You can adjust these based on your preference
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

    hmin = 119
    hmax = 165
    smin = 40
    smax = 178
    vmin = 68
    vmax = 255

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
        mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        cv2.imshow("Original Image", img_resized)
        cv2.imshow("Masked Image", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


image_path = "dataset/60/1727344430767872426.png"
process_image(image_path)
