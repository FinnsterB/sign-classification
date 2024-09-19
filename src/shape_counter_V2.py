import cv2

total = 0

img_path = r'C:\Users\Blast\Desktop\Machine Learning\opecvtutorials\images\100sign.jpg'
img = cv2.imread(img_path)

# Find contours and hierarchy
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Get contours and hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop over each contour
for i, c in enumerate(contours):
    # Ignore small contours (noise)
    if cv2.contourArea(c) < 25:
        continue

    # Use hierarchy to check if the contour is inside another contour
    if hierarchy[0][i][3] == -1:  # Check if the contour has no parent (outer contour)
        # Draw the outer contour in one color
        cv2.drawContours(img, [c], -1, (204, 0, 255), 2)
    else:
        # Draw inner contours in a different color (nested contours)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

    total += 1

# Show the image with contours drawn
cv2.imshow('Drawn', img)


cv2.waitKey(0)


# Wait for 'q' to exit
if cv2.waitKey(1) & 0xFF == ord('q'):

    cv2.destroyAllWindows()
