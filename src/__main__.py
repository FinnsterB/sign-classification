import preprocessing
import segmentation
import feature_extraction
import classification

import os
import glob
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    """ Test segmentation functions"""
    data_path = r'/home/finn/Documents/MOB-ROB-minor/MACH_LEARN/miniproject_dataset'

    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.jpg'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # loop over the image paths
    for filename in file_list:
        
        # load image and blur a bit
        img = cv.imread(filename)
        img = cv.resize(img, (400, 300))
        img = cv.blur(img,(3,3))

        # mask background 
        mask = segmentation.maskGrayBG(img)
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # show result and wait a bit        
        cv.imshow("Masked image", mask)
        k = cv.waitKey() & 0xFF

        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break 