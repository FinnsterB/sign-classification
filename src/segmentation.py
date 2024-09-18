import os
import glob
import cv2 as cv
import numpy as np

def maskGrayBG(img):
    """ Asssuming the background is gray, segment the image and return a
        BW image with foreground (white) and background (black)
    """ 
    # Change image color space
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Note that 0≤V≤1, 0≤S≤1, 0≤H≤360 and if H<0 then H←H+360
    # 8-bit images: V←255V,S←255S,H←H/2(to fit to 0 to 255)
    # see https://docs.opencv.org/4.5.3/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv

    # Define background color range in HSV space
    light_gray = (0,0,20)  # converted from HSV value obtained with colorpicker (150,50,0)
    dark_gray  = (255,5,255)  # converted from HSV value obtained with colorpicker (250,100,100)

    # light_blue = (0,0,0)  # converted from HSV value obtained with colorpicker (150,50,0)
    # dark_blue  = (255,50,255)  # converted from HSV value obtained with colorpicker (250,100,100)

    # Mark pixels outside background color range
    mask = ~cv.inRange(img_hsv, light_gray, dark_gray)
    return mask
