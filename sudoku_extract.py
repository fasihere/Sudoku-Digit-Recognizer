import cv2 as cv
import numpy as np
import copy
from get_contours import getContours

img = cv.imread('sudoku-2.jpeg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 3)
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 45, 50)
cv.imshow('Canny', canny)
img_copy = img.copy()

img_corners = getContours(canny, img_copy)

cv.waitKey(0)