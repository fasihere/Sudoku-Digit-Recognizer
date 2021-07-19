from numpy.core.fromnumeric import shape
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from predictor import predict

def getContours(img, orig_img):
    new_img, contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 60000:
            cv.drawContours(orig_img, cnt, -1, (0,255,0), 3)

            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)

            width, height = 900,900

            pts1 = np.float32([[bx,by],[ax,ay],[cx,cy],[dx,dy]])
            pts2 = np.float32([[0,0],[width//2,0],[0,height//2],[width//2,height//2]])

            matrix = cv.getPerspectiveTransform(pts1,pts2)
            img_perspective = cv.warpPerspective(orig_img, matrix, (width//2,height//2))
            img_corners = cv.cvtColor(img_perspective, cv.COLOR_BGR2GRAY)
            threshold, img = cv.threshold(img_corners, 140, 255, cv.THRESH_BINARY_INV)
            crop_value = 7
            cells_arr = []
            for i in range(0,9):
                line = []
                for j in range(0,9):
                    cell = img[i*50+crop_value:(i+1)*50-crop_value,j*50+crop_value:(j+1)*50-crop_value]
                    threshold, cell = cv.threshold(cell, 250, 255, cv.THRESH_BINARY)
                    #kernel = np.ones((3,3))
                    #cell = cv.erode(cell, kernel, iterations = 1)
                    cell = cv.resize(cell,(28,28))
                    if(i==5 and j==1):
                        plt.imshow(cell, cmap='gray')
                        plt.show()
                    line.append(cell)
                cells_arr.append(line)
                cells = np.array(cells_arr)
    cells = np.reshape(cells, (81,28,28,1))
    print(cells.shape)
    predicted_classes = predict(cells)
    predicted_classes = np.reshape(predicted_classes, (9,9))
    for i, cell in enumerate(cells):
        if(np.count_nonzero(cell) < 10):
            predicted_classes[int(i/9)][i%9] = 0
    print(predicted_classes) 
    return img_corners