from concurrent.futures import thread
import cv2 as cv
import os
from matplotlib.colors import Colormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormap
from matplotlib.widgets import Slider


def nothing(x):
    pass


img_path = ".\\img\\"
img_list = os.listdir(img_path)
img_ratio = 0.5
kernel = np.ones((15, 15), np.uint8)

# ---------------------- load to image -------------------------------------
img_ori = cv.imread(img_path+img_list[0], cv.IMREAD_UNCHANGED)
img = cv.resize(img_ori, (0, 0), fx=img_ratio, fy=img_ratio)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# ---------------------- pre-processing -------------------------------------
median = cv.medianBlur(img, 5)
gray = cv.cvtColor(median, cv.COLOR_BGR2GRAY)
adThresh = cv.adaptiveThreshold(
    gray, 25, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)

ret, thresh = cv.threshold(adThresh, 60, 255, 0)

# closing = cv.morphologyEx(adThresh, cv.MORPH_CLOSE, kernel)
# opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
# opening = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel)
# closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
edge = cv.Canny(median, 50, 400)

# -----------------------Object Detecting -------------------------------------
masking = cv.bitwise_or(img, img.copy(), mask=adThresh)

contours, hierarchy = cv.findContours(
    edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


for idex, contour in enumerate(contours):
    rect = cv.minAreaRect(contour)
    area = cv.contourArea(contour)
    print(area)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    if area > 300:
        boundaryBox = cv.drawContours(masking, [box], 0, (0, 255, 0), 3)
    # cv.putText(img, str(idex), box,
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 56), 2, cv.LINE_AA)


# ---------------------- Figure image -----------------------------------------
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(edge, cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(adThresh, cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(masking)

plt.Slider('thresh', 0, 255, valinit=127)

plt.show()


# cv.waitKey()
# cv.destroyAllWindows()
