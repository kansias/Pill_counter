import cv2 as cv
import os

img_path = ".\\img\\"
img_list = os.listdir(img_path)
img_ratio = 0.2

img_ori = cv.imread(img_path+img_list[0], cv.IMREAD_UNCHANGED)
# img_RGB = cv.cvtColor(img_ori, cv.COLOR_BGR2RGB)
img = cv.resize(img_ori, (0, 0), fx=img_ratio, fy=img_ratio)


cv.imshow('origin', img)
cv.waitKey()
cv.destroyAllWindows()
