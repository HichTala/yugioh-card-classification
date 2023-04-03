import cv2
import numpy as np

image = cv2.imread('gg_0.jpg')
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(image, -1, sharpen_kernel)

cv2.imshow('sharpen', sharpen)
cv2.imwrite('gg_1.jpg', sharpen)
