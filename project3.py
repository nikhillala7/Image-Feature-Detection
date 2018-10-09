import cv2
import numpy as np
from matplotlib import pyplot as plt
#please use 'template1.png' to run this file
img1 = cv2.imread('/Users/kamallala/Downloads/task3_bonus/t3_5.jpg')
img= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
template=cv2.imread('/Users/kamallala/Downloads/template1.png',0)


blr=cv2.GaussianBlur(img,(3,3),0)

laplacian = cv2.Laplacian(blr,cv2.CV_32F)
laplacian_template= cv2.Laplacian(template,cv2.CV_32F)
new=np.array(laplacian, dtype=np.float32)
new1=np.array(laplacian_template, dtype=np.float32)


w, h = template.shape[::-1]


res = cv2.matchTemplate(new,new1,cv2.TM_CCOEFF_NORMED)
threshold = 0.33
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)



cv2.imshow('result',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()