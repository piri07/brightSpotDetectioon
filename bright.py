# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 09:41:57 2020

@author: PRIYANSHU
"""


from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2


image1 = cv2.imread("img.png")
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11,11), 0)


thresh = cv2.threshold(blurred,200,255,cv2.THRESH_BINARY)[1]

#still after converting the bright spots to white there is noise 
#to remove we perfrom erosion and dilation 

thresh = cv2.erode(thresh,None,iterations=4)
thresh = cv2.dilate(thresh, None,iterations=4)

#after applying erode and dilation the noise in the image is reduced


#now to ensure that there is no more noise we perform a CONNECTED-COMPONENT-CHECK
labels = measure.label(thresh,neighbors=8,background=0)
#make a mask of the dimensions same as the threshold image
mask = np.zeros(thresh.shape,dtype= "uint8")

for label in np.unique(labels):
    if label==0:
        #if this is a background label the ignore it 
        continue
    labelMask = np.zeros(thresh.shape,dtype= "uint8")
    labelMask[label==labels]=255
    numpixels = cv2.countNonZero(labelMask)
    
    if numpixels>300:
        #if the count non zeros becomes greater than 300
        #then we can add these pixels to the mask
        mask = cv2.add(mask,labelMask)
        
        
cnts =cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#find contours then sort them from left to right
cnts = contours.sort_contours(cnts)[0]

#loop over the contours
for i,c in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    ((cX,cY),radius) =cv2.minEnclosingCircle(c)
    cv2.circle(image1,(int(cX),int(cY)),int(radius),(0,0,255),3)
    cv2.putText(image1, "#{}".format(i + 1), (x, y - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Image", image1)
cv2.waitKey(0)




    
    

    
    
    
    
    
    
    
    
    
    
