# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:57:26 2022

@author: betul
"""

import cv2
import numpy as np

def process_img(img):
    blurred = cv2.GaussianBlur(img, (11,11), 0) 
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower1= np.array([110,50,50])
    upper1= np.array([130, 255, 255])
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)
    mask = lower_mask + upper_mask;
    mask = cv2.erode(mask, None, iterations = 1)
    mask = cv2.dilate(mask, None, iterations = 2)
    (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours,mask
    

def draw_img(contours,imgOriginal):
    c=max(contours,key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    ((x,y), (width,height), rotation) = rect    
    s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    M = cv2.moments(c)
    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    cv2.drawContours(imgOriginal, [box], 0, (0,255,255),2)
    cv2.circle(imgOriginal, center, 5, (255,0,255),-1)
    cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
    return center
    