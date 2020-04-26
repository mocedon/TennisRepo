#!/usr/bin/python
import numpy as np
import cv2
import sys
import scipy.io
import matplotlib
import skimage

cap = cv2.VideoCapture('./../../vid.mp4')
ret, prv = cap.read()
prv = cv2.cvtColor(prv, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im = gray - prv
    cv2.imshow('frame',im)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    prv = gray
cap.release()
cv2.destroyAllWindows()
