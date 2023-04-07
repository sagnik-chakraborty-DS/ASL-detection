import cv2
import time
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
################################
# wCam, hCam = 2000, 2000
################################

cap = cv2.VideoCapture(1)
# cap.set(3, wCam)
# cap.set(4, hCam)
pTime = 0

detector = HandDetector(maxHands=1, detectionCon=.9)
imgSize = 300
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-20:y+h+20, x-20:x+w+20]



        aspectRatio = h/w
        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal,imgSize))
            imgResizeShape = imgResize.shape
            imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgCrop

        cv2.imshow("ImgCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (900, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Img", img)

    cv2.waitKey(1)