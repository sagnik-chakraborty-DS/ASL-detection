import cv2
import time
from cvzone.HandTrackingModule import HandDetector
################################
wCam, hCam = 2000, 2000
################################

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    detector.findHands(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (900, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)