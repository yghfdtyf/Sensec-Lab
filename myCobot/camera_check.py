
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 水平翻转帧
    frame = cv2.flip(frame, 1)  # 1 表示沿x轴翻转
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()