import cv2
import numpy as np

cap = cv2.VideoCapture(0)
num = 1

x1 = 0
x2 = 210

y1 = 70
y2 = 320

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 0, 255)
stroke = 1

while True:
    _, frame = cap.read()

    # roi = frame[y1:y2, x1:x2]

    kernel = np.ones((3, 3), np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([110, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    median = cv2.medianBlur(res, 3)
    cv2.imshow('Median Blur', median)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 0)

    # cv2.imwrite('train-dir-D/D.' + str(num) + '.jpg', frame[y1:y2, x1:x2])

    # cv2.imshow('mask', mask)
    # cv2.imshow('roi', roi)
    cv2.imshow('frame', frame)
    num += 1

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif num > 1500:
        break

cap.release()
cv2.destroyAllWindows()
