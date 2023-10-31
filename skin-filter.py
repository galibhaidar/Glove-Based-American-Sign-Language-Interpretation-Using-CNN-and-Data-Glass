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




    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 60, 70], dtype="uint8")
    upper = np.array([13, 255, 255], dtype="uint8")

    skinMask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # kernel = (5, 5)
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=4)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # skinMask = cv2.medianBlur(skinMask, 3)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 0)

    # cv2.imwrite('train-dir-D/D.' + str(num) + '.jpg', res[y1:y2, x1:x2])

    # cv2.imshow('mask', mask)
    # cv2.imshow('roi', roi)
    cv2.imshow('frame', skin)
    cv2.imshow('frame1', frame)
    num += 1

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif num > 1500:
        break
    #  cv2.ro

cap.release()
cv2.destroyAllWindows()
