import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    red_circles = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(red_circles, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: 
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) 

        M = cv2.moments(contour) 
        if M['m00'] != 0.0: 
            x = int(M['m10']/M['m00']) 
            y = int(M['m01']/M['m00']) 

            if len(approx) > 6:
                x_normalized = int((x / frame.shape[1]) * 10)
                y_normalized = int((y / frame.shape[0]) * 10)

                print(f'Red Circle ({x_normalized}, {y_normalized})') 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()