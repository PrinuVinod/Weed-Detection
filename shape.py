import cv2 
import numpy as np 
import serial
import time

ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

cap = cv2.VideoCapture(0)

detected_circles = []
servos_active = False

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

    if len(contours) == 0 and servos_active:
        ser.write("0,0\n".encode())
        servos_active = False

    for contour in contours: 
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) 

        M = cv2.moments(contour) 
        if M['m00'] != 0.0: 
            x = int(M['m10']/M['m00']) 
            y = int(M['m01']/M['m00']) 

            if len(approx) > 6:
                x_normalized = int((x / frame.shape[1]) * 10)
                y_normalized = int((y / frame.shape[0]) * 10)

                if (x_normalized, y_normalized) not in detected_circles:
                    detected_circles.append((x_normalized, y_normalized))
                    print(f'Detected: ({x_normalized}, {y_normalized})')

                    ser.write(f"{x_normalized},{y_normalized}\n".encode())
                    
                    if not servos_active:
                        ser.write("1,1\n".encode())
                        servos_active = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()

cap.release()
cv2.destroyAllWindows()
