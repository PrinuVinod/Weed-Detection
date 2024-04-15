import cv2
import numpy as np
import serial

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

arduino = serial.Serial('COM8', 9600)

detected_circles = set()

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=50)

    if circles is not None:
        scaled_circles = np.round(circles[0, :]).astype("int")
        scaled_circles[:, :2] = np.clip(np.round(scaled_circles[:, :2] / 640 * 10), 0, 10)

        for (x, y, r) in scaled_circles:
            if (x, y) not in detected_circles:
                detected_circles.add((x, y))
                print(f"Detected circle at ({x}, {y})")
                arduino.write(f"{x},{y}\n".encode())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
