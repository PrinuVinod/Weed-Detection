import cv2 
import numpy as np 

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set to store coordinates of detected circles
detected_circles = set()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Convert the resulting image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=50)

    if circles is not None:
        # Convert coordinates to range from 0 to 10
        scaled_circles = np.round(circles[0, :]).astype("int")
        scaled_circles[:, :2] = np.clip(np.round(scaled_circles[:, :2] / 640 * 10), 0, 10)

        # Print the coordinates of the detected circles if they are not already detected
        for (x, y, r) in scaled_circles:
            if (x, y) not in detected_circles:
                detected_circles.add((x, y))
                print(f"Detected circle at ({x}, {y})")

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
