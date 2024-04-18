import cv2 
import numpy as np 

# Accessing the camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Get the frame dimensions to calculate scaling factors
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale_x = 10 / frame_width
scale_y = 10 / frame_height

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # converting frame into grayscale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # setting threshold of gray image 
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 

    # using a findContours() function 
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # loop through all contours
    for contour in contours: 

        # calculate the area of the contour
        area = cv2.contourArea(contour)

        # filter contours based on area
        if area > 100: # adjust the threshold based on your image and requirements

            # calculate the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)

            # approximate the contour to detect shape
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # filter contours to detect red circles
            if len(approx) > 8: # adjust the number of sides for circles
                # calculate center and radius of the circle
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                # filter circles based on color
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])
                mask = cv2.inRange(hsv, lower_red, upper_red)

                # calculate the number of non-zero pixels in the mask
                total_pixels = cv2.countNonZero(mask)
                if total_pixels > 0.5 * perimeter: # adjust threshold based on your image and requirements
                    # Map coordinates to 0-10 scale
                    scaled_x = int(x * scale_x)
                    scaled_y = int(y * scale_y)
                    
                    # Output scaled coordinates
                    print(f"Detected circle at ({scaled_x}, {scaled_y})")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
