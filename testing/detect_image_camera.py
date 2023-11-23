# phone camera using IP Webcam (front cam)
import cv2
import numpy as np
import time

# Loading class labels YOLO model was trained on
labelsPath = 'obj.names'
LABELS = open(labelsPath).read().strip().split("\n")

# Load weights and cfg
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'
# Color selection for drawing bbox
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
print("Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Specify the IP address and port number used by the DroidCam app
# Replace 'your_phone_ip' and 'your_port_number' with the actual values
camera_address = 'http://172.20.10.3:2500/video'

# Start capturing video from the phone's camera via DroidCam
cap = cv2.VideoCapture(camera_address)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error reading from camera.")
        break

    # Grab the spatial dimensions
    (H, W) = frame.shape[:2]

    # Parameters
    confi = 0.5
    thresh = 0.5

    # Determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Show timing information on YOLO
    print("YOLO took {:.6f} seconds".format(end - start))

    # Initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confi:
                # Scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
    print("Detections done\nDrawing bounding boxes...")
    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w - 100, y + h - 100), color, 2)
            print("[ACCURACY] : accuracy -> ", confidences[i])
            print("[OUTPUT]   : detected label -> ", LABELS[classIDs[i]])
            text = "{} : {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the frame
    cv2.imshow('Output', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
print("[STATUS]   : Completed")
print("[END]")





# phone camera using USB
# import cv2
# import numpy as np
# import time

# # Loading class labels YOLO model was trained on
# labelsPath = 'obj.names'
# LABELS = open(labelsPath).read().strip().split("\n")

# # Load weights and cfg
# weightsPath = 'crop_weed_detection.weights'
# configPath = 'crop_weed.cfg'
# # Color selection for drawing bbox
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# print("Loading YOLO model...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# # Use the DroidCam virtual camera index
# droidcam_index = 1  # Change this to the correct DroidCam virtual camera index

# # Start capturing video from the DroidCam virtual camera
# cap = cv2.VideoCapture(droidcam_index)

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()

#     if not ret:
#         print("Error reading from camera.")
#         break

#     # Grab the spatial dimensions
#     (H, W) = frame.shape[:2]

#     # Parameters
#     confi = 0.5
#     thresh = 0.5

#     # Determine only the *output* layer names that we need from YOLO
#     ln = net.getLayerNames()
#     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#     # Construct a blob from the input image and then perform a forward
#     # pass of the YOLO object detector, giving us our bounding boxes and
#     # associated probabilities
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (512, 512), swapRB=True, crop=False)
#     net.setInput(blob)
#     start = time.time()
#     layerOutputs = net.forward(ln)
#     end = time.time()

#     # Show timing information on YOLO
#     print("YOLO took {:.6f} seconds".format(end - start))

#     # Initialize our lists of detected bounding boxes, confidences, and
#     # class IDs, respectively
#     boxes = []
#     confidences = []
#     classIDs = []

#     # Loop over each of the layer outputs
#     for output in layerOutputs:
#         # Loop over each of the detections
#         for detection in output:
#             # Extract the class ID and confidence (i.e., probability) of
#             # the current object detection
#             scores = detection[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]

#             # Filter out weak predictions by ensuring the detected
#             # probability is greater than the minimum probability
#             if confidence > confi:
#                 # Scale the bounding box coordinates back relative to the
#                 # size of the image, keeping in mind that YOLO actually
#                 # returns the center (x, y)-coordinates of the bounding
#                 # box followed by the boxes' width and height
#                 box = detection[0:4] * np.array([W, H, W, H])
#                 (centerX, centerY, width, height) = box.astype("int")

#                 # Use the center (x, y)-coordinates to derive the top and
#                 # and left corner of the bounding box
#                 x = int(centerX - (width / 2))
#                 y = int(centerY - (height / 2))

#                 # Update our list of bounding box coordinates, confidences,
#                 # and class IDs
#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 classIDs.append(classID)

#     # Apply non-maxima suppression to suppress weak, overlapping bounding
#     # boxes
#     idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
#     print("Detections done\nDrawing bounding boxes...")
#     # Ensure at least one detection exists
#     if len(idxs) > 0:
#         # Loop over the indexes we are keeping
#         for i in idxs.flatten():
#             # Extract the bounding box coordinates
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])

#             # Draw a bounding box rectangle and label on the image
#             color = [int(c) for c in COLORS[classIDs[i]]]
#             cv2.rectangle(frame, (x, y), (x + w - 100, y + h - 100), color, 2)
#             print("[ACCURACY] : accuracy -> ", confidences[i])
#             print("[OUTPUT]   : detected label -> ", LABELS[classIDs[i]])
#             text = "{} : {:.4f}".format(LABELS[classIDs[i]], confidences[i])
#             cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     # Display the frame
#     cv2.imshow('Output', frame)

#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()
# print("[STATUS]   : Completed")
# print("[END]")




# laptop camera
# import cv2
# import numpy as np
# import time

# # loading class labels YOLO model was trained on
# labelsPath = 'obj.names'
# LABELS = open(labelsPath).read().strip().split("\n")

# # load weights and cfg
# weightsPath = 'crop_weed_detection.weights'
# configPath = 'crop_weed.cfg'
# # color selection for drawing bbox
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# print("Loading YOLO model...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# # start capturing video from the camera
# cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (you can change it if you have multiple cameras)

# while True:
#     # read a frame from the camera
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Error reading from camera.")
#         break

#     # grab the spatial dimensions
#     (H, W) = frame.shape[:2]

#     # parameters
#     confi = 0.5
#     thresh = 0.5

#     # determine only the *output* layer names that we need from YOLO
#     ln = net.getLayerNames()
#     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#     # construct a blob from the input image and then perform a forward
#     # pass of the YOLO object detector, giving us our bounding boxes and
#     # associated probabilities
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (512, 512), swapRB=True, crop=False)
#     net.setInput(blob)
#     start = time.time()
#     layerOutputs = net.forward(ln)
#     end = time.time()

#     # show timing information on YOLO
#     print("YOLO took {:.6f} seconds".format(end - start))

#     # initialize our lists of detected bounding boxes, confidences, and
#     # class IDs, respectively
#     boxes = []
#     confidences = []
#     classIDs = []

#     # loop over each of the layer outputs
#     for output in layerOutputs:
#         # loop over each of the detections
#         for detection in output:
#             # extract the class ID and confidence (i.e., probability) of
#             # the current object detection
#             scores = detection[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]

#             # filter out weak predictions by ensuring the detected
#             # probability is greater than the minimum probability
#             if confidence > confi:
#                 # scale the bounding box coordinates back relative to the
#                 # size of the image, keeping in mind that YOLO actually
#                 # returns the center (x, y)-coordinates of the bounding
#                 # box followed by the boxes' width and height
#                 box = detection[0:4] * np.array([W, H, W, H])
#                 (centerX, centerY, width, height) = box.astype("int")

#                 # use the center (x, y)-coordinates to derive the top and
#                 # and left corner of the bounding box
#                 x = int(centerX - (width / 2))
#                 y = int(centerY - (height / 2))

#                 # update our list of bounding box coordinates, confidences,
#                 # and class IDs
#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 classIDs.append(classID)

#     # apply non-maxima suppression to suppress weak, overlapping bounding
#     # boxes
#     idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
#     print("Detections done\nDrawing bounding boxes...")
#     # ensure at least one detection exists
#     if len(idxs) > 0:
#         # loop over the indexes we are keeping
#         for i in idxs.flatten():
#             # extract the bounding box coordinates
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])

#             # draw a bounding box rectangle and label on the image
#             color = [int(c) for c in COLORS[classIDs[i]]]
#             cv2.rectangle(frame, (x, y), (x + w - 100, y + h - 100), color, 2)
#             print("[ACCURACY] : accuracy -> ", confidences[i])
#             print("[OUTPUT]   : detected label -> ", LABELS[classIDs[i]])
#             text = "{} : {:.4f}".format(LABELS[classIDs[i]], confidences[i])
#             cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     # Display the frame
#     cv2.imshow('Output', frame)

#     # break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()
# print("[STATUS]   : Completed")
# print("[END]")