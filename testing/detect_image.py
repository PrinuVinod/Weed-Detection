import cv2
import numpy as np
import os
import ctypes

# Loading class labels YOLO model was trained on
labelsPath = 'obj.names'
LABELS = open(labelsPath).read().strip().split("\n")

# Load weights and cfg
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'
# Color selection for drawing bbox
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Initialize YOLO model
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get the list of image files in the 'images' folder
image_folder = 'images'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Loop through each image
for image_file in image_files:
    # Load our input image and grab its spatial dimensions
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # Parameters
    confi = 0.5
    thresh = 0.5

    # Determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the input image and then perform a forward
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions
            if confidence > confi:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update lists of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w - 100, y + h - 100), color, 2)
            print(f"[ACCURACY] : accuracy -> {confidences[i]}")
            print(f"[OUTPUT]   : detected label -> {LABELS[classIDs[i]]}")
            text = f"{LABELS[classIDs[i]]} : {confidences[i]:.4f}"
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Get screen width and height
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    # Resize image to fit the screen
    scale_factor = min(screen_width / W, screen_height / H)
    resized_image = cv2.resize(image, (int(W * scale_factor), int(H * scale_factor)))

    # Set the window name and resize the window
    window_name = 'Output'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(W * scale_factor), int(H * scale_factor))

    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("[STATUS]   : Completed")
print("[END]")


# old code 1
# import cv2
# import numpy as np
# import os

# # Loading class labels YOLO model was trained on
# labelsPath = 'obj.names'
# LABELS = open(labelsPath).read().strip().split("\n")

# # Load weights and cfg
# weightsPath = 'crop_weed_detection.weights'
# configPath = 'crop_weed.cfg'
# # Color selection for drawing bbox
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# # Initialize YOLO model
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# # Get the list of image files in the 'images' folder
# image_folder = 'images'
# image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# # Loop through each image
# for image_file in image_files:
#     # Load our input image and grab its spatial dimensions
#     image_path = os.path.join(image_folder, image_file)
#     image = cv2.imread(image_path)
#     (H, W) = image.shape[:2]

#     # Parameters
#     confi = 0.5
#     thresh = 0.5

#     # Determine only the *output* layer names that we need from YOLO
#     ln = net.getLayerNames()
#     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#     # Construct a blob from the input image and then perform a forward
#     blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
#     net.setInput(blob)
#     layerOutputs = net.forward(ln)

#     # Initialize lists of detected bounding boxes, confidences, and class IDs
#     boxes = []
#     confidences = []
#     classIDs = []

#     # Loop over each of the layer outputs
#     for output in layerOutputs:
#         # Loop over each of the detections
#         for detection in output:
#             # Extract the class ID and confidence of the current object detection
#             scores = detection[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]

#             # Filter out weak predictions
#             if confidence > confi:
#                 box = detection[0:4] * np.array([W, H, W, H])
#                 (centerX, centerY, width, height) = box.astype("int")
#                 x = int(centerX - (width / 2))
#                 y = int(centerY - (height / 2))

#                 # Update lists of bounding box coordinates, confidences, and class IDs
#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 classIDs.append(classID)

#     # Apply non-maxima suppression
#     idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

#     # Ensure at least one detection exists
#     if len(idxs) > 0:
#         # Loop over the indexes we are keeping
#         for i in idxs.flatten():
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])

#             color = [int(c) for c in COLORS[classIDs[i]]]
#             cv2.rectangle(image, (x, y), (x + w - 100, y + h - 100), color, 2)
#             print(f"[ACCURACY] : accuracy -> {confidences[i]}")
#             print(f"[OUTPUT]   : detected label -> {LABELS[classIDs[i]]}")
#             text = f"{LABELS[classIDs[i]]} : {confidences[i]:.4f}"
#             cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     cv2.imshow('Output', det)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# print("[STATUS]   : Completed")
# print("[END]")


# old code
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import os


# #loading class labels YOLO model was trained on
# labelsPath = 'obj.names'
# LABELS = open(labelsPath).read().strip().split("\n")

# #load weights and cfg
# weightsPath = 'crop_weed_detection.weights'
# configPath = 'crop_weed.cfg'
# #color selection for drawing bbox
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
# print("Loading Images from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# #load our input image and grab its spatial dimensions
# image = cv2.imread('images/21.jpg')
# (H, W) = image.shape[:2]

# #parameters
# confi = 0.5
# thresh = 0.5

# #determine only the *output* layer names that we need from YOLO
# # ln = net.getLayerNames()
# # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ln = net.getLayerNames()
# ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# #construct a blob from the input image and then perform a forward
# #pass of the YOLO object detector, giving us our bounding boxes and
# #associated probabilities
# blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512),swapRB=True, crop=False)
# net.setInput(blob)
# start = time.time()
# layerOutputs = net.forward(ln)
# end = time.time()

# #show timing information on YOLO
# print("YOLO took {:.6f} seconds".format(end - start))

# #initialize our lists of detected bounding boxes, confidences, and
# #class IDs, respectively
# boxes = []
# confidences = []
# classIDs = []

# #loop over each of the layer outputs
# for output in layerOutputs:
# 	#loop over each of the detections
# 	for detection in output:
# 		#extract the class ID and confidence (i.e., probability) of
# 		#the current object detection
# 		scores = detection[5:]
# 		classID = np.argmax(scores)
# 		confidence = scores[classID]

# 		#filter out weak predictions by ensuring the detected
# 		#probability is greater than the minimum probability
# 		if confidence > confi:
# 			#scale the bounding box coordinates back relative to the
# 			#size of the image, keeping in mind that YOLO actually
# 			#returns the center (x, y)-coordinates of the bounding
# 			#box followed by the boxes' width and height
# 			box = detection[0:4] * np.array([W, H, W, H])
# 			(centerX, centerY, width, height) = box.astype("int")

# 			#use the center (x, y)-coordinates to derive the top and
# 			#and left corner of the bounding box
# 			x = int(centerX - (width / 2))
# 			y = int(centerY - (height / 2))

# 			#update our list of bounding box coordinates, confidences,
# 			#and class IDs
# 			boxes.append([x, y, int(width), int(height)])
# 			confidences.append(float(confidence))
# 			classIDs.append(classID)

# #apply non-maxima suppression to suppress weak, overlapping bounding
# #boxes
# idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
# print("Detections done\nDrawing bounding boxes...")
# #ensure at least one detection exists
# if len(idxs) > 0:
# 	#loop over the indexes we are keeping
# 	for i in idxs.flatten():
# 		#extract the bounding box coordinates
# 		(x, y) = (boxes[i][0], boxes[i][1])
# 		(w, h) = (boxes[i][2], boxes[i][3])

# 		#draw a bounding box rectangle and label on the image
# 		color = [int(c) for c in COLORS[classIDs[i]]]
# 		cv2.rectangle(image, (x, y), (x + w-100, y + h-100), color, 2)
# 		print("[ACCURACY] : accuracy -> ", confidences[i])
# 		print("[OUTPUT]   : detected label -> ",LABELS[classIDs[i]])
# 		text = "{} : {:.4f}".format(LABELS[classIDs[i]] , confidences[i])
# 		cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.8, color, 2)
# det = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# cv2.imshow('Output', det)
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("[STATUS]   : Completed")
# print("[END]")