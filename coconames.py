import cv2
import numpy as np
# Load COCO class labels
coco_names = open('coco.names').read().strip().split('\n')

# Load the YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')   

# Load YOLO network
layer_names = net.getUnconnectedOutLayersNames()

# Open video file
video_path = 'Path to file'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    height, width = frame.shape[:2]

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run forward pass and get output
    detections = net.forward(layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]

            if confidence > 0.5 and coco_names[class_id] == 'person':
                # Display the label "Human"
                cv2.putText(frame, 'Human', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Human Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()