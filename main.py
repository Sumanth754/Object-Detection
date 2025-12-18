import cv2
import numpy as np
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "dnn_model")
VIDEO_PATH = os.path.join(BASE_DIR, "sample.mp4")
CONFIG_PATH = os.path.join(MODEL_DIR, "yolov3.cfg")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

# Load YOLO
try:
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    if net.empty():
        raise cv2.error("Failed to load network from %s and %s" % (WEIGHTS_PATH, CONFIG_PATH))
except cv2.error as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'yolov3.weights' and 'yolov3.cfg' are in the 'dnn_model' folder.")
    print("You might need to download them if you haven't already.")
    exit()


# Get output layer names
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
if len(unconnected_out_layers.shape) == 1: # Handle single output layer case
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
else: # Handle multiple output layers case (more common with newer OpenCV)
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_out_layers]


# Load class names
classes = []
try:
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: {NAMES_PATH} not found. Please ensure 'coco.names' is in the 'dnn_model' folder.")
    exit()

# Set up video capture
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}.")
    print("Please ensure 'sample.mp4' is in the project root directory.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Processing video: {VIDEO_PATH}")
print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Confidence and NMS threshold
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 30), font, 2, color, 2)

    cv2.imshow("Real-Time Retail Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
