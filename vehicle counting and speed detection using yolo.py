import cv2
import numpy as np

# Load YOLO model and configuration files
yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load video
cap = cv2.VideoCapture("los_angeles.mp4")

# Define output layers
output_layers = yolo_net.getUnconnectedOutLayersNames()

# Define the counting line (you can adjust these values)
count_line_pos = 700
line_thickness = 2
offset = 6
counter = 0

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
prev_positions = {}  # Store previous positions of detected vehicles

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2 and class_id == 2:  # Class ID 2 represents cars
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                detect.append(center_handle(x, y, w, h))

                for (x, y) in detect:
                    if count_line_pos - offset < y < count_line_pos + offset:
                        counter += 1
                        detect.remove((x, y))
                        print("Counter: " + str(counter))

                if y in prev_positions:
                    speed = count_line_pos - (y + h + 10)  # Simple relative speed based on position change
                    speed_text = f"Speed: {speed} km/h"
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 0, 0), -1)
                    cv2.putText(frame, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
                    print("Speed of vehicle is", speed)

                prev_positions[y] = y + h

    cv2.putText(frame, "Vehicle Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.line(frame, (0, count_line_pos), (width, count_line_pos), (255, 127, 0), line_thickness)

    cv2.imshow("Car Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
