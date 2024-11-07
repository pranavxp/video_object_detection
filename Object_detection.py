import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    # Temporary workaround for libiomp5md.dll error

import cv2
import time
from ultralytics import YOLO  # Requires torch

model = YOLO('yolov8l.pt')
video_path = "test/dash.mp4"
cap = cv2.VideoCapture(video_path)
new_width = 680
new_height = 480

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            print(box.xyxy)

            if len(box.xyxy) > 0:
                x1, y1, x2, y2 = box.xyxy[0][:4].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()

                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow("Object Detection", resized_frame)
    t2 = time.time()
    print('Time Taken Per Frame = ', t2 - t1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
