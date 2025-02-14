import cv2
import torch
import os
import numpy as np
import random
from ultralytics import YOLO

model_path = "yolov8n-seg.pt"
model = YOLO(model_path)

video_path = "for cv-project.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

original_fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_width, output_height = frame_width, frame_height
output_path = "output_yolo_seg.mp4"

if os.path.exists(output_path):
    os.remove(output_path)
    print("Previous output file deleted.")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, original_fps, (output_width, output_height))

CLASS_COLORS = {}

def get_color(class_id):
    """ Assigns muted, professional colors dynamically for each object class. """
    if class_id not in CLASS_COLORS:
        muted_palette = [
            (60, 180, 100),  
            (90, 140, 200), 
            (180, 80, 80),  
            (190, 160, 90),  
            (140, 140, 140)  
        ]
        CLASS_COLORS[class_id] = muted_palette[class_id % len(muted_palette)]
    return CLASS_COLORS[class_id]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))

    results = model(frame)[0]

    if results.masks:
        blended_frame = frame.copy()
        for mask, class_id in zip(results.masks.data, results.boxes.cls):
            mask = mask.cpu().numpy()
            mask = cv2.resize(mask, (output_width, output_height))

            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            color = get_color(int(class_id))

            for c in range(3):
                colored_mask[:, :, c] = (mask * color[c]).astype(np.uint8)

            colored_mask = cv2.GaussianBlur(colored_mask, (5, 5), 3)

            frame = cv2.addWeighted(frame, 0.85, colored_mask, 0.15, 0)  

    for box, class_id in zip(results.boxes.data, results.boxes.cls):
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = get_color(int(cls))

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        font_scale = 0.5
        thickness = 1
        text_color = (200, 200, 200) 
        text_bg_color = (50, 50, 50)  

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (int(x1), int(y1) - text_h - 5), (int(x1) + text_w + 5, int(y1)), text_bg_color, -1)

        cv2.putText(frame, label, (int(x1) + 2, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("YOLOv8 Segmentation + Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_path)
