import cv2
import torch
import os
import numpy as np
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


print(f"Original FPS: {original_fps}")


resize_width, resize_height = 1280, 720


output_path = "output_yolo_seg.mp4"


if os.path.exists(output_path):
    os.remove(output_path)
    print("Previous output file deleted.")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, original_fps, (resize_width, resize_height))


CLASS_COLORS = {
    "person": (50, 50, 255),        
    "bicycle": (255, 100, 100),      
    "car": (180, 180, 50),          
    "motorcycle": (255, 140, 70),   
    "bus": (128, 0, 128),            
    "truck": (255, 165, 90),        
    "stop sign": (255, 180, 203),    
    "traffic light": (50, 255, 50)   
}


def get_color(class_id):
    class_name = model.names[int(class_id)].lower()
    return CLASS_COLORS.get(class_name, (200, 200, 200))  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.resize(frame, (resize_width, resize_height))

    
    results = model(frame)[0]

   
    if results.masks:
        blended_frame = frame.copy()  
        for mask in results.masks.data:
            mask = mask.cpu().numpy()  
            mask = cv2.resize(mask, (resize_width, resize_height)) 

            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            color = (200, 200, 100)  

            for c in range(3):  
                colored_mask[:, :, c] = (mask * color[c]).astype(np.uint8)

           
            blended_frame = cv2.addWeighted(blended_frame, 0.85, colored_mask, 0.15, 0)

        frame = blended_frame 

    
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = get_color(int(cls))

        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        
        font_scale = 0.5  
        thickness = 1     
        text_offset = 5  

        
        cv2.putText(frame, label, (int(x1), int(y1) - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)  
        cv2.putText(frame, label, (int(x1), int(y1) - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)  

   
    out.write(frame)
    cv2.imshow("YOLOv8 Segmentation + Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_path)
