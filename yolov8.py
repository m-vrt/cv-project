import cv2
import torch
from ultralytics import YOLO


model_path = "yolov8n.pt"  
model = YOLO(model_path)


video_path = "for cv-project.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


output_path = "output_yolo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_skip = max(1, fps // 10)  
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue 

   
    results = model(frame)[0]  

    
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0)  

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    out.write(frame)

    
    cv2.imshow("YOLOv8 Object Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_path)
