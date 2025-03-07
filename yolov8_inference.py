import cv2
import os
import numpy as np
import torch
from torch.serialization import add_safe_globals
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules.conv import Conv  
from torch.nn import Sequential  


add_safe_globals([SegmentationModel, Sequential, Conv])


torch_load = torch.load  

def custom_torch_load(file, *args, **kwargs):
    kwargs["weights_only"] = False  
    return torch_load(file, *args, **kwargs)

torch.load = custom_torch_load  


model_path = "yolov8n-seg.pt"
model = YOLO(model_path)


video_path = "C:/Users/Donna/Documents/m-vrt/cv-project/output_yolo_seg_trimmed.mp4"
predictions_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/yolov8"
extracted_frames_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/extracted_frames/yolov8"
os.makedirs(predictions_folder, exist_ok=True)
os.makedirs(extracted_frames_folder, exist_ok=True)


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_filename_txt = f"frame_{frame_count:04d}.txt"
    frame_output_path_txt = os.path.join(predictions_folder, frame_filename_txt)

   
    frame_filename_png = f"frame_{frame_count:04d}.png"
    frame_output_path_png = os.path.join(extracted_frames_folder, frame_filename_png)
    cv2.imwrite(frame_output_path_png, frame) 

   
    results = model(frame)[0]

    if results.boxes is None or len(results.boxes.data) == 0:
        print(f"Warning: No objects detected in frame {frame_count}.")
        continue

    with open(frame_output_path_txt, "w") as f:
        for i, box in enumerate(results.boxes.data):
            if len(box) < 6:  
                continue
            x1, y1, x2, y2, conf, cls_id = box.cpu().numpy()

            f.write(f"{int(cls_id)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")

           
            if hasattr(results, "masks") and results.masks.data is not None and i < len(results.masks.data):
                mask = results.masks.data[i].cpu().numpy()
                
                if mask.shape[0] > 0 and mask.shape[1] > 0:  
                    mask_filename = f"frame_{frame_count:04d}_mask_{int(cls_id)}.png"
                    mask_output_path = os.path.join(predictions_folder, mask_filename)
                    cv2.imwrite(mask_output_path, (mask * 255).astype(np.uint8))

cap.release()
print(f"Inference complete. Predictions and frames saved in:\n - Predictions: {predictions_folder}\n - Extracted Frames: {extracted_frames_folder}")
