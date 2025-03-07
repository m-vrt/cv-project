import cv2
import torch
import os
import numpy as np
import json
import warnings
from segment_anything import sam_model_registry, SamPredictor


warnings.filterwarnings("ignore")


model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint="segment_anything/sam_model.pth")
sam.to("cuda")
predictor = SamPredictor(sam)


video_path = "C:/Users/Donna/Documents/m-vrt/cv-project/output_sam_trimmed.mp4"  
predictions_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/sam"
os.makedirs(predictions_folder, exist_ok=True)


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
    frame_filename = f"frame_{frame_count:04d}.json"
    frame_output_path = os.path.join(predictions_folder, frame_filename)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

   
    masks, _, _ = predictor.predict(multimask_output=True)

    objects = []
    for mask in masks:
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            objects.append({"class_id": 1, "bbox": bbox})  

   
    with open(frame_output_path, "w") as f:
        json.dump({"frame": frame_filename, "objects": objects}, f, indent=4)

cap.release()
print(f"Inference complete. Predictions saved in: {predictions_folder}")