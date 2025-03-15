import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules.conv import Conv
from torch.nn import Sequential


add_safe_globals([SegmentationModel, Sequential, Conv])
torch_load = torch.load

def custom_torch_load(file, *args, **kwargs):
    kwargs["weights_only"] = False
    return torch_load(file, *args, **kwargs)

torch.load = custom_torch_load


model_path = r"C:\Users\Donna\Documents\m-vrt\cv-project\yolov8n-seg.pt"
image_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\yolo\test\images"
output_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\yolo\predictions"


if os.path.exists(output_folder):
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Deleted existing prediction files in {output_folder}")
else:
    os.makedirs(output_folder, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)


image_files = sorted(os.listdir(image_folder))[:300]

for image_filename in image_files:
    image_path = os.path.join(image_folder, image_filename)

    if os.path.exists(image_path):
       
        image = cv2.imread(image_path)

        
        results = model(image, conf=0.05, iou=0.5)[0]  

        
        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        txt_output_path = os.path.join(output_folder, txt_filename)

        with open(txt_output_path, "w") as f:
            if results.boxes is not None:
                for box in results.boxes.data.cpu().numpy():
                    cls_id, x1, y1, x2, y2, conf = int(box[5]), box[0], box[1], box[2], box[3], box[4]
                    f.write(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")

       
        if results.masks is not None and len(results.masks.data) > 0:
            mask = np.zeros_like(image[:, :, 0])  
            for m in results.masks.data.cpu().numpy():  
                mask = np.maximum(mask, (m * 255).astype(np.uint8))  
            
            mask_output_filename = f"{os.path.splitext(image_filename)[0]}_mask.jpg"
            mask_output_path = os.path.join(output_folder, mask_output_filename)
            cv2.imwrite(mask_output_path, mask)
            print(f"Segmentation mask saved at: {mask_output_path}")

print(f"Batch Inference Complete! Predictions and segmentation masks saved in {output_folder}")