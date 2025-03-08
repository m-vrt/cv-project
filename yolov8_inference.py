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


kitti_images_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/kitti/image_2/testing/image_2"
predictions_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/yolov8"
os.makedirs(predictions_folder, exist_ok=True)


MAX_IMAGES = 300
image_files = sorted(os.listdir(kitti_images_folder))[:MAX_IMAGES]  

for image_filename in image_files:
    if image_filename.endswith(".png"):
        image_path = os.path.join(kitti_images_folder, image_filename)
        image = cv2.imread(image_path)

        
        results = model(image)[0]

        
        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        txt_output_path = os.path.join(predictions_folder, txt_filename)

        if results.boxes is None or len(results.boxes.data) == 0:
            print(f"Warning: No objects detected in {image_filename}.")
            continue

        with open(txt_output_path, "w") as f:
            for i, box in enumerate(results.boxes.data):
                if len(box) < 6:
                    continue
                x1, y1, x2, y2, conf, cls_id = box.cpu().numpy()
                f.write(f"{int(cls_id)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")

print(f"Inference complete on {MAX_IMAGES} images. Predictions saved in: {predictions_folder}")
