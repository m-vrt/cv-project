import os
import cv2
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np


model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model_weights = r"C:\Users\Donna\Documents\m-vrt\cv-project\detectron_models\model_final_f10217.pkl"
image_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\mask_rcnn\test\images"
output_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\mask_rcnn\predictions"

os.makedirs(output_folder, exist_ok=True)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_config))  
cfg.MODEL.WEIGHTS = model_weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
print(f"✅ Model loaded successfully on {cfg.MODEL.DEVICE.upper()}.\n")


valid_extensions = (".jpg", ".jpeg", ".png")
image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(valid_extensions)]

for image_filename in image_files:
    image_path = os.path.join(image_folder, image_filename)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"⚠️ Skipping non-image file: {image_filename}")
        continue

    print(f"🔹 Processing: {image_filename}")

    
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

   
    if not instances.has("pred_boxes"):
        print(f"⚠️ No objects detected in {image_filename}. Skipping.")
        continue

    pred_boxes = instances.pred_boxes.tensor.numpy()
    pred_classes = instances.pred_classes.numpy()
    pred_scores = instances.scores.numpy()

    
    txt_filename = os.path.splitext(image_filename)[0] + ".txt"
    txt_output_path = os.path.join(output_folder, txt_filename)

    height, width = float(image.shape[0]), float(image.shape[1])  

    with open(txt_output_path, "w") as f:
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            x_min, y_min, x_max, y_max = box
            x_min_norm = x_min / width
            y_min_norm = y_min / height
            x_max_norm = x_max / width
            y_max_norm = y_max / height
            f.write(f"{cls} {x_min_norm:.6f} {y_min_norm:.6f} {x_max_norm:.6f} {y_max_norm:.6f} {score:.6f}\n")

    print(f"Saved {len(pred_boxes)} predictions for {image_filename}")

print(f"\nMask R-CNN Batch Inference Complete! Predictions saved in {output_folder}")
