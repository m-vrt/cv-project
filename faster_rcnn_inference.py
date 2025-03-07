import cv2
import torch
import os
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  
cfg.MODEL.DEVICE = "cuda"  

predictor = DefaultPredictor(cfg)


video_path = "C:/Users/Donna/Documents/m-vrt/cv-project/output_faster_rcnn_trimmed.mp4" 
predictions_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/faster_rcnn"
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
    frame_filename = f"frame_{frame_count:04d}.xml"
    frame_output_path = os.path.join(predictions_folder, frame_filename)

    
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")

    
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = frame_filename

    for i in range(len(instances.pred_classes)):
        obj = ET.SubElement(annotation, "object")
        class_id = int(instances.pred_classes[i])
        bbox = instances.pred_boxes.tensor[i].numpy()

        ET.SubElement(obj, "name").text = str(class_id)

       
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

    tree = ET.ElementTree(annotation)
    tree.write(frame_output_path)

cap.release()
print(f"Inference complete. Predictions saved in: {predictions_folder}")
