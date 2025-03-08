import cv2
import torch
import os
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  


if torch.cuda.is_available():
    free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    if free_memory < 500 * 1024 * 1024:  
        print("Warning: Low GPU memory. Switching to CPU mode.")
        cfg.MODEL.DEVICE = "cpu"
    else:
        cfg.MODEL.DEVICE = "cuda"
else:
    cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


kitti_images_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/kitti/image_2/testing/image_2"
predictions_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/faster_rcnn"
os.makedirs(predictions_folder, exist_ok=True)


MAX_IMAGES = 300
image_files = sorted(os.listdir(kitti_images_folder))[:MAX_IMAGES]  

for image_filename in image_files:
    if image_filename.endswith(".png"):
        image_path = os.path.join(kitti_images_folder, image_filename)
        image = cv2.imread(image_path)

        
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        
        xml_filename = os.path.splitext(image_filename)[0] + ".xml"
        xml_output_path = os.path.join(predictions_folder, xml_filename)

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = image_filename

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
        tree.write(xml_output_path)

       
        torch.cuda.empty_cache()

print(f"Inference complete on {MAX_IMAGES} images. Predictions saved in: {predictions_folder}")
