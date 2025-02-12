import cv2
import torch
import detectron2
from detectron2 import model_zoo  
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


video_path = "for cv-project.MOV"  
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  
cfg.MODEL.DEVICE = "cpu" 

predictor = DefaultPredictor(cfg)


frame_count = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    
    if frame_count % 3 != 0:
        continue

    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  
    outputs = predictor(frame_rgb)

  
    v = Visualizer(frame_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

  
    processed_frame = cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR)

    
    cv2.imshow("Object Detection", processed_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
