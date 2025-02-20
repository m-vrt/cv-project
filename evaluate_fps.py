import cv2
import time
import torch
from detectron2 import model_zoo
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

video_path = "for cv-project.mp4"

def measure_fps(model_name, predictor):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        if model_name == "YOLOv8-Seg":
            results = predictor(frame)
        elif model_name == "Faster R-CNN":
            outputs = predictor(frame)

        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count += 1

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"{model_name} FPS: {avg_fps:.2f}")
    cap.release()



yolo_model = YOLO("yolov8n-seg.pt")
measure_fps("YOLOv8-Seg", yolo_model)


cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"
faster_rcnn_predictor = DefaultPredictor(cfg)
measure_fps("Faster R-CNN", faster_rcnn_predictor)
