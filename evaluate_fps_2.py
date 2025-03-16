import cv2
import time
import torch
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

       
        if model_name == "Mask R-CNN":
            outputs = predictor(frame)

        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count += 1

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"{model_name} FPS: {avg_fps:.2f}")
    cap.release()


cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = r"C:\Users\Donna\Documents\m-vrt\cv-project\detectron_models\model_final_f10217.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mask_rcnn_predictor = DefaultPredictor(cfg)


measure_fps("Mask R-CNN", mask_rcnn_predictor)
