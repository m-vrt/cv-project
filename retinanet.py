import cv2
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")


video_path = "for cv-project.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


resize_width, resize_height = 640, 360
frame_skip = max(1, fps // 10) 


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4  
cfg.MODEL.DEVICE = "cuda"  

predictor = DefaultPredictor(cfg)


output_path = "output_retinanet.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, max(1, fps // frame_skip), (resize_width, resize_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    
    if frame_count % frame_skip != 0:
        continue

  
    frame = cv2.resize(frame, (resize_width, resize_height))


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
    outputs = predictor(frame_rgb)

  
    v = Visualizer(frame_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

   
    processed_frame = cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR)

   
    out.write(processed_frame)

   
    cv2.imshow("Object Detection", processed_frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_path)