import cv2
import time
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

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

        if model_name == "SAM":
            frame_rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 1024))  
            frame_rgb = np.moveaxis(frame_rgb, -1, 0) 
            frame_tensor = torch.from_numpy(frame_rgb).float().to("cuda")  

            predictor.set_image(frame_tensor)  

            masks, _, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=False)

        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count += 1

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"{model_name} FPS: {avg_fps:.2f}")
    cap.release()


sam = sam_model_registry["vit_b"](checkpoint="segment_anything/sam_model.pth").to("cuda")
sam_predictor = SamPredictor(sam)


measure_fps("SAM", sam_predictor)
