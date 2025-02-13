import cv2
import torch
import numpy as np
import warnings
from segment_anything import sam_model_registry, SamPredictor


warnings.filterwarnings("ignore")


model_type = "vit_b"  
sam = sam_model_registry[model_type](checkpoint="segment_anything/sam_model.pth")
sam.to("cuda")  
predictor = SamPredictor(sam)


video_path = "for cv-project.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

resize_width, resize_height = frame_width // 2, frame_height // 2
frame_skip = max(1, fps // 10) 


output_path = "output_sam.mp4"
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

  
    predictor.set_image(frame_rgb)

    
    input_point = np.array([[frame_width // 2, frame_height // 2]])  
    input_label = np.array([1])  
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

   
    final_mask = np.zeros_like(frame_rgb, dtype=np.uint8)
    
    for mask in masks:
        mask = mask.astype(np.uint8) * 255  
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  
        final_mask = cv2.addWeighted(final_mask, 1.0, colored_mask, 0.5, 0)

    blended = cv2.addWeighted(frame, 0.7, final_mask, 0.3, 0)

    
    out.write(blended)
    cv2.imshow("Segment Anything Model", blended)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_path)
