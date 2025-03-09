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

output_width, output_height = frame_width, frame_height
output_path = "output_sam.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predictor.set_image(frame_rgb)

  
    input_boxes = np.array([[50, 50, frame_width - 50, frame_height - 50]])
    masks, _, _ = predictor.predict(box=input_boxes, multimask_output=True)

    if masks is not None and len(masks) > 0:
        for mask in masks:
            mask = mask.astype(np.uint8) * 255

           
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            
            frame = cv2.addWeighted(frame, 0.85, colored_mask, 0.15, 0)

    out.write(frame)
    cv2.imshow("Segment Anything Model (SAM)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to: {output_path}")
