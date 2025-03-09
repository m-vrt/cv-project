import cv2
import torch
import os
import numpy as np
import json
import warnings
from segment_anything import sam_model_registry, SamPredictor


warnings.filterwarnings("ignore")


kitti_images_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/kitti/image_2/testing/image_2"
predictions_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/sam"
os.makedirs(predictions_folder, exist_ok=True)


MAX_IMAGES = 300
image_files = sorted(os.listdir(kitti_images_folder))[:MAX_IMAGES]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM model on {device}...")

try:
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint="segment_anything/sam_model.pth").to(device)
    predictor = SamPredictor(sam)
    print(f"SAM Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading SAM model: {e}")
    exit()


processed_images = 0
for i, image_filename in enumerate(image_files):
    try:
        if not image_filename.endswith(".png"):
            continue

        image_path = os.path.join(kitti_images_folder, image_filename)
        print(f"\n🔹 [{i+1}/{MAX_IMAGES}] Processing: {image_filename}")

      
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipped unreadable image: {image_filename}")
            continue

        
        original_shape = image.shape[:2]
        target_size = (640, 480)  
        resized_image = cv2.resize(image, target_size)
        print(f"Resized from {original_shape} to {target_size}")

        
        frame_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)

        
        masks, _, _ = predictor.predict(multimask_output=True)
        print(f"{len(masks)} mask(s) detected.")

        objects = []
        for mask in masks:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                
                if w * h > 100:  
                    bbox = [x, y, x + w, y + h]
                    objects.append({"class_id": 1, "bbox": bbox})

        
        json_filename = os.path.splitext(image_filename)[0] + ".json"
        json_output_path = os.path.join(predictions_folder, json_filename)

        with open(json_output_path, "w") as f:
            json.dump({"frame": json_filename, "objects": objects}, f, indent=4)

        print(f"JSON saved: {json_filename} ({len(objects)} objects detected)")

        
        del frame_rgb, resized_image, masks
        torch.cuda.empty_cache()

        processed_images += 1

    except Exception as e:
        print(f"Error processing {image_filename}: {e}")
        continue  

print(f"\nSAM inference completed on {processed_images}/{MAX_IMAGES} images. Predictions saved to {predictions_folder}")
