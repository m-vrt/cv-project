import os
import numpy as np


source_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\yolo\test\labels"
destination_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\yolo\annotations"


os.makedirs(destination_folder, exist_ok=True)

def convert_to_bounding_box(yolo_segmentation_file, output_file):
    """
    Convert YOLO segmentation masks to bounding box format.
    """
    with open(yolo_segmentation_file, "r") as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue  

        class_id = parts[0]
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)  

       
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)

        
        width, height = 1280, 720  
        x_min, y_min = x_min * width, y_min * height
        x_max, y_max = x_max * width, y_max * height

        
        converted_lines.append(f"{class_id} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")

    
    with open(output_file, "w") as f_out:
        f_out.writelines(converted_lines)


for filename in os.listdir(source_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(source_folder, filename)
        output_path = os.path.join(destination_folder, filename)
        convert_to_bounding_box(input_path, output_path)

print(f"Ground truth conversion complete! Converted files saved in {destination_folder}")
