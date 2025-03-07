import os


kitti_labels_path = "C:/Users/Donna/Documents/m-vrt/cv-project/kitti/label_2/training/label_2"
output_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/ground_truth/yolov8"
os.makedirs(output_folder, exist_ok=True)


class_mapping = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}  


for filename in os.listdir(kitti_labels_path):
    if filename.endswith(".txt"):
        input_path = os.path.join(kitti_labels_path, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
            for line in f_in:
                data = line.strip().split()
                class_name = data[0]
                if class_name in class_mapping:
                    class_id = class_mapping[class_name]
                    x1, y1, x2, y2 = map(float, data[4:8])  
                    f_out.write(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")

print(f"Conversion complete! YOLO annotations saved in: {output_folder}")
