import os
import xml.etree.ElementTree as ET


kitti_labels_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/kitti/label_2/training/label_2"  
images_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/kitti/image_2/training/image_2"  
voc_output_folder = "C:/Users/Donna/Documents/m-vrt/cv-project/ground_truth/faster_rcnn"  


os.makedirs(voc_output_folder, exist_ok=True)


class_mapping = {
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Pedestrian": "person",
    "Person_sitting": "person",
    "Cyclist": "bicycle",
    "Tram": "train",
    "Misc": "misc",
    "DontCare": "ignore"
}

def convert_kitti_to_voc():
    for txt_file in os.listdir(kitti_labels_folder):
        if not txt_file.endswith(".txt"):
            continue

        frame_id = os.path.splitext(txt_file)[0] 
        image_filename = frame_id + ".png"
        image_path = os.path.join(images_folder, image_filename)
        label_path = os.path.join(kitti_labels_folder, txt_file)
        xml_output_path = os.path.join(voc_output_folder, frame_id + ".xml")

        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_filename} not found, skipping.")
            continue

       
        objects = []
        with open(label_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            class_name = parts[0]
            if class_name not in class_mapping or class_mapping[class_name] == "ignore":
                continue  

            class_label = class_mapping[class_name]

           
            x1, y1, x2, y2 = map(float, parts[4:8])

            objects.append({
                "name": class_label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        if not objects:
            print(f"Warning: No valid objects in {txt_file}, skipping.")
            continue

        
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "KITTI"
        ET.SubElement(annotation, "filename").text = image_filename
        ET.SubElement(annotation, "path").text = image_path

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "KITTI"

        for obj in objects:
            obj_elem = ET.SubElement(annotation, "object")
            ET.SubElement(obj_elem, "name").text = obj["name"]

            bndbox = ET.SubElement(obj_elem, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(obj["bbox"][0])
            ET.SubElement(bndbox, "ymin").text = str(obj["bbox"][1])
            ET.SubElement(bndbox, "xmax").text = str(obj["bbox"][2])
            ET.SubElement(bndbox, "ymax").text = str(obj["bbox"][3])

        
        tree = ET.ElementTree(annotation)
        tree.write(xml_output_path)

    print(f"Conversion complete! Pascal VOC XML annotations saved in: {voc_output_folder}")

if __name__ == "__main__":
    convert_kitti_to_voc()
