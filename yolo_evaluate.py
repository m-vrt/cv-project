import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from glob import glob


ground_truth_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\yolo\annotations"
predictions_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\yolo\predictions"
output_file = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\yolo\yolo_evaluation_results.txt"


fps_values = {
    "YOLOv8-Seg": 43.68,
    "Faster R-CNN": 4.40,
    "SAM": 1.26
}


IOU_THRESHOLDS = [0.5]  


CONFIDENCE_THRESHOLD = 0.15  
IOU_MATCH_THRESHOLD = 0.5  


def normalize_to_absolute(box, img_width=1280, img_height=720):
    """Converts YOLO normalized coordinates (relative) to absolute pixel values."""
    x_center, y_center, w, h = box
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    return x1, y1, x2, y2


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_yolo_boxes(txt_file, is_prediction=False):
    """Load YOLO bounding boxes and convert normalized values to absolute pixel values."""
    boxes = []
    if os.path.exists(txt_file):
        with open(txt_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if is_prediction:
                        x_center, y_center, w, h, conf = map(float, parts[1:])
                        if conf >= CONFIDENCE_THRESHOLD:
                            abs_box = normalize_to_absolute((x_center, y_center, w, h))
                            boxes.append((class_id, *abs_box, conf))
                    else:
                        x_center, y_center, w, h = map(float, parts[1:])
                        abs_box = normalize_to_absolute((x_center, y_center, w, h))
                        boxes.append((class_id, *abs_box))
    return boxes


y_true, y_pred = [], []
ious, mAP_values = [], []
total_tp, total_fp, total_fn = 0, 0, 0

for gt_file in glob(os.path.join(ground_truth_folder, "*.txt")):
    filename = os.path.basename(gt_file)
    pred_file = os.path.join(predictions_folder, filename)

    gt_boxes = load_yolo_boxes(gt_file)
    pred_boxes = load_yolo_boxes(pred_file, is_prediction=True)

    if len(gt_boxes) == 0:
        continue  

    matched = set()
    tp, fp, fn = 0, 0, 0

    for pred_box in pred_boxes:
        best_iou, best_match = 0, None
        for gt_box in gt_boxes:
            iou = compute_iou(pred_box[1:5], gt_box[1:5])
            if iou > best_iou:
                best_iou, best_match = iou, gt_box

        if best_iou >= IOU_MATCH_THRESHOLD:
            tp += 1
            matched.add(best_match)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)
    total_tp += tp
    total_fp += fp
    total_fn += fn

   
    aps = []
    for iou_thresh in IOU_THRESHOLDS:
        tp_thresh = sum(1 for i in ious if i >= iou_thresh)
        ap = tp_thresh / (tp_thresh + fp + fn + 1e-6)
        aps.append(ap)
    mAP_values.append(np.mean(aps))


precision = total_tp / (total_tp + total_fp + 1e-6)
recall = total_tp / (total_tp + total_fn + 1e-6)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)


with open(output_file, "w", encoding="utf-8") as f:
    f.write("YOLOv8 Evaluation Results:\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Mean IoU: {np.mean(ious):.4f}\n")
    f.write(f"mAP: {np.mean(mAP_values):.4f}\n")
    f.write(f"\nTotal True Positives: {total_tp}\n")
    f.write(f"Total False Positives: {total_fp}\n")
    f.write(f"Total False Negatives: {total_fn}\n")

print(f"Evaluation complete! Results saved to: {output_file}")