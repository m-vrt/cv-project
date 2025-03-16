import os
import numpy as np
from sklearn.metrics import auc
from glob import glob


ground_truth_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\mask_rcnn\annotations"
predictions_folder = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\mask_rcnn\predictions"
output_file = r"C:\Users\Donna\Documents\m-vrt\cv-project\bdd100k\mask_rcnn\mask_rcnn_evaluation_results.txt"

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
CONFIDENCE_THRESHOLD = 0.10  
IOU_MATCH_THRESHOLD = 0.45  


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


def load_boxes(txt_file, is_prediction=False):
    """Load bounding boxes from Mask R-CNN predictions or ground truth annotations."""
    boxes = []
    if os.path.exists(txt_file):
        with open(txt_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_min, y_min, x_max, y_max = map(float, parts[1:5])
                    if is_prediction:
                        conf = float(parts[5])
                        if conf >= CONFIDENCE_THRESHOLD:
                            boxes.append((class_id, x_min, y_min, x_max, y_max, conf))
                    else:
                        boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes


y_true, y_pred = [], []
ious, mAP_values = [], []
total_tp, total_fp, total_fn = 0, 0, 0

for gt_file in glob(os.path.join(ground_truth_folder, "*.txt")):
    filename = os.path.basename(gt_file)
    pred_file = os.path.join(predictions_folder, filename)

    gt_boxes = load_boxes(gt_file)
    pred_boxes = load_boxes(pred_file, is_prediction=True)

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
        fp_thresh = total_fp
        fn_thresh = total_fn
        
        precision_at_iou = tp_thresh / (tp_thresh + fp_thresh + 1e-6)
        recall_at_iou = tp_thresh / (tp_thresh + fn_thresh + 1e-6)
        
        recall_points = np.linspace(0, 1, 11)
        precision_interpolated = [precision_at_iou for _ in recall_points]
        
        ap = auc(recall_points, precision_interpolated)
        aps.append(ap)
    aps = np.maximum.accumulate(sorted(aps, reverse=True))  
    
    mAP_values.append(np.mean(aps))


precision = total_tp / (total_tp + total_fp + 1e-6)
recall = total_tp / (total_tp + total_fn + 1e-6)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)


with open(output_file, "w", encoding="utf-8") as f:
    f.write("Mask R-CNN Evaluation Results:\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Mean IoU: {np.mean(ious):.4f}\n")
    f.write(f"mAP: {np.mean(mAP_values):.4f}\n")
    f.write(f"\nTotal True Positives: {total_tp}\n")
    f.write(f"Total False Positives: {total_fp}\n")
    f.write(f"Total False Negatives: {total_fn}\n")

print(f"Evaluation complete! Results saved to: {output_file}")
