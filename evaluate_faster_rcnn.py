import os
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import xml.etree.ElementTree as ET


GROUND_TRUTH_PATH = "C:/Users/Donna/Documents/m-vrt/cv-project/ground_truth/faster_rcnn"
PREDICTIONS_PATH = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/faster_rcnn"


IOU_THRESHOLD = 0.5  
MAX_IMAGES = 300  


gt_filenames = set(f.split(".")[0] for f in os.listdir(GROUND_TRUTH_PATH) if f.endswith(".xml"))
pred_filenames = set(f.split(".")[0] for f in os.listdir(PREDICTIONS_PATH) if f.endswith(".xml"))

matching_filenames = sorted(list(gt_filenames & pred_filenames))[:MAX_IMAGES]

def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(boxA_area + boxB_area - inter_area) if boxA_area + boxB_area - inter_area > 0 else 0

def load_voc_annotations(folder):
    """Loads Pascal VOC .xml annotations and extracts bounding boxes."""
    annotations = {}
    for frame_id in matching_filenames:
        file_path = os.path.join(folder, frame_id + ".xml")
        if not os.path.exists(file_path):
            continue

        tree = ET.parse(file_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            boxes.append([
                float(bbox.find("xmin").text),
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
            ])
        annotations[frame_id] = boxes
    return annotations

def evaluate_faster_rcnn(gt_folder, pred_folder, log_file_path):
    """Evaluates Faster R-CNN using IoU-matched Precision, Recall, F1 Score, and mAP."""
    print("\nEvaluating Faster R-CNN...")
    start_time = time.time()

    ground_truth = load_voc_annotations(gt_folder)
    predictions = load_voc_annotations(pred_folder)

    y_true, y_pred = [], []
    total_gt_objects, total_pred_objects, false_positives = 0, 0, 0

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Faster R-CNN Evaluation Results\n")
        log_file.write("=" * 50 + "\n")

        for frame_id in matching_filenames:
            gt_boxes = ground_truth.get(frame_id, [])
            pred_boxes = predictions.get(frame_id, [])
            total_gt_objects += len(gt_boxes)
            total_pred_objects += len(pred_boxes)

            unmatched_preds = len(pred_boxes)
            
            for pred in pred_boxes:
                best_iou = max([iou(gt, pred) for gt in gt_boxes] or [0])
                if best_iou >= IOU_THRESHOLD:
                    unmatched_preds -= 1
                    y_true.append(1)  
                    y_pred.append(1)
                else:
                    y_true.append(0)  
                    y_pred.append(1)

            false_positives += unmatched_preds
            
            log_file.write(f"Frame: {frame_id} | GT: {len(gt_boxes)} | Predictions: {len(pred_boxes)} | False Positives: {unmatched_preds}\n")

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        try:
            average_precision = average_precision_score(y_true, y_pred)
        except:
            average_precision = 0

        total_time = time.time() - start_time
        processing_time_per_frame = (total_time / MAX_IMAGES) * 1000

        log_file.write("\nFinal Metrics:\n")
        log_file.write(f"Total GT Objects: {total_gt_objects}\n")
        log_file.write(f"Total Predicted Objects: {total_pred_objects}\n")
        log_file.write(f"Total False Positives: {false_positives}\n")
        log_file.write(f"Mean Precision: {precision:.4f}\n")
        log_file.write(f"Mean Recall: {recall:.4f}\n")
        log_file.write(f"Mean F1 Score: {f1:.4f}\n")
        log_file.write(f"mAP: {average_precision:.4f}\n")
        log_file.write(f"Processing Time per Frame: {processing_time_per_frame:.2f} ms\n")
        log_file.write(f"Total Evaluation Time: {total_time:.2f} s\n\n")

    print(f"\nFaster R-CNN Evaluation Complete. Results saved in `{log_file_path}`.")

log_file_path = "C:/Users/Donna/Documents/m-vrt/cv-project/evaluation_faster_rcnn.txt"
evaluate_faster_rcnn(GROUND_TRUTH_PATH, PREDICTIONS_PATH, log_file_path)
