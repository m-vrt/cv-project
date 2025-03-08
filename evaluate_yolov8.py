import os
import numpy as np
import time
import warnings
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


GROUND_TRUTH_PATH = "C:/Users/Donna/Documents/m-vrt/cv-project/ground_truth/yolov8"
PREDICTIONS_PATH = "C:/Users/Donna/Documents/m-vrt/cv-project/predictions/yolov8"


IOU_THRESHOLD = 0.5  
MAX_IMAGES = 300  


image_filenames = sorted(os.listdir(GROUND_TRUTH_PATH))[:MAX_IMAGES]
image_filenames = [os.path.splitext(f)[0] for f in image_filenames]  


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


def load_yolo_annotations(folder):
    """Loads YOLO .txt annotations and converts to [xmin, ymin, xmax, ymax]."""
    annotations = {}
    for frame_id in image_filenames:
        file_path = os.path.join(folder, frame_id + ".txt")
        if not os.path.exists(file_path):
            continue  

        with open(file_path, "r") as f:
            lines = [list(map(float, line.strip().split())) for line in f.readlines() if line.strip()]
            boxes = [convert_yolo_bbox(line[1:5]) for line in lines]  
            annotations[frame_id] = boxes

    return annotations


def convert_yolo_bbox(yolo_bbox):
    """Convert YOLO format (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax)."""
    x_center, y_center, width, height = yolo_bbox
    xmin = x_center - (width / 2)
    ymin = y_center - (height / 2)
    xmax = x_center + (width / 2)
    ymax = y_center + (height / 2)
    return [xmin, ymin, xmax, ymax]


def evaluate_yolov8(gt_folder, pred_folder, log_file_path):
    """Evaluates YOLOv8 using IoU-matched Precision, Recall, F1 Score, and mAP."""
    print("\n🔍 Evaluating YOLOv8...")

    start_time = time.time()

    
    ground_truth = load_yolo_annotations(gt_folder)
    predictions = load_yolo_annotations(pred_folder)

    y_true, y_pred = [], []
    total_gt_objects = 0  
    total_pred_objects = 0  

    for frame_id in image_filenames:
        gt_boxes = ground_truth.get(frame_id, [])
        pred_boxes = predictions.get(frame_id, [])

        total_gt_objects += len(gt_boxes)
        total_pred_objects += len(pred_boxes)

        if not gt_boxes and not pred_boxes:
            continue  

        if not gt_boxes:
            
            y_true.extend([0] * len(pred_boxes))
            y_pred.extend([1] * len(pred_boxes))
            continue

        if not pred_boxes:
            
            y_true.extend([1] * len(gt_boxes))
            y_pred.extend([0] * len(gt_boxes))
            continue

        
        iou_matches = [max([iou(gt, pred) for gt in gt_boxes] or [0]) for pred in pred_boxes]
        tp = [1 if i >= IOU_THRESHOLD else 0 for i in iou_matches]
        fp = [1 - x for x in tp]

        y_true.extend([1] * len(gt_boxes))  
        y_pred.extend(tp + fp)  

    
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    
    try:
        average_precision = average_precision_score(y_true, y_pred)
    except:
        average_precision = 0  

    total_time = time.time() - start_time
    processing_time_per_frame = (total_time / MAX_IMAGES) * 1000  

    
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("YOLOv8 Evaluation Results\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Total GT Objects: {total_gt_objects}\n")
        log_file.write(f"Total Predicted Objects: {total_pred_objects}\n")
        log_file.write(f"Mean Precision: {precision:.4f}\n")
        log_file.write(f"Mean Recall: {recall:.4f}\n")
        log_file.write(f"Mean F1 Score: {f1:.4f}\n")
        log_file.write(f"mAP: {average_precision:.4f}\n")
        log_file.write(f"Processing Time per Frame: {processing_time_per_frame:.2f} ms\n")
        log_file.write(f"Total Evaluation Time: {total_time:.2f} s\n\n")

    print(f"\n YOLOv8 Evaluation Complete. Results saved in `{log_file_path}`.")



log_file_path = "C:/Users/Donna/Documents/m-vrt/cv-project/evaluation_yolov8.txt"
evaluate_yolov8(GROUND_TRUTH_PATH, PREDICTIONS_PATH, log_file_path)
