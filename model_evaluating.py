import numpy as np
from sklearn.metrics import average_precision_score
import cv2
import tensorflow

# Define a function to calculate Intersection over Union (IoU)


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = intersection / float(boxA_area + boxB_area - intersection)
    return iou

# Define a function to calculate Precision, Recall, and F1-score


def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_box in pred_boxes:
        iou_max = 0
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_max:
                iou_max = iou
        if iou_max >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(gt_boxes) - true_positives

    precision = true_positives / max(true_positives + false_positives, 1e-6)
    recall = true_positives / max(true_positives + false_negatives, 1e-6)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)

    return precision, recall, f1_score

# Define a function to calculate Mean Average Precision (mAP)


def calculate_map(gt_boxes, pred_boxes, iou_threshold=0.5):
    average_precision = 0
    for class_id in range(len(gt_boxes)):
        y_true = np.zeros(len(gt_boxes[class_id]))
        y_scores = np.zeros(len(pred_boxes[class_id]))

        for i in range(len(pred_boxes[class_id])):
            # pred_box = pred_boxes[class_id][i]
            # iou_scores = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes[class_id]]
            iou_scores = [calculate_iou(pred_boxes[class_id], gt_boxes[class_id])]
            max_iou = max(iou_scores)
            if max_iou >= iou_threshold:
                y_true[i] = 1
            y_scores[i] = max_iou

        average_precision += average_precision_score(y_true, y_scores)

    mAP = average_precision / max(len(gt_boxes), 1e-6)
    return mAP

# Example usage:


if __name__ == "__main__":
    pass
    # Load ground truth and predicted bounding boxes for each class
    # gt_boxes = {
    #     0: [[x1, y1, x2, y2], [x1, y1, x2, y2]],
    #     1: [[x1, y1, x2, y2], [x1, y1, x2, y2]],
    # }
    #
    # pred_boxes = {
    #     0: [[x1, y1, x2, y2], [x1, y1, x2, y2]],
    #     1: [[x1, y1, x2, y2], [x1, y1, x2, y2]],
    # }

    # iou_threshold = 0.5
    #
    # # Calculate precision, recall, and F1-score for each class
    # for class_id in gt_boxes.keys():
    #     precision, recall, f1_score = calculate_precision_recall(gt_boxes[class_id], pred_boxes[class_id], iou_threshold)
    #     print(f"Class {class_id} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}")
    #
    # # Calculate Mean Average Precision (mAP)
    # mAP = calculate_map(gt_boxes, pred_boxes, iou_threshold)
    # print(f"mAP: {mAP:.2f}")
