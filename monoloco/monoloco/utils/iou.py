
import numpy as np


def calculate_iou(box1, box2):

    # Calculate the (x1, y1, x2, y2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)  # Max keeps into account not overlapping box

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


def get_iou_matrix(boxes, boxes_gt):
    """
    Get IoU matrix between predicted and ground truth boxes
    Dim: (boxes, boxes_gt)
    """
    iou_matrix = np.zeros((len(boxes), len(boxes_gt)))
    for idx, box in enumerate(boxes):
        for idx_gt, box_gt in enumerate(boxes_gt):
            iou_matrix[idx, idx_gt] = calculate_iou(box, box_gt)
    return iou_matrix


def get_iou_matches(boxes, boxes_gt, thresh):
    """From 2 sets of boxes and a minimum threshold, compute the matching indices for IoU matchings"""

    iou_matrix = get_iou_matrix(boxes, boxes_gt)
    if not iou_matrix.size:
        return []

    matches = []
    iou_max = np.max(iou_matrix)
    while iou_max > thresh:
        # Extract the indeces of the max
        args_max = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        matches.append(args_max)
        iou_matrix[args_max[0], :] = 0
        iou_matrix[:, args_max[1]] = 0
        iou_max = np.max(iou_matrix)
    return matches


def reorder_matches(matches, boxes, mode='left_rigth'):
    """
    Reorder a list of (idx, idx_gt) matches based on position of the detections in the image
    ordered_boxes = (5, 6, 7, 0, 1, 4, 2, 4)
    matches = [(0, x), (2,x), (4,x), (3,x), (5,x)]
    Output --> [(5, x), (0, x), (3, x), (2, x), (5, x)]
    """

    assert mode == 'left_right'

    # Order the boxes based on the left-right position in the image and
    ordered_boxes = np.argsort([box[0] for box in boxes])  # indices of boxes ordered from left to right
    matches_left = [idx for (idx, _) in matches]

    return [matches[matches_left.index(idx_boxes)] for idx_boxes in ordered_boxes if idx_boxes in matches_left]
