import numpy as np

def non_max_suppression(boxes, scores, threshold):
    """
    Perform non-maximum suppression on bounding boxes.

    Parameters:
    - boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
    - scores: List of confidence scores corresponding to each box
    - threshold: Overlapping threshold to suppress boxes

    Returns:
    - List of selected bounding boxes after non-maximum suppression
    """

    if len(boxes) == 0:
        return [], []

    # Convert the list of boxes to NumPy array for easier manipulation
    boxes = np.array(boxes)

    # Initialize the list to store selected boxes after NMS
    selected_boxes = []
    selected_scores = []

    # Sort the boxes based on their confidence scores (in descending order)
    order = np.argsort(scores)[::-1]

    while len(order) > 0:
        # Select the box with the highest confidence score
        selected_box = boxes[order[0]]
        selected_score = scores[order[0]]

        # Add the selected box to the final list
        selected_boxes.append(selected_box)
        selected_scores.append(selected_score)

        # Calculate the intersection over union (IoU) with all other boxes
        iou = calculate_iou(selected_box, boxes[order[1:]])

        # Keep only the boxes with IoU less than the threshold
        order = order[1:][iou < threshold]

    return selected_boxes, selected_scores


def calculate_iou(box, boxes):
    """
    Calculate the intersection over union (IoU) between a box and a list of boxes.

    Parameters:
    - box: The reference bounding box (x1, y1, x2, y2)
    - boxes: List of bounding boxes [(x1, y1, x2, y2), ...]

    Returns:
    - NumPy array of IoU values between the reference box and each box in the list
    """

    # Calculate the intersection coordinates
    x1_int = np.maximum(box[0], boxes[:, 0])
    y1_int = np.maximum(box[1], boxes[:, 1])
    x2_int = np.minimum(box[2], boxes[:, 2])
    y2_int = np.minimum(box[3], boxes[:, 3])

    # Calculate the area of intersection
    intersection_area = np.maximum(0, x2_int - x1_int + 1) * np.maximum(0, y2_int - y1_int + 1)

    # Calculate the area of the reference box and boxes in the list
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # Calculate the union area
    union_area = box_area + boxes_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou
