import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import njit

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []


    n_rows = cost_matrix.shape[0]
    n_cols = cost_matrix.shape[1]

    n = n_rows + n_cols
    cost_c_extended = np.empty((n, n), dtype=np.double)


    cost_c_extended[:] = thresh / 2.


    cost_c_extended[n_rows:, n_cols:] = 0
    cost_c_extended[:n_rows, :n_cols] = cost_matrix


    _, x = linear_sum_assignment(cost_c_extended)
    y = np.ones(len(x)) * -1
    for i, j in enumerate(x):
        y[j] = i
    x = np.where(x < n_cols, x, -1)[:n_rows]
    y = np.where(y < n_rows, y, -1)[:n_cols]


    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b




@njit
def nb_box_ious(
        boxes,
        query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=float)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = (boxes[n, 2] - boxes[n, 0] + 1) *(boxes[n, 3] - boxes[n, 1] + 1) +box_area - iw * ih

                    overlaps[n, k] = iw * ih / ua
    return overlaps


def calculate_ious(a_bounding_boxes, b_bounding_boxes):
    iou_matrix = np.zeros((len(a_bounding_boxes), len(b_bounding_boxes)), dtype=float)
    if iou_matrix.size == 0:
        return iou_matrix

    iou_matrix = nb_box_ious(
        np.ascontiguousarray(a_bounding_boxes, dtype=float),
        np.ascontiguousarray(b_bounding_boxes, dtype=float)
    )

    return iou_matrix


def iou_distance(a_tracks, b_tracks):
    if (len(a_tracks) > 0 and isinstance(a_tracks[0], np.ndarray)) or (len(b_tracks) > 0 and isinstance(b_tracks[0], np.ndarray)):
        a_bounding_boxes = a_tracks
        b_bounding_boxes = b_tracks
    else:
        a_bounding_boxes = [track.tlbr for track in a_tracks]
        b_bounding_boxes = [track.tlbr for track in b_tracks]

    iou_matrix = calculate_ious(a_bounding_boxes, b_bounding_boxes)
    cost_matrix = 1 - iou_matrix

    return cost_matrix
