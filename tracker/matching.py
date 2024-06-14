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


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = nb_box_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious



def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack] tracks
    :type btracks: list[STrack] new detections

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
