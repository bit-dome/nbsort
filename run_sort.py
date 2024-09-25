import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from nms import non_max_suppression
from tracker.sort import SORT
from tracker.tracking_box import TrackingBox

# Expand crop region, this should help the Lens to be more accurate?
CROP_EXPAND = 1.2

VIDEO_PATH = 'toy_store.mp4'
RAID_JSON_DIR = 'raid_results'
RAID_INVALID_CLASSES = ["Person"]
RAID_THRESHOLD = 0.1



color_pallet = [[165, 194, 102], [98, 141, 252], [203, 160, 141], [195, 138, 231], [84, 216, 166], [47, 217, 255], [148, 196, 229], [179, 179, 179]]


raid_result_paths = Path(RAID_JSON_DIR).glob("*.json")
raid_result_paths = list(raid_result_paths)
raid_result_paths.sort()
track_history = defaultdict(lambda: [])


objedt_track_sharpness = {}


def raid_detection(frame_i: int):
    """Please Replace this with real RAID api.
    """
    with open(raid_result_paths[frame_i], 'r') as f:
        detections = json.load(f)

    return detections


def filter_detections(detections: list[dict]):
    boxes = []
    scores = []
    for det in detections:
        if det["display_object_name"] in RAID_INVALID_CLASSES or det[
                "label_score"] < RAID_THRESHOLD:
            continue

        x1 = int(det["region"]["bounding_box"]["xmin"] *
                 det["region"]["image_width"])
        y1 = int(det["region"]["bounding_box"]["ymin"] *
                 det["region"]["image_height"])
        x2 = int(det["region"]["bounding_box"]["xmax"] *
                 det["region"]["image_width"])
        y2 = int(det["region"]["bounding_box"]["ymax"] *
                 det["region"]["image_height"])

        boxes.append([x1, y1, x2, y2])
        scores.append(det["label_score"])

    selected_boxes, selected_scores = non_max_suppression(boxes, scores, 0.4)
    return selected_boxes, selected_scores


def write_cropped_image(frame, x1, y1, x2, y2, track_id: int):
    """Write cropped image to output folder.
    """
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    width = x2 - x1
    height = y2 - y1
    new_x1 = max(0, int(x1 - (width * (CROP_EXPAND - 1) / 2)))
    new_y1 = max(0, int(y1 - (height * (CROP_EXPAND - 1) / 2)))
    new_x2 = min(frame_width, int(x2 + (width * (CROP_EXPAND - 1) / 2)))
    new_y2 = min(frame_height, int(y2 + (height * (CROP_EXPAND - 1) / 2)))

    object_im = frame[new_y1:new_y2, new_x1:new_x2]

    try:
        cv2.imwrite(f"output/track_id_{track_id}.jpg", object_im)
    except Exception as e:
        #print(e)
        pass


def save_best_image(raw_frame, annotated_frame, online_targets: list[TrackingBox],
                    frame_sharpness: float):
    """Check image sharpness value and save cropped to output folder if it better than the previous one.

    Args:
        raw_frame (_type_): Frame before crop
        annotated_frame (_type_): Frame for annotation only
        online_targets (list[STrack]): Currently tracking objects from tracker
        frame_sharpness (float): sharpness value
    """
    online_tlwhs = []
    online_ids = []
    online_scores = []

    for t in online_targets:

        tlwh = t.tlwh.astype(int)
        track_id = t.track_id
        online_tlwhs.append(tlwh)
        online_ids.append(track_id)
        online_scores.append(t.score)

        x1 = int(tlwh[0])
        y1 = int(tlwh[1])
        x2 = int(tlwh[0] + tlwh[2])
        y2 = int(tlwh[1] + tlwh[3])

        cx, cy = tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2

        is_best_shot = False

        # If already in dict, compare the sharpness.
        if track_id in objedt_track_sharpness:
            if objedt_track_sharpness[track_id] < frame_sharpness:
                objedt_track_sharpness[track_id] = frame_sharpness
                write_cropped_image(raw_frame, x1, y1, x2, y2, track_id)
                is_best_shot = True

        # Is first crop.
        else:
            objedt_track_sharpness[track_id] = frame_sharpness
            write_cropped_image(raw_frame, x1, y1, x2, y2, track_id)
            is_best_shot = True

        if is_best_shot:
            cv2.rectangle(annotated_frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3),
                          (102, 255, 178), -1)

        color = color_pallet[track_id % len(color_pallet)]
        track_history[track_id].append((float(cx), float(cy)))

        if len(track_history[track_id]) > 5:
            track_history[track_id].pop(0)

        points = np.hstack(track_history[track_id]).astype(np.int32).reshape(
            (-1, 1, 2))
        cv2.rectangle(annotated_frame, (tlwh[0], tlwh[1]),
                      (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]), color, 2)

        cv2.polylines(annotated_frame, [points],
                      isClosed=False,
                      color=color,
                      thickness=5)
        cv2.putText(annotated_frame, str(track_id), (tlwh[0], tlwh[1]), 0,
                    5e-3 * 200, color, 2)


def main(tracker: SORT):

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_i = 0
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        raw_frame = frame.copy()
        frame_sharpness = cv2.Laplacian(frame, cv2.CV_64F).var()

        # Run raid detector on image.
        detections = raid_detection(frame_i)

        # Filter unwanted boxes.
        detection_boxes, detection_scores = filter_detections(detections)

        # Convert list to numpy array.
        if len(detection_boxes) > 0:
            detections = np.hstack([
                np.array(detection_boxes).reshape(-1, 4),
                np.array(detection_scores).reshape(-1, 1)
            ])
        else:
            detections = []

        # Feed RAID detection results to the tracker.
        online_targets = tracker.update_tracks(detections, raw_frame)

        # Save product images to output folder
        save_best_image(raw_frame, frame, online_targets, frame_sharpness)

        cv2.imshow('Visualize', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_i += 1


if __name__ == '__main__':

    tracker = SORT()

    main(tracker)
