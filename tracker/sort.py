import numpy as np


from tracker import matching
from tracker.gmc import GMC
from tracker.tracking_box import TrackingBoxState, TrackingBox, multi_gmc



def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret



class SORT(object):

    def __init__(self, 
                 track_high_thresh=0.25,
                 track_low_thresh=0.2,
                 new_track_thresh=0.3,
                 track_buffer_frames=60, 
                 match_thresh=0.8,
                 frame_rate=30):


        self.next_track_id = 0

        self.tracked_stracks = []  
        self.lost_stracks = []  
        self.removed_stracks = []  


        self.frame_id = 0


        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer_frames)
        self.max_time_lost = self.buffer_size


        self.gmc = GMC()



    def parse_detections(self, current_detections):
        if len(current_detections) == 0:
            return [], [], [], []

        bboxes = current_detections[:, :4]
        scores = current_detections[:, 4]

        # Filter out low-scoring detections
        valid_inds = scores > self.track_low_thresh
        bboxes, scores = bboxes[valid_inds], scores[valid_inds]

        # Find high-scoring detections
        high_score_inds = scores > self.track_high_thresh
        detections_data = bboxes[high_score_inds]
        scores_filtered = scores[high_score_inds]

        # Create TrackingBox objects for valid detections
        detections = [TrackingBox(tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(detections_data, scores_filtered)]

        return detections, bboxes, detections_data, scores




    def update_tracks(self, current_detections, img):
        self.frame_id += 1
        activated_tracks, refound_tracks, lost_tracks, removed_tracks = [], [], [], []

        # Parse detections
        detections, bboxes, dets, scores = self.parse_detections(current_detections)

        # Step 1: Separate confirmed and unconfirmed tracks
        unconfirmed_tracks = [track for track in self.tracked_stracks if not track.is_activated]
        confirmed_tracks = [track for track in self.tracked_stracks if track.is_activated]

        # Step 2: Jointly manage the track pool and apply camera motion correction
        track_pool = joint_stracks(confirmed_tracks, self.lost_stracks)
        warped_image = self.gmc.apply(img, dets)
        multi_gmc(track_pool, warped_image)
        multi_gmc(unconfirmed_tracks, warped_image)

        # Associate high-score detection boxes with tracked tracks
        iou_distances = matching.iou_distance(track_pool, detections)
        remain_track_indices, remain_det_indices, activated_tracks, refound_tracks = self.first_association(
            detections, track_pool, iou_distances, activated_tracks, refound_tracks
        )

        # Second association with low-score detections
        activated_tracks, refound_tracks, lost_tracks = self.second_association(
            scores, bboxes, remain_track_indices, lost_tracks, track_pool, activated_tracks, refound_tracks
        )

        # Handle unconfirmed detections
        detections, remaining_unconfirmed, activated_tracks, removed_tracks = self.handle_unconfirmed(
            detections, unconfirmed_tracks, remain_det_indices, activated_tracks, removed_tracks
        )

        # Initialize new tracks and manage existing ones
        self.init_new_track(detections, remaining_unconfirmed, activated_tracks)
        self.remove_lost_stracks(removed_tracks)
        self.merge_tracks(activated_tracks, refound_tracks, lost_tracks, removed_tracks)

        return [track for track in self.tracked_stracks]


    def first_association(self, detections, strack_pool, ious_dists, activated_starcks, refind_stracks):

        matches, remain_track_1, remain_det_1 = matching.linear_assignment(ious_dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackingBoxState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        return remain_track_1, remain_det_1,activated_starcks,refind_stracks




    def second_association(self, scores, bboxes, remain_track_1, lost_stracks, strack_pool, activated_starcks, refind_stracks):
        # Filter detections based on score thresholds
        if len(scores):
            inds_second = np.logical_and(scores > self.track_low_thresh, scores < self.track_high_thresh)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
        else:
            return activated_starcks, refind_stracks, lost_stracks  # Early return if no scores

        # Create detections from filtered boxes and scores
        detections_second = [TrackingBox(tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_second, scores_second)] if len(dets_second) > 0 else []

        # Track associations
        r_tracked_stracks = [strack_pool[i] for i in remain_track_1 if strack_pool[i].state == TrackingBoxState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, remain_track_2, _ = matching.linear_assignment(dists, thresh=0.5)

        # Update and classify tracks based on matches
        for itracked, idet in matches:
            track, det = r_tracked_stracks[itracked], detections_second[idet]
            (track.update(det, self.frame_id) if track.state == TrackingBoxState.Tracked else track.re_activate(det, self.frame_id))
            (activated_starcks if track.state == TrackingBoxState.Tracked else refind_stracks).append(track)

        # Mark remaining tracks as lost
        for it in remain_track_2:
            track = r_tracked_stracks[it]
            if track.state != TrackingBoxState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        return activated_starcks, refind_stracks, lost_stracks



    def init_new_track(self, detections, unconfirmed_remain_detections, activated_starcks):
        for inew in unconfirmed_remain_detections:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.frame_id, self.next_track_id)
            self.next_track_id += 1
            activated_starcks.append(track)

        return activated_starcks


    def remove_lost_stracks(self, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        return removed_stracks


    def handle_unconfirmed(self, detections, unconfirmed, remain_det_1, activated_starcks, removed_stracks):
        detections = [detections[i] for i in remain_det_1]
        ious_dists = matching.iou_distance(unconfirmed, detections)


        matches, u_unconfirmed, unconfirmed_remain_detections = matching.linear_assignment(ious_dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        return detections, unconfirmed_remain_detections, activated_starcks, removed_stracks

    def merge_tracks(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks):
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackingBoxState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
