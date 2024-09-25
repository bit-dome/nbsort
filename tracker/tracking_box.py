import numpy as np
from collections import OrderedDict
from collections import deque




def multi_gmc(stracks, H=np.eye(2, 3)):
    if len(stracks) > 0:
        multi_xywh = np.asarray([st._xywh.copy() for st in stracks])
        
        R = H[:2, :2]  # Rotation part
        t = H[:2, 2]   # Translation part

        for i, xywh in enumerate(multi_xywh):
            xywh[:2] = R.dot(xywh[:2]) + t  # Transform x, y
            stracks[i]._xywh = xywh.copy()




class TrackingBoxState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4



class TrackingBox():


    track_id = 0
    is_activated = False
    state = TrackingBoxState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    @property
    def end_frame(self):
        return self.frame_id

    def mark_lost(self):
        self.state = TrackingBoxState.Lost

    def mark_long_lost(self):
        self.state = TrackingBoxState.LongLost

    def mark_removed(self):
        self.state = TrackingBoxState.Removed



    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self._xywh = None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)




    def activate(self, frame_id, track_id):
        """Start a new tracklet"""
        self.track_id = track_id

        self._xywh = self.tlwh_to_xywh(self._tlwh)


        self.tracklet_len = 0
        self.state = TrackingBoxState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id):

        self._xywh = self.tlwh_to_xywh(new_track.tlwh)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackingBoxState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
       
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self._xywh= self.tlwh_to_xywh(new_tlwh)

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackingBoxState.Tracked
        self.is_activated = True

        self.score = new_track.score


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self._xywh is None:
            return self._tlwh.copy()
        ret = self._xywh[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
