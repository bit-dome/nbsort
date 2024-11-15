import numpy as np
from collections import OrderedDict
from collections import deque
from tracker.kalman_filter import KalmanFilter


def multi_predict(stracks):
    if len(stracks) > 0:
        multi_xywh = np.asarray([st._xywh.copy() for st in stracks])
        multi_velocity = np.asarray([st._velocity.copy() for st in stracks])
       
        multi_covariance = np.asarray([st._covariance for st in stracks])


        for i, st in enumerate(stracks):
            if st.state != TrackingBoxState.Tracked:
                multi_velocity[i][2] = 0
                multi_velocity[i][3] = 0
        multi_xywh, multi_velocity, multi_covariance = TrackingBox.shared_kalman.multi_predict(multi_xywh, multi_velocity, multi_covariance)

        for i, (xywh,velocity, cov) in enumerate(zip(multi_xywh, multi_velocity, multi_covariance)):
            stracks[i]._xywh = xywh
            stracks[i]._velocity = velocity
            stracks[i]._covariance = cov



def multi_gmc(stracks, H=np.eye(2, 3)):

    if len(stracks) > 0:
        multi_xywh = np.asarray([st._xywh.copy()[:4] for st in stracks])
        
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

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)

        # 1x4
        self._xywh = None
        # 1x4
        self._velocity = None
        # 8x8
        self._covariance = None

        self.kalman_filter = None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9





    def activate(self, kalman_filter, frame_id, track_id):
        self.track_id = track_id
        self.kalman_filter = kalman_filter

        if (self.kalman_filter is not None):    
            self._xywh, self._velocity, self._covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
            
        else:
            self._xywh = self.tlwh_to_xywh(self._tlwh)

        self.tracklet_len = 0
        self.state = TrackingBoxState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id):


        if (self.kalman_filter is not None):
            self._xywh, self._velocity, self._covariance = self.kalman_filter.update(self._xywh,self._velocity, self._covariance, self.tlwh_to_xywh(new_track.tlwh))
        else:
            self._xywh = self.tlwh_to_xywh(new_track.tlwh)

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackingBoxState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
       
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

        

        if (self.kalman_filter is not None):

            self._xywh, self._velocity, self._covariance = self.kalman_filter.update(self._xywh, self._velocity, self._covariance, self.tlwh_to_xywh(new_track.tlwh))
        else:
            self._xywh = self.tlwh_to_xywh(new_track.tlwh)



        self.state = TrackingBoxState.Tracked
        self.is_activated = True

        self.score = new_track.score

    def predict(self):
 
        if self.state != TrackingBoxState.Tracked:
            self._velocity[2] = 0
            self._velocity[3] = 0


        self._xywh, self._velocity, self._covariance = self.kalman_filter.predict(self._xywh, self._velocity, self._covariance)


    @property
    def tlwh(self):
        if self._xywh is None:
            return self._tlwh.copy()
        ret = self._xywh[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        # top left, bottom right   
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret


    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

