import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    A Kalman filter for tracking bounding boxes in image space.

    The state space (8 dimensions):
        x, y, w, h, vx, vy, vw, vh
    represents the bounding box center position (x, y), width (w), height (h),
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, w, h) is directly observed in the state space (linear observation model).
    """

    def __init__(self):
        state_dim, dt = 4, 1.0

        # Define model matrices for the Kalman filter
        self._transition_matrix = np.eye(2 * state_dim, 2 * state_dim)
        for i in range(state_dim):
            self._transition_matrix[i, state_dim + i] = dt
        self._observation_matrix = np.eye(state_dim, 2 * state_dim)

        # Set uncertainty weights for position and velocity
        self._pos_std_weight = 1.0 / 20
        self._vel_std_weight = 1.0 / 160

    def initiate(self, measurement):
        """
        Initialize the track from an unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center (x, y),
            width w, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Mean vector (8-dimensional) and covariance matrix (8x8) of the new track.
            Unobserved velocities are initialized to zero.
        """
        pos_mean = measurement
        vel_mean = np.zeros_like(pos_mean)
        state_mean = np.r_[pos_mean, vel_mean]

        pos_std = [2 * self._pos_std_weight * measurement[2],  # x
                   2 * self._pos_std_weight * measurement[3],  # y
                   2 * self._pos_std_weight * measurement[2],  # w
                   2 * self._pos_std_weight * measurement[3]]  # h
        vel_std = [10 * self._vel_std_weight * measurement[2],
                   10 * self._vel_std_weight * measurement[3],
                   10 * self._vel_std_weight * measurement[2],
                   10 * self._vel_std_weight * measurement[3]]
        state_cov = np.diag(np.square(pos_std + vel_std))
        
        return state_mean[:4], state_mean[4:], state_cov

    def predict(self, pos, vel, covariance):
        """
        Run the Kalman filter prediction step.

        Parameters
        ----------
        pos : ndarray
            The 4-dimensional position mean vector (x, y, w, h) of the previous state.
        vel : ndarray
            The 4-dimensional velocity mean vector (vx, vy, vw, vh) of the previous state.
        covariance : ndarray
            The 8x8 covariance matrix of the previous state.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            Predicted position mean, velocity mean, and covariance matrix.
        """
        state_mean = np.hstack([pos, vel])
        
        pos_std = [self._pos_std_weight * state_mean[2],
                   self._pos_std_weight * state_mean[3],
                   self._pos_std_weight * state_mean[2],
                   self._pos_std_weight * state_mean[3]]
        vel_std = [self._vel_std_weight * state_mean[2],
                   self._vel_std_weight * state_mean[3],
                   self._vel_std_weight * state_mean[2],
                   self._vel_std_weight * state_mean[3]]
        process_cov = np.diag(np.square(np.r_[pos_std, vel_std]))

        state_mean = np.dot(state_mean, self._transition_matrix.T)
        covariance = np.linalg.multi_dot((self._transition_matrix, covariance, self._transition_matrix.T)) + process_cov

        return state_mean[:4], state_mean[4:], covariance

    def project(self, state_mean, covariance):
        """
        Project the state distribution to the measurement space.

        Parameters
        ----------
        state_mean : ndarray
            The state's 8-dimensional mean vector.
        covariance : ndarray
            The 8x8 covariance matrix of the state.

        Returns
        -------
        (ndarray, ndarray)
            Projected mean and covariance matrix in the measurement space.
        """
        pos_std = [self._pos_std_weight * state_mean[2],
                   self._pos_std_weight * state_mean[3],
                   self._pos_std_weight * state_mean[2],
                   self._pos_std_weight * state_mean[3]]
        measurement_cov = np.diag(np.square(pos_std))

        projected_mean = np.dot(self._observation_matrix, state_mean)
        projected_covariance = np.linalg.multi_dot((self._observation_matrix, covariance, self._observation_matrix.T))
        
        return projected_mean, projected_covariance + measurement_cov

    def multi_predict(self, pos, vel, covariances):
        """
        Vectorized version of the prediction step for multiple states.

        Parameters
        ----------
        pos : ndarray
            The Nx4 position mean matrix for N states.
        vel : ndarray
            The Nx4 velocity mean matrix for N states.
        covariances : ndarray
            The Nx8x8 covariance matrices for N states.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            Predicted position means, velocity means, and covariance matrices.
        """
        state_means = np.hstack([pos, vel])

        pos_std = [self._pos_std_weight * state_means[:, 2],
                   self._pos_std_weight * state_means[:, 3],
                   self._pos_std_weight * state_means[:, 2],
                   self._pos_std_weight * state_means[:, 3]]
        vel_std = [self._vel_std_weight * state_means[:, 2],
                   self._vel_std_weight * state_means[:, 3],
                   self._vel_std_weight * state_means[:, 2],
                   self._vel_std_weight * state_means[:, 3]]
        squared_std = np.square(np.r_[pos_std, vel_std]).T

        process_cov = [np.diag(squared_std[i]) for i in range(len(state_means))]
        process_cov = np.asarray(process_cov)

        state_means = np.dot(state_means, self._transition_matrix.T)
        left = np.dot(self._transition_matrix, covariances).transpose((1, 0, 2))
        covariances = np.dot(left, self._transition_matrix.T) + process_cov

        return state_means[:, :4], state_means[:, 4:], covariances

    def update(self, pos, vel, covariance, measurement):
        """
        Run the Kalman filter correction step.

        Parameters
        ----------
        pos : ndarray
            The predicted state's 4-dimensional position mean vector.
        vel : ndarray
            The predicted state's 4-dimensional velocity mean vector.
        covariance : ndarray
            The state's 8x8 covariance matrix.
        measurement : ndarray
            The 4-dimensional measurement vector (x, y, w, h).

        Returns
        -------
        (ndarray, ndarray, ndarray)
            Corrected position mean, velocity mean, and covariance matrix.
        """
        state_mean = np.hstack([pos, vel])
        
        projected_mean, projected_cov = self.project(state_mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._observation_matrix.T).T, check_finite=False).T
        innovation = measurement - projected_mean

        corrected_mean = state_mean + np.dot(innovation, kalman_gain.T)
        corrected_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        
        return corrected_mean[:4], corrected_mean[4:], corrected_covariance
