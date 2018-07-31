"""
Base classes for authentication methods
"""

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import faiss


def one_class_cv_scorer(negative_enroll, model, X, indices):
    # The y's (indices) must be the indices of X in the enrollment data
    negative_data = negative_enroll[indices]
    X_val = np.vstack((X, negative_data))
    y_val = [1] * X.shape[0] + [-1] * negative_data.shape[0]
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)


class AuthBase(object):
    """
    Base class for authentication methods
    """
    def __init__(self, dev_data):
        """
        Args
            dev_data : (n_samples, n_features)
                Data for development phase
        """
        raise NotImplementedError
    
    def _init_normalization(self, normalization, z_sessions, t_features):
        self.normalization = normalization
        self.z_sessions = z_sessions
        self.t_models = None

        if t_features is not None:
            print("Enrolling T-models...")
            normalization = "z" if self.normalization == "zt" else None
            self.t_models = []

            for features in tqdm(t_features):
                t_model = self.enroll(features, normalization=normalization)
                self.t_models.append(t_model)

    def _normalize(self, user_model, normalization):
        if normalization == "default":
            normalization = self.normalization

        if normalization is None:
            return user_model

        elif normalization == "z":
            assert self.z_sessions is not None
            return ZScorer(user_model, self.z_sessions)

        elif normalization == "t":
            assert self.t_models is not None
            return TScorer(user_model, self.t_models)

        elif normalization == "zt":
            assert self.z_sessions is not None
            assert self.t_models is not None
            return ZTScorer(user_model, self.z_sessions, self.t_models)
        
        else:
            raise ValueError("Invalid normalization type")

    def _init_index(self, dev_data):
        # Initialize faiss index of development data
        _, d = dev_data.shape
        self.dev_data = dev_data
        self.index = faiss.IndexFlatL2(d)
        self.index.add(dev_data)

    def _get_nearest_neighbors(self, X, k):
        # Find k nearest neighbors of X in faiss index
        _, I = self.index.search(X, k)
        return self.dev_data[I.ravel()]

    def enroll(self, enroll_data, **kwargs):
        """
        Args
            enroll_data : (n_enroll, n_features)
                Data for enrollment phase
        Returns
            user_model : UserModel
                User model
        """
        raise NotImplementedError


class UserModel(object):
    """
    Base class for user models
    """
    def __init__(self, model):
        """
        Args
            model : object
                Internal model used
        """
        raise NotImplementedError

    def score(self, session):
        """
        Args
            session : (auth_window, n_features)
                Cyles in authentication session
        Returns
            score : float
        """
        raise NotImplementedError

    def score_sessions(self, sessions):
        """
        Args
            sessions : (n_sessions, auth_window, n_features)
        Returns
            scores : (n_sessions,)
        """
        return np.array([self.score(session) for session in sessions])


class ZScorer(UserModel):
    min_thresh = -2.0
    max_thresh = 20.0

    def __init__(self, user_model, z_sessions):
        self.user_model = user_model
        z_scores = self.user_model.score_sessions(z_sessions)
        self.z_mean = np.mean(z_scores)
        self.z_std = np.std(z_scores, ddof=1)
    
    def score_sessions(self, sessions):
        return (self.user_model.score_sessions(sessions) - self.z_mean) / self.z_std


class TScorer(UserModel):
    min_thresh = -2.0
    max_thresh = 10.0
    
    def __init__(self, user_model, t_models):
        self.user_model = user_model
        self.t_models = t_models
    
    def score_sessions(self, sessions):
        t_scores = [t_model.score_sessions(sessions)
            for t_model in self.t_models] # (cohorts, sessions)
        mean = np.mean(t_scores, axis=0)
        std = np.std(t_scores, ddof=1, axis=0)
        return (self.user_model.score_sessions(sessions) - mean) / std


class ZTScorer(UserModel):
    min_thresh = -2.0
    max_thresh = 10.0

    def __init__(self, user_model, z_sessions, t_models):
        assert np.all([isinstance(t_model, ZScorer) for t_model in t_models])
        self.scorer = TScorer(ZScorer(user_model, z_sessions), t_models)

    def score_sessions(self, sessions):
        return self.scorer.score_sessions(sessions)
