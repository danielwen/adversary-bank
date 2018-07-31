"""
One-class SVM authentication methods
"""

from functools import partial

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import RandomizedSearchCV
import sklearn.utils
from scipy.special import expit
import scipy.stats

from auth_base import AuthBase, UserModel, one_class_cv_scorer

class OSVMUserModel(UserModel):
    min_thresh = 0.0
    max_thresh = 1.0

    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature

    def score_sessions(self, sessions):
        _, auth_window, n_features = sessions.shape
        sessions = sessions.reshape((-1, n_features))

        distances = self.model.decision_function(sessions)
        probs = expit(distances / self.temperature)
        
        probs = probs.reshape((-1, auth_window))
        return np.median(probs, axis=1)


class OSVMBase(AuthBase):
    def __init__(self, dev_data, temperature=10, kernel="rbf",
            gamma_scale=0.05, n_iter=12):
        self._init_index(dev_data)

        self.temperature = temperature
        self.n_iter = n_iter
        self.kernel = kernel

        self.param_distros = {
            "nu" : scipy.stats.beta(2, 2),
            "gamma" : scipy.stats.expon(scale=gamma_scale)
        }

    def enroll(self, enroll_data):
        enroll_data = sklearn.utils.shuffle(enroll_data)
        negative_data = self._get_nearest_neighbors(enroll_data, 1)
        indices = np.arange(enroll_data.shape[0])

        svm = OneClassSVM(kernel=self.kernel)

        cv = RandomizedSearchCV(svm, self.param_distros,
            scoring=partial(one_class_cv_scorer, negative_data),
            n_iter=self.n_iter)
        
        # The y's (indices) are ignored during training, but are used by our
        # scoring function
        cv.fit(enroll_data, indices) 
        
        return OSVMUserModel(cv, self.temperature)
