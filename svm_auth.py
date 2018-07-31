"""
SVM authentication methods
"""

import numpy as np
from scipy.stats import expon
from scipy.special import expit
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import sklearn.utils

from auth_base import AuthBase, UserModel


class SVMUserModel(UserModel):
    min_thresh = 0.0
    max_thresh = 1.0

    def __init__(self, model, prob, temperature):
        self.model = model
        self.prob = prob
        self.temperature = temperature
    
    def score_sessions(self, sessions):
        _, auth_window, n_features = sessions.shape
        sessions = sessions.reshape((-1, n_features))
        
        if self.prob:
            probs = self.model.predict_proba(sessions)[:, 0]
        else:
            distances = self.model.decision_function(sessions)
            probs = 1 - expit(distances / self.temperature)
        
        probs = probs.reshape((-1, auth_window))
        return np.median(probs, axis=1)


class SVMBase(AuthBase):
    def __init__(self, dev_data, ubm, n_neighbors=2, balanced=False,
            prob=True, n_iter=100, C_scale=100, gamma_scale=0.05,
            temperature=10, kernel="rbf", normalization=None,
            z_sessions=None, t_features=None):
        """
        Params
            dev_data : None or (n_samples, n_features)
                Development data from which nearest neighbors will be used
                    as negative samples during enrollment
            ubm : None or sklearn.mixture.GaussianMixture
                GMM background model from which random samples will be used
                    as negative samples during enrollment
            n_neighbors : int
                Number of nearest neighbors or random samples per positive
                    enrollment sample
            balanced : bool
                Whether to balance classes by weighting data
            prob : bool
                Whether to train sklearn built-in probability estimator, or
                    use squashed decision function
            n_iter : int
                Number of iterations 
            C_scale : float
                Scale parameter of exponential distribution for C hyperparameter
            temperature : float
                Temperature scaling used in softmax of decision function if
                    `prob` is set to False
            kernel : string or callable
                `kernel` parameter of sklearn.svm.SVC
        """
        self.ubm = ubm
        
        if self.ubm is None:
            self._init_index(dev_data)

        self.n_neighbors = n_neighbors
        self.prob = prob
        self.n_iter = n_iter
        self.temperature = temperature
        self.kernel = kernel
        self.class_weight = "balanced" if balanced else None

        self.param_distros = {
            "C" : expon(scale=C_scale),
            "gamma" : expon(scale=gamma_scale)
        }

        self._init_normalization(normalization, z_sessions, t_features)
    
    def enroll(self, enroll_data, normalization="default"):
        if self.ubm is None:
            negative_data = self._get_nearest_neighbors(enroll_data, self.n_neighbors)
        else:
            negative_data, _ = self.ubm.sample(self.n_neighbors * enroll_data.shape[0])
        
        svm = SVC(probability=self.prob, class_weight=self.class_weight,
            kernel=self.kernel)
        
        X = np.vstack((enroll_data, negative_data))
        y = [0] * enroll_data.shape[0] + [1] * negative_data.shape[0]
        
        X, y = sklearn.utils.shuffle(X, y)

        cv = RandomizedSearchCV(svm, self.param_distros, n_iter=self.n_iter,
            n_jobs=-1)
        cv.fit(X, y)
        # print(cv.best_params_, cv.best_score_)
        
        user_model = SVMUserModel(cv, self.prob, self.temperature)

        return self._normalize(user_model, normalization)

