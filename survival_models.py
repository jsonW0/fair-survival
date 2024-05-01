import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin

class FittedUniformBaseline(BaseEstimator,SurvivalAnalysisMixin):
    def __init__(self, alpha):
        self.estimator = CoxPHSurvivalAnalysis(alpha)
    def fit(self, X, y):
        self.estimator.fit(np.zeros((X.shape[0],1)),y) # just fit the intercept
        return self
    def predict(self, X):
        return self.estimator.predict(np.zeros((X.shape[0],1)))
    def predict_cumulative_hazard_function(self, X, return_array=False):
        return self.estimator.predict_cumulative_hazard_function(np.zeros((X.shape[0],1)),return_array=return_array)
    def predict_survival_function(self, X, return_array=False):
        return self.estimator.predict_survival_function(np.zeros((X.shape[0],1)),return_array=return_array)