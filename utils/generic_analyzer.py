import pandas as pd
import numpy as np
from scipy import stats
from utils.common_helpers import *

class GenericAnalyzer():
    def __init__(self, X_test, y_test, protected_groups, priv_values, metric_names=None):
        self.protected_groups = protected_groups
        self.priv_values = priv_values
        self.X_test = X_test
        self.y_test = y_test
        self.test_groups = set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        self.metric_names = metric_names
        self.results = {}

    def compute_metrics(self, y_preds, model_name, save_results=False, metric_names=None):
    	y_pred_all = pd.Series(y_preds, index=self.y_test.index)
    	'''
    	if self.metric_names == None:
    		if metric_names == None:
    			raise Exception("metric_names is empty. Pass list of metric names to compute!")
    		else:
    			self.metric_names = metric_names
		'''
    	results = {}
    	results['overall'] = confusion_matrix_metrics(self.y_test, y_pred_all)

    	for group_name in self.test_groups.keys():
    		X_test_group = self.test_groups[group_name]
    		results[group_name] = confusion_matrix_metrics(self.y_test[self.test_groups[group_name].index], y_pred_all[self.test_groups[group_name].index])

    	self.results = results
    	return results