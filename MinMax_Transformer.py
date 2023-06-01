from h2oaicore.systemutils import segfault, loggerinfo, main_logger
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
import logging

class MinMaxTransformer(CustomTransformer):
	_regression = True
	_binary = True
	_multiclass = True
	_numeric_output = True
	_is_reproducible = True
	#_modules_needed_by_name = ["custom_package==1.0.0"]

	@staticmethod
	def do_acceptance_test():
		return True

	@staticmethod
	def get_default_properties():
		return dict(col_type = "numeric", min_cols = 1, max_cols = 1, relative_importance = 1)

	def fit_transform(self, X: dt.Frame, y: np.array = None):
		X_pandas = X.to_pandas()
		X_minmax = (X_pandas - X_pandas.min()) / (X_pandas.max() - X_pandas.min())
		return X_minmax

	def transform(self, X: dt.Frame):
		X_pandas = X.to_pandas()
		X_minmax = (X_pandas - X_pandas.min()) / (X_pandas.max() - X_pandas.min())
		return X_minmax
