import numpy as np
from ex01 import np_matrix_from_any

def vec_log_gradient_(x, y_true, y_pred):
	"""
	Computes the gradient.
	Args:
		x: a 1d or 2d numpy ndarray for the samples
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
	Returns:
		The gradient as a scalar or a numpy ndarray of the width of x.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	x = np_matrix_from_any(x)
	y_true = np_matrix_from_any(y_true)
	y_pred = np_matrix_from_any(y_pred)
	if len(y_pred[0]) != len(y_true[0]) or len(x) != len(y_true[0]):
		print(str(len(y_pred[0]) != len(y_true[0])) + " " + str(len(x) != len(y_true[0])))
		return None
	return np.dot(y_pred - y_true, x).T