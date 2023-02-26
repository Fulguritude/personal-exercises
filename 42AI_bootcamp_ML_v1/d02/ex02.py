import numpy as np
from ex01 import np_matrix_from_any

#should be without numpy for ex02, hence this is ex04, but who cares

def log_gradient_(x, y_true, y_pred):
	"""
	Computes the gradient.
	Args:
		x: a list or a matrix (list of lists) for the samples
		y_true: a scalar or a list for the correct labels
		y_pred: a scalar or a list for the predicted labels
	Returns:
		The gradient as a scalar or a list of the width of x.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	x = np_matrix_from_any(x)
	y_true = np_matrix_from_any(y_true)
	y_pred = np_matrix_from_any(y_pred)
	#print("y_true:" + str(y_true))
	#print("y_pred:" + str(y_pred))
	if len(y_pred[0]) != len(y_true[0]) or len(x) != len(y_true[0]):
		print(str(len(y_pred[0]) != len(y_true[0])) + " " + str(len(x) != len(y_true[0])))
		return None
	loss_vec = (y_pred - y_true)
	gradient = np.dot(loss_vec, x).T
	result = np.ones((len(gradient) + 1, 1))
	result[0, 0] = sum(loss_vec.T)[0]
	result[1:, 0] = gradient[:, 0]
	return result