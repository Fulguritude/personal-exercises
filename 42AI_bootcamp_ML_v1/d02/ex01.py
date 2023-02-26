import numpy as np
from ex00 import sigmoid_

#should have numpy forbidden, but who cares, this is the same as ex03
#https://machinelearningmastery.com/cross-entropy-for-machine-learning/

def np_matrix_from_any(X):
	if not isinstance(X, np.ndarray):
		X = np.array(X if isinstance(X, list) else [X])
	if not isinstance(X[0], np.ndarray):
		X = np.array(X if isinstance(X, list) else [X])
	return X

def log_predict_(theta, X):
	X = np_matrix_from_any(X)
	if len(X) == 0 or len(X[0]) == 0 or len(X[0]) != len(theta) - 1:
		return None
	return sigmoid_(np.dot(X, theta[1:]) + theta[0])

def log_loss_(y_true, y_pred, m, eps=1e-15):
	"""
	Description:
		Computes the logistic loss value, or cross-entropy loss.
	Args:
		y_true: a scalar or a list for the correct labels
		y_pred: a scalar or a list for the predicted labels
		m: the length of y_true (should also be the length of y_pred)
		eps: machine precision epsilon (default=1e-15)
	Returns:
		The logistic loss value as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	y_true = np_matrix_from_any(y_true)
	y_pred = np_matrix_from_any(y_pred)
	if len(y_pred[0]) != m or len(y_true[0]) != m:
		#print(str(len(y_pred[0]) != m) + " " + str(len(y_true[0]) != m))
		return None
	return (-(np.dot(np.log(y_pred), y_true.T) + np.dot(np.log(1 - y_pred), (1 - y_true).T)) / m)[0][0]


