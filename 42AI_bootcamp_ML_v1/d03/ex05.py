import numpy as np

from ex01 import vec_regularization

def reg_log_loss_(y_true, y_pred, m, theta, lambda_, eps= 1e-15):
	"""
	Compute the logistic loss value.
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
		m: the length of y_true (should also be the length of y_pred)
		lambda_: a float for the regularization parameter
		eps: epsilon (default=1e-15)
	Returns:
		The logistic loss value as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if len(y_pred) != m or len(y_true) != m:
		print(str(len(y_pred) != m) + " " + str(len(y_true) != m))
		return None
	y_pred = [(eps if y_pred_i <= 0 else (1 - eps if y_pred_i >= 1 else y_pred_i)) for y_pred_i in y_pred]
	y_pred = np.array(y_pred)
	log_ypred = np.log(y_pred)
	log_1minus_ypred = np.log(1 - y_pred)
	cost = -(np.dot(y_true.T, log_ypred) + np.dot((1 - y_true.T), log_1minus_ypred))
	reg_value = vec_regularization(theta, lambda_)
	return (cost + reg_value) / m