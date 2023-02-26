import numpy as np

from ex01 import predict_


def vec_reg_linear_grad(X, Y, theta, lambda_):
	"""
	Computes the regularized linear gradient of three non-empty numpy.ndarray, with two for-loop. The three arrays must have compatible dimensions.
	Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of dimension n * 1. 
		lambda_: has to be a float.
	Returns:
		A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if (len(X) == 0 or len(X[0]) == 0 or len(theta) == 0 or len(Y) == 0
		or len(X) != len(Y) or len(X[0]) != len(theta)):
		return None
	loss_vec = predict_(X, theta) - Y
	gradient = np.dot(X.T, loss_vec)
	reg_param = lambda_ * theta
	#result = np.zeros(len(theta) + 1)
	#result[0] = sum(loss_vec)
	#result[1:] = (gradient + reg_param)
	#result = result / len(loss_vec)
	gradient = (gradient + reg_param) / len(loss_vec)
	return gradient
