import numpy as np

from ex01 import predict_


def reg_mse(X, Y, theta, lambda_):
	"""
	Computes the regularized mean squared error of three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions.
	Args:
		Y: has to be a numpy.ndarray, a vector of dimension m * 1.
		X: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a 	vector of dimension n * 1.
		lambda: has to be a float.
	Returns:
		The mean squared error as a float.
		None if Y, X, or theta are empty numpy.ndarray.
		None if Y, X or theta does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if (len(X) == 0 or len(X[0]) == 0 or len(theta) == 0 or len(Y) == 0
		or len(X) != len(Y) or len(X[0]) != len(theta)):
		return None
	loss_vec = predict_(X, theta) - Y
	return (np.dot(loss_vec.T, loss_vec) + lambda_ * np.dot(theta.T, theta)) / len(loss_vec)

