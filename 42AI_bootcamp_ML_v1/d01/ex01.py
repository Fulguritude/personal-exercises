import numpy as np

from ex00 import predict_


def cost_elems_(theta, X, Y):
	"""
	Description:
		Calculates all the elements 0.5*M*(y_pred - y)^2 of the cost function.
	Args:
		theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
		X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
		Y: has to be a numpy.ndarray, a matrix of dimensions (number of training examples, 1)
	Returns:
		J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
		None if there is a dimension matching problem between X, Y or theta.
	Raises:
		This function should not raise any Exception.
	"""
	if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
		len(X) != len(Y) or len(X[0]) != len(theta) - 1 or len(Y[0]) != 1 or len(theta[0]) != 1):
		return None
	return 0.5 * ((predict_(theta, X) - Y) ** 2) / len(Y)


def cost_(theta, X, Y):
	"""
	Description:
		Calculates the value of cost function.
	Args:
		theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
		X: has to be a numpy.ndarray, a vector of dimension (number of training examples, number of features).
		Y: has to be a numpy.ndarray, a matrix of dimensions (number of training examples, 1)
	Returns:
		J_value : has to be a float.
		None if X does not match the dimension of theta.
	Raises:
		This function should not raise any Exception.
	"""
	if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
		len(X) != len(Y) or len(X[0]) != len(theta) - 1 or len(Y[0]) != 1):
		return None
	return np.sum(cost_elems_(theta, X, Y))
