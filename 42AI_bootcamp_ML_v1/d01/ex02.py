import numpy as np

#https://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables
#https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy

def vec_gradient_(theta, X, Y):
	"""
	Computes a gradient vector. The two arrays must have the compatible dimensions.
	NB: this function get the gradient by minimizing the error as much as possible
	Args:
		theta: has to be a numpy.ndarray, a vector of dimension n.
		X: has to be a numpy.ndarray, a matrix of dimension m * n.
		Y: has to be a numpy.ndarray, a vector of dimension m.
	Returns:
		The gradient as a numpy.ndarray, a vector of dimensions n * 1.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
		len(X) != len(Y) or len(X[0]) != len(theta) or len(Y[0]) != 1):
		return None
	Y_hat = np.dot(X, theta)
	loss_vec = (Y_hat - Y)
	gradient = np.dot(np.transpose(X), loss_vec) / len(Y)
	return gradient

def fit_(theta, X, Y, alpha = 0.001, n_cycle = 3000):
	"""
	Description:
		Performs a fit of Y(output) with respect to X.
	Args:
		theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
		X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
		Y: has to be a numpy.ndarray, a vector of dimension (number of training examples, 1).
	Returns:
		new_theta: numpy.ndarray, a vector of dimension (number of the features +1, 1).
		None if there is a matching dimension problem.
	Raises:
		This function should not raise any Exception.
	"""
	if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
		len(X) != len(Y) or len(X[0]) != len(theta) - 1 or len(Y[0]) != 1):
		return None
	extended_X = np.ones([len(X), len(X[0]) + 1])
	extended_X[:, 1:] = X
	for cycle in range(n_cycle):
		gradient = vec_gradient_(theta, extended_X, Y)
		new_theta = theta - alpha * gradient
		theta = new_theta
	return theta

