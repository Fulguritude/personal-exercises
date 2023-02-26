import numpy as np

def predict_(X, theta):
	if len(X) == 0 or len(X[0]) == 0 or len(theta) == 0:
		return None
	if len(X[0]) == len(theta):
		return np.dot(X, theta)
	if len(X[0]) + 1 == len(theta):
		return np.dot(X, theta[1:]) + theta[0]
	return None

def vec_regularization(theta, lambda_):
	"""Computes the regularization term of a non-empty numpy.ndarray, without any for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of dimension n * 1.
		lambda_: has to be a float.
	Returns:
		The regularization term of theta.
		None if theta is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if len(theta) == 0:
		return None
	return np.dot(theta.T, theta) * lambda_