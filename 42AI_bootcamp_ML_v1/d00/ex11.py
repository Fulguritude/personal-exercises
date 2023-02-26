from ex01 import mean_
from ex04 import dot_

def gradient_(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.ndarray, using
		a for-loop. The two arrays must have the compatible dimensions.
	NB: this function get the gradient by minimizing the error as much as possible
	Args:
		x: has to be an numpy.ndarray, a matrice of dimension m * n.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector n * 1.
	Returns:
		The gradient as a numpy.ndarray, a vector of dimensions n * 1.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
	Raises:
		This function should not raise any Exception.
	"""

	if len(x) == 0 or len(x[0]) == 0 or len(y) == 0 or len(theta) == 0 or len(x[0]) != len(theta) or len(x) != len(y):
		return None
	y_hat = [dot_(theta, x_i) for x_i in x]
	dist_vec = [y_hat[i] - y[i] for i in range(len(y))]
	vec = [dist_vec[i] * x[i] for i in range(len(y))]
	return mean_(vec)