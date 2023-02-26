from ex00 import sum_
from ex04 import dot_

def linear_mse_(x, y, theta):
	"""
	Computes the mean squared error of three non-empty numpy.ndarray,
		using a for-loop. The three arrays must have compatible dimensions.
	NB: the prediction hypothesis y^hat_i = h_theta(x_i) = dot(theta, x_i)
	Args:
		x: has to be an numpy.ndarray, a matrix of dimesion m * n.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector of dimension n * 1.
	Returns:
		The mean squared error as a float.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if len(x) == 0 or len(x[0]) == 0 or len(y) == 0 or len(theta) == 0 or len(x[0]) != len(theta) or len(x) != len(y):
		return None
	y_hat = [dot_(theta, x_i) for x_i in x]
	dist_vec = [y_hat[i] - y[i] for i in range(len(y))]
	return sum_(dist_vec, lambda x: x**2) / len(y)