from ex04 import dot_

def vec_linear_mse_(x, y, theta):
	"""
	Computes the mean squared error of three non-empty numpy.ndarray,
		without any for-loop. The three arrays must have compatible dimensions.
	Args:
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		x: has to be an numpy.ndarray, a matrix of dimesion m * n.
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
	return dot_(dist_vec, dist_vec) / len(y)