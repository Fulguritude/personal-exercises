from ex00 import sum_

def mse_(y, y_hat):
	"""
	Computes the mean squared error of two non-empty numpy.ndarray, using
		a for-loop. The two arrays must have the same dimensions.
	Args:
		y: has to be an numpy.ndarray, a vector.
		y_hat: has to be an numpy.ndarray, a vector.
	Returns:
		The mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.ndarray.
		None if y and y_hat does not share the same dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if len(y) == 0 or len(y_hat) == 0 or len(y) != len(y_hat):
		return None
	dist_vec = [y[i] - y_hat[i] for i in range(len(y))]
	return sum_(dist_vec, lambda x: x ** 2) / len(y)