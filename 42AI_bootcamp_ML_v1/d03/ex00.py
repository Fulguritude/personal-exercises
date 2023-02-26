def regularization(theta, lambda_):
	"""Computes the regularization term of a non-empty numpy.ndarray, with a for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of dimension n * 1.
		lambda: has to be a float.
	Returns:
		The regularization term of theta.
		None if theta is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if len(theta) == 0:
		return None
	return lambda_ * sum([theta_i ** 2 for theta_i in theta])