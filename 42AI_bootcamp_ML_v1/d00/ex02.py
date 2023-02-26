from ex01 import mean_

def variance_(x):
	"""
	Computes the variance of a non-empty numpy.ndarray, using a for-loop.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		The variance as a float.
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if len(x) == 0:
		return None
	mean_val = mean_(x)
	acc = 0
	for x_i in x:
		acc = acc + (x_i - mean_val) ** 2
	return acc / len(x)