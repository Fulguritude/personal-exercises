from ex00 import sum_

def mean_mapped_(x, f):
	if len(x) == 0:
		return None
	return sum_(x, f) / len(x)

def mean_(x):
	"""
	Computes the mean of a non-empty numpy.ndarray, using a for-loop.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		The mean as a float.
		None if x is an empty numpy.ndarray. 
	Raises:
		This function should not raise any Exception.
	"""
	if len(x) == 0:
		return None
	return sum_(x, lambda x: x) / len(x)