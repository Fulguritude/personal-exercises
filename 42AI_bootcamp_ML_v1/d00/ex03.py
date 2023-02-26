from ex02 import variance_
import math

def std_deviation_(x):
	"""
	Computes the standard deviation of a non-empty numpy.ndarray, using a
		for-loop.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		The standard deviation as a float.
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""

	if len(x) == 0:
		return None
	return math.sqrt(variance_(x))