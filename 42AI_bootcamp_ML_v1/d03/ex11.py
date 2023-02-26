import numpy as np

def minmax(x):
	"""
	Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		x' as a numpy.ndarray.
		None if x is a non-empty numpy.ndarray.
	Raises:
		This function shouldn't raise any Exception.
	"""
	min_x = np.min(x)
	return (x - min_x) / (np.max(x) - min_x)