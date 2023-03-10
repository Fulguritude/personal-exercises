import math
import numpy as np

def zscore(X):
	"""
	Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		x' as a numpy.ndarray.
		None if x is a non-empty numpy.ndarray.
	Raises:
		This function shouldn't raise any Exception.
	"""
	return (X - np.mean(X)) / math.sqrt(np.var(X))