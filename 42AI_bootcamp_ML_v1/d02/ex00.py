import numpy as np

def sigmoid_(x):
	"""
	Compute the sigmoid of a scalar or a list.
	Args:
		x: a scalar or list
	Returns:
		The sigmoid value as a scalar or list.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if isinstance(x, list):
		return list(map(lambda t: 1 / (1 + np.exp(-t)), x))
	return 1 / (1 + np.exp(-x))