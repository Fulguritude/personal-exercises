import numpy as np
from math import log2 as lg

#Information Entropy can be thought of as how how unpredictable a dataset is.

def entropy(array):
	"""
	Computes the Shannon Entropy of a non-empty numpy.ndarray
	Args:
		- array: numpy.ndarray
	Returns:
		float: shannon's entropy as a float
		None if input is not a non-empty numpy.ndarray
	"""
	if ((not isinstance(array, list) and not isinstance(array, np.ndarray)) or
		len(array) == 0):
		return None
	classes = list(set(array))
	class_amount = len(classes)
	acc = 0
	inv_len = 1 / len(array)
	for label in classes:
		prob = sum([elem == label for elem in array]) * inv_len
		acc = acc - prob * lg(prob)
	return acc