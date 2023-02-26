import numpy as np
from ex00 import entropy

#Gini Impurity is the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the dataset.
#Gini impurity is a metric that evaluates the quality of a split in the dataset.

#G(X) = sum_{c € Classes} (p(c) * (1 - p(c)))

def gini(array):
	"""
	Computes the gini impurity of a non-empty numpy.ndarray
	Args:
		array: numpy.ndarray
	Return
		float: gini_impurity as a float or None if input is not a non-empty numpy.ndarray
	"""
	if ((not isinstance(array, list) and not isinstance(array, np.ndarray)) or
		len(array) == 0):
		return None
	inv_len = 1 / len(array)
	#array = inv_len * array
	classes = list(set(array))
	class_amount = len(classes)
	acc = 0
	inv_len = 1 / len(array)
	for label in classes:
		prob = sum([elem == label for elem in array]) * inv_len
		acc = acc + prob * (1 - prob)
	return acc

