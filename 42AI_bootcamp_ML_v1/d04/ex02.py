import numpy as np
from ex00 import entropy
from ex01 import gini


#https://victorzhou.com/blog/information-gain/
#Information Gain = how much Entropy we removed
#Information Gain is a metric that evaluates the quality of a split in the dataset.
#It is calculated for a split by subtracting the weighted entropies of each branch from the original entropy.

#In the case of Gini below, information gain is referred to as "Gini gain"

def information_gain(array_source, array_target, criterion='gini'):
	"""
    Computes the information gain between the first and second array using the criterion ('gini' or 'entropy'). Also called Kullback Leibler Divergence
    Args:
		numpy.ndarray array_source: an array for data
		list array_children_list: list of numpy.ndarray, representing a split in the dataset
		str criterion: Should be in ['gini', 'entropy']
	Return:
		float: Shannon entropy as a float
		None if input is not a non-empty numpy.ndarray
		None if invalid input
	"""
	if ((not isinstance(array_source, list) and not isinstance(array_source, np.ndarray)) or
		len(array_source) == 0 or
		(not isinstance(array_target, list) and not isinstance(array_target, np.ndarray)) or
		len(array_target) == 0):
		return None
	start_information = 0.
	end_information = 0.
	if criterion == "gini":
		start_information = gini(array_source)
		end_information = gini(array_target)
	else:
		start_information = entropy(array_source)
		end_information = entropy(array_target)
	return start_information - end_information
