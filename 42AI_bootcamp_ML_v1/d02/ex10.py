import numpy as np
import pandas as pd

from ex06 import get_true_positive_amount, get_true_negative_amount, get_false_positive_amount, get_false_negative_amount, get_subset_count

#						[predicted labels]
#						label_1		label_2
# [observed] label_1		.			.
# [ labels ] label_2		.			.



def confusion_matrix_(y_true, y_pred, labels=None, df_option=False):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
		labels: optional, a list of labels to index the matrix. This may be used to reorder or select a subset of labels. (default=None)
		df_option: optional, if set to True the function will return a pandas dataframe instead of a numpy array. (default=False)
	Returns:
		The confusion matrix as a numpy ndarray.or pandas dataframe
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	result = None
	if labels == None:
		labels = list(set(list(y_true) + list(y_pred)))
	if len(y_pred) != len(y_true) or (len(labels) <= 1):
		return None
#	if len(labels) == 2:
#		result = [
#		    [get_true_negative_amount(y_true, y_pred, labels[0]), get_false_positive_amount(y_true, y_pred, labels[1])],
#		    [get_false_negative_amount(y_true, y_pred, labels[0]), get_true_positive_amount(y_true, y_pred, labels[1])]
#		]
#	else:
	result = [
		[
			get_subset_count(y_true, y_pred, labels[i], labels[j]) for j in range(len(labels))
		] for i in range(len(labels))
	]
	if df_option:
		return pd.DataFrame(result)
	return np.array(result)