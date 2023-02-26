from ex06 import get_true_positive_amount, get_false_negative_amount

def recall_score_(y_true, y_pred, label=None):
	"""
	Compute the recall score. TP / (TP + FN)
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
		The recall score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if len(y_pred) != len(y_true):
		return None
	TP = get_true_positive_amount(y_true, y_pred, label)
	FN = get_false_negative_amount(y_true, y_pred, label)
	return TP / (TP + FN)