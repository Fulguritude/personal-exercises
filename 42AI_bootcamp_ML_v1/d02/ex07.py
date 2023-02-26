from ex06 import get_true_positive_amount, get_false_positive_amount

def precision_score_(y_true, y_pred, label=None):
	"""
	Compute the precision score. TP / (TP + FP)
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
		The precision score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if len(y_pred) != len(y_true):
		return None
	TP = get_true_positive_amount(y_true, y_pred, label)
	FP = get_false_positive_amount(y_true, y_pred, label)
	return TP / (TP + FP)