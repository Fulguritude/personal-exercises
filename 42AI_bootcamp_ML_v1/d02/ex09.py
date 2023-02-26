from ex07 import precision_score_
from ex08 import recall_score_

def f1_score_(y_true, y_pred, label=None):
	"""
	Compute the f1 score.
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
		The f1 score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if len(y_pred) != len(y_true):
		return None
	precision = precision_score_(y_true, y_pred, label)
	recall = recall_score_(y_true, y_pred, label)
	return 2 * precision * recall / (precision + recall)