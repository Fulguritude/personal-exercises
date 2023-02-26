

def get_subset_count(y_true, y_pred, label_true, label_pred):
	if len(y_pred) != len(y_true):
		return None
	return sum([y_true[i] == label_true and y_pred[i] == label_pred for i in range(len(y_pred))])

def get_true_positive_amount(y_true, y_pred, label=None):
	if len(y_pred) != len(y_true):
		return None
	if label:
		return sum([y_pred[i] == label and y_true[i] == label for i in range(len(y_pred))])##return sum([y_true[i] == y_pred[i] and y_pred[i] == label  for i in range(len(y_pred))])
	return sum([y_true[i] == y_pred[i] and y_pred[i] == 1 for i in range(len(y_pred))])

def get_true_negative_amount(y_true, y_pred, label=None):
	if len(y_pred) != len(y_true):
		return None
	if label:
		return sum([y_pred[i] != label and y_true[i] != label for i in range(len(y_pred))])##return sum([y_true[i] == y_pred[i] and y_pred[i] != label  for i in range(len(y_pred))])
	return sum([y_true[i] == y_pred[i] and y_pred[i] == 0 for i in range(len(y_pred))])

def get_false_positive_amount(y_true, y_pred, label=None):
	if len(y_pred) != len(y_true):
		return None
	if label:
		return sum([y_pred[i] == label and y_true[i] != label for i in range(len(y_pred))])##return sum([y_true[i] != y_pred[i] and y_pred[i] == label  for i in range(len(y_pred))])
	return sum([y_true[i] != y_pred[i] and y_pred[i] == 1 for i in range(len(y_pred))])

def get_false_negative_amount(y_true, y_pred, label=None):
	if len(y_pred) != len(y_true):
		return None
	if label:
		return sum([y_pred[i] != label and y_true[i] == label for i in range(len(y_pred))])##return sum([y_true[i] != y_pred[i] and y_pred[i] != label for i in range(len(y_pred))])
	return sum([y_true[i] != y_pred[i] and y_pred[i] == 0 for i in range(len(y_pred))])

def accuracy_score_(y_true, y_pred):
	"""
	Compute the accuracy score. (TP+TN)/(TP+TN+FP+FN) 
	Args:
		y_true: a scalar or a numpy ndarray for the correct labels
		y_pred: a scalar or a numpy ndarray for the predicted labels
	Returns:
		The accuracy score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if len(y_pred) != len(y_true):
		return None
	print(y_true)
	print(y_pred)
	m = len(y_true)
	#TP = get_true_positive_amount(y_true, y_pred)
	#TN = get_true_negative_amount(y_true, y_pred)
	#return (TP + TN) / m
	return sum([y_true[i] == y_pred[i] for i in range(m)]) / m