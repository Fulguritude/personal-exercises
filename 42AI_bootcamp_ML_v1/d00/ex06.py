from ex04 import dot_

def mat_mat_prod_(x, y):
	"""
	Computes the product of two non-empty numpy.ndarray,
		for-loop. The two arrays must have compatible dimensions.
	Args:
		x: has to be an numpy.ndarray, a matrix of dimension m
		y: has to be an numpy.ndarray, a vector of dimension n
	Returns:
		The product of the matrices as a matrix of dimension m
		None if x or y are empty numpy.ndarray.
		None if x and y does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if len(x) == 0 or len(y) == 0 or len(x[0]) == 0 or len(y[0]) == 0 or len(x[0]) != len(y):
		return None
	res = []
	for i in range(len(x)):
		row = x[i]
		new_row = [dot_(row, y[:, j]) for j in range(len(y[0]))]
		res.append(new_row)
	return res