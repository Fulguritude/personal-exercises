def dot_(x, y):
	"""
	Computes the dot product of two non-empty numpy.ndarray, using a
		for-loop. The two arrays must have the same dimensions.
	Args:
		x: has to be an numpy.ndarray, a vector.
		y: has to be an numpy.ndarray, a vector.
	Returns:
		The dot product of the two vectors as a float.
		None if x or y are empty numpy.ndarray.
		None if x and y does not share the same dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if len(x) == 0 or len(y) == 0 or len(x) != len(y):
		return None
	acc = 0
	for i in range(len(x)):
		acc = acc + x[i] * y[i]
	return acc