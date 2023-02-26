import numpy

class NumPyCreator():

	def from_list(lst):
		return np.array(lst)

	def from_tuple(tpl):
		return np.asarray(tpl)

	def from_iterable(itr):
		return np.fromiter(itr)

	def from_shape(shape, value):
		return np.full(shape, value)

	def random(shape):
		return np.random.random(shape)

	def identity(n):
		return np.eye(n)