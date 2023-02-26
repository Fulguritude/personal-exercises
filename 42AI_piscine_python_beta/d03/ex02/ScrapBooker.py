#https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python

import numpy as np

class ScrapBooker():

	@staticmethod
	def crop(array, new_img_size, positions=(0, 0)):
		if (array.ndim < 2 or 3 < array.ndim):
			raise ValueError("Not a valid image np.array.")
		h, w = new_img_size
		y, x = positions
		max_h, max_w = array.shape[0:2]
		if (x + w > max_w or y + h > max_h):
			raise ValueError("Improper cropping parameters.")
		cropped_img = array[y:y+h, x:x+w]
		return cropped_img

	@staticmethod
	def thin(array, n, axis):
		if (axis != 0 and axis != 1):
			raise ValueError("Invalid axis integer")
		elif (axis == 0):
			array = array[::n, :]
		else:
			array = array[:, ::n]
		return array

	@staticmethod
	def juxtapose(array, n, axis):
		res = np.copy(array)
		if (axis != 0 and axis != 1):
			raise ValueError("Invalid axis integer")
		elif (axis == 0):
			for i in range(n - 1):
				res = np.concatenate((res, array), axis=0)
		else:
			for i in range(n - 1):
				res = np.concatenate((res, array), axis=1)
		return res

	@staticmethod
	def mosaic(array, grid_size):
		res = ScrapBooker.juxtapose(array, grid_size[0], 0)
		res = ScrapBooker.juxtapose(res, grid_size[1], 1)
		return res