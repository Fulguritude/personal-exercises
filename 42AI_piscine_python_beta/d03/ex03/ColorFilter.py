import numpy as np
import math

class ColorFilter():

	@staticmethod
	def invert(array):
		if array.ndim != 3:
			raise ValueError("Improper image array.")
		white_array = np.full(array.shape, 255)
		return white_array - array

	@staticmethod
	def to_green(array):
		slice_r = np.expand_dims(np.full(array.shape[0:2], 0), 2)
		slice_g = np.expand_dims(np.full(array.shape[0:2], 1), 2)
		slice_b = np.expand_dims(np.full(array.shape[0:2], 0), 2)
		green_array = np.concatenate((slice_r, slice_g, slice_b), axis=2)
		return green_array * array

	@staticmethod
	def celluloid(array):
		start = 20
		steps = 4
		end = 221
		func = lambda x : int((math.ceil((x - start) / float(end - start) * steps) / float(steps)) * (end - start) + start)
		lfunc = np.vectorize(func)
		return lfunc(array)
		"""
		def to_cell_shade(color_val):
			r = start
			while r < color_val[0] and r < end:
				r += incr
			g = start
			while g < color_val[1] and g < end:
				g += incr
			b = start
			while b < color_val[2] and b < end:
				b += incr 
			return [r,g,b]
		"""

		"""
		dir_r = np.arange(20, 220, 40)
		dir_g = np.arange(20, 220, 40)
		dir_b = np.arange(20, 220, 40)
		color_tensor = np.full((len(dir_r), len(dir_g), len(dir_b)), (0,0,0))
		for r in range(dir_r):
			for g in range(dir_g):
				for b in range(dir_b):
					color_tensor[r,g,b] = (dir_r[r], dir_g[g], dir_b[b])
		"""

	@staticmethod
	def to_grayscale(array, filter):
		if filter == 'm' or filter == 'mean':
			base_weight = np.array([0.333333333, 0.333333333, 0.333333333])
		elif filter == 'w' or filter == 'weigthed':
			base_weight = np.array([0.299, 0.587, 0.114])
		else:
			raise ValueError("Invalid filter.")
		weight_tensor = np.tile(base_weight, (array.shape[0], array.shape[1], 1))
		weighted_array = np.multiply(weight_tensor, array)
		grayscale_map = np.sum(weighted_array, 2)
		f = lambda x: int(x)
		vf = np.vectorize(f)
		grayscale_map = vf(grayscale_map)
		grayscale_array = np.expand_dims(grayscale_map, 2)
		grayscale_array = np.tile(grayscale_array, (1,1,3))
		print(str(grayscale_array))
		return grayscale_array
