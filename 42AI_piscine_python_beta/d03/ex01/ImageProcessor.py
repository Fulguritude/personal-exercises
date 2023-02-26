#https://practice2code.blogspot.com/2017/07/cheat-sheets-for-data-science-machine.html
#https://intellipaat.com/mediaFiles/2018/12/Python-NumPy-Cheat-Sheet-1.png
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html
#https://stackoverflow.com/questions/15345790/scipy-misc-module-has-no-attribute-imread
#https://stackoverflow.com/questions/14812342/matplotlib-has-no-attribute-pyplot

import numpy as np
#from scipy import misc
import imageio
import matplotlib.pyplot as plt

class ImageProcessor():

	@staticmethod
	def load(img_path):
		img = imageio.imread(img_path)
		print("Loading image of dimensions " + str(img.shape[0]) + " x " + str(img.shape[1]) )
		return img

	@staticmethod
	def display(img):
		fig, ax = plt.subplots()
		im = ax.imshow(img)
		plt.show()