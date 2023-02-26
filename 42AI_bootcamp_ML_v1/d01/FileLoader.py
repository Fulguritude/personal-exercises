#https://practice2code.blogspot.com/2017/07/cheat-sheets-for-data-science-machine.html

#	https://github.com/legolas140/competitive-data-science-1/blob/master/assignment1/PandasBasics.ipynb
#	https://www.machinelearningplus.com/python/101-pandas-exercises-python/
#	https://pynative.com/python-pandas-exercise/
#	https://www.w3resource.com/python-exercises/pandas/index.php
#	https://www.pythonprogramming.in/pandas-examples.html


import pandas as pd

class FileLoader():

	@staticmethod
	def load(fpath):
		df = pd.read_csv(fpath)
		print("Loading dataset of dimensions " + str(df.shape[0]) + " x " + str(df.shape[1]))
		return df

	@staticmethod
	def display(df, n):
		if n > 0:
			print(str(df[:n])) 
		elif n < 0:
			print(str(df[n:]))