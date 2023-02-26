#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html

import pandas as pd
import matplotlib.pyplot as plt
import math

class MyPlotLib():

	@staticmethod
	def histogram(df, features):
		df.dropna(subset=features, inplace=True)
		df.drop_duplicates(['Name'], inplace = True)
		df = df[features]
		df.hist()
		plt.show()

	@staticmethod
	def density(df, features):
		df.dropna(subset=features, inplace=True)
		df.drop_duplicates(['Name'], inplace = True)
		df = df[features]
		df.plot.density()
		plt.show()

	@staticmethod
	def pair_plot(df, features):
		df.dropna(subset=features, inplace=True)
		df.drop_duplicates(['Name'], inplace = True)
		df = df[features]
		pd.plotting.scatter_matrix(df)
		plt.show()

	@staticmethod
	def box_plot(df, features):
		df.dropna(subset=features, inplace=True)
		df.drop_duplicates(['Name'], inplace = True)
		df = df[features]
		df.plot.box()
		plt.show()