import pandas as pd
import numpy as np
from ex03 import LinearRegression as lr


data = pd.read_csv("./resources/spacecraft_data.csv")

def plot_age_vs_price():
	lr_model = lr(np.array([[0.], [0.]]))
	lr_model.fit_data_(data, "Age", "Sell_price")
	lr_model.plot_model_(data, "Age", "Sell_price")

def plot_thrust_vs_price():
	lr_model = lr(np.array([[0.], [0.]]))
	lr_model.fit_data_(data, "Thrust_power", "Sell_price", alpha=0.00001)
	lr_model.plot_model_(data, "Thrust_power", "Sell_price")

def plot_tmeters_vs_price():
	lr_model = lr(np.array([[0.], [0.]]))
	lr_model.fit_data_(data, "Terameters", "Sell_price", alpha=0.00001, n_cycle=20000)
	lr_model.plot_model_(data, "Terameters", "Sell_price")

def plot_multilinear_regression():
	lr_model = lr(np.array([[1.], [1.], [1.], [1.]]))
	lr_model.fit_data_multilinear_(data, ["Age", "Thrust_power", "Terameters"], "Sell_price", alpha=0.00001, n_cycle = 200000)
	lr_model.plot_model_multilinear_(data, ["Age", "Thrust_power", "Terameters"], "Age", "Sell_price")
	lr_model.plot_model_multilinear_(data, ["Age", "Thrust_power", "Terameters"], "Thrust_power", "Sell_price")
	lr_model.plot_model_multilinear_(data, ["Age", "Thrust_power", "Terameters"], "Terameters", "Sell_price")
