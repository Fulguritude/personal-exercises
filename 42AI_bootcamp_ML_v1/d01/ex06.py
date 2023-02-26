import pandas as pd
import numpy as np
from ex03 import LinearRegression as lr

data = pd.read_csv("./resources/spacecraft_data.csv")

def plot_normalequation():
	lr_model = lr(np.array([[1.], [1.], [1.], [1.]]))
	lr_model.normalequation_data_(data, ["Age", "Thrust_power", "Terameters"], "Sell_price")
	lr_model.plot_model_multilinear_(data, ["Age", "Thrust_power", "Terameters"], "Age", "Sell_price")
	lr_model.plot_model_multilinear_(data, ["Age", "Thrust_power", "Terameters"], "Thrust_power", "Sell_price")
	lr_model.plot_model_multilinear_(data, ["Age", "Thrust_power", "Terameters"], "Terameters", "Sell_price")
