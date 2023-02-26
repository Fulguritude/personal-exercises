import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

from ex03 import LinearRegression as lr

data = pd.read_csv("./resources/data.csv")
X = np.array(data.iloc[:, 0:2]).reshape(-1,2)
Y = np.array(data.iloc[:, 2]).reshape(-1,1)

linear_model1 = lr(np.array([[1], [1], [1]]))
linear_model2 = lr(np.array([[1], [1], [1]]))
linear_model1.fit_(X, Y)
linear_model2.fit_(X, Y)
Y_model1 = linear_model1.predict_(X)
Y_model2 = linear_model2.predict_(X)

print(math.sqrt(linear_model1.mse_(X, Y)))
print(math.sqrt(mean_squared_error(Y, Y_model1)))
print(math.sqrt(linear_model2.mse_(X, Y)))
print(math.sqrt(mean_squared_error(Y, Y_model2)))