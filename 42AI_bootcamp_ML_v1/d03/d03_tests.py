#https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros

# good seaborn link for practical visualization of higher dimensional data
#https://jovianlin.io/data-visualization-seaborn-part-3/

import pandas as pd
import numpy as np
from scipy.special import expit
import random
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
#import sklearn

from ex00 import regularization
from ex01 import vec_regularization
from ex02 import reg_mse
from ex03 import reg_linear_grad
from ex04 import vec_reg_linear_grad
from ex05 import reg_log_loss_
from ex06 import reg_logistic_grad
from ex07 import vec_reg_logistic_grad
from ex08 import LinearRegression, LinearRegressionRidge, PolynomialRegression
from ex09 import LogisticRegression, LogisticRegressionRidge
from ex10 import zscore
from ex11 import minmax

#ex00
print("ex00")

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(regularization(X, 0.3))
#284.7
print(regularization(X, 0.01))
#9.47
print(regularization(X, 0))
#0.0



#ex01
print("\nex01")

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(vec_regularization(X, 0.3))
#284.7
print(vec_regularization(X, 0.01))
#9.47
print(vec_regularization(X, 0))
#0.0



#ex02
print("\nex02")

X = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]
])

Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3, 0.5, -6])
print(reg_mse(X, Y, Z, 0))
#2641.0
print(reg_mse(X, Y, Z, 0.1))
#2641.6464285714287
print(reg_mse(X, Y, Z, 0.5))
#2644.2321428571427



#ex03
print("\nex03")

X = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]
])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3, 10.5, -6])
print(reg_linear_grad(X, Y, Z, 1))
#array([-192.64285714, 887.5, -679.57142857])
print(reg_linear_grad(X, Y, Z, 0.5))#, 0.5))
#array([-192.85714286, 886.75, -679.14285714])
print(reg_linear_grad(X, Y, Z, 0.0))
#array([-193.07142857, 886., -678.71428571])



#ex04
print("\nex04")

X = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]
])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3, 10.5, -6])
print(vec_reg_linear_grad(X, Y, Z, 1))
#array([-192.64285714, 887.5, -679.57142857])
print(vec_reg_linear_grad(X, Y, Z, 0.5))#, 0.5))
#array([-192.85714286, 886.75, -679.14285714])
print(vec_reg_linear_grad(X, Y, Z, 0.0))
#array([-193.07142857, 886., -678.71428571])



#ex05
print("\nex05")
# Test n.1
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = expit(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.0))
# 7.233346147374828

# Test n.2
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = expit(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.5))
# 8.898346147374827

# Test n.3
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = expit(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 1))
# 10.563346147374826

# Test n.4
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = expit(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 1))
# 49.346258798303566

# Test n.5
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = expit(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.3))
# 22.86292546497024

# Test n.6
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = expit(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.9))
# 45.56292546497025



#ex06
print("\nex06")
X = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]
])
Y = np.array([1,0,1,1,1,0,0])
Z = np.array([1.2,0.5,-0.32])
print(reg_logistic_grad(X, Y, Z, 1))
#array([ 6.69780169, -0.33235792, 2.71787754])
print(reg_logistic_grad(X, Y, Z, 0.5))
#array([ 6.61208741, -0.3680722, 2.74073468])
print(reg_logistic_grad(X, Y, Z, 0.0))
#array([ 6.52637312, -0.40378649, 2.76359183])



#ex07
print("\nex07")
X = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]
])
Y = np.array([1,0,1,1,1,0,0])
Z = np.array([1.2,0.5,-0.32])
print(vec_reg_logistic_grad(X, Y, Z, 1))
#array([ 6.69780169, -0.33235792, 2.71787754])
print(vec_reg_logistic_grad(X, Y, Z, 0.5))
#array([ 6.61208741, -0.3680722, 2.74073468])
print(vec_reg_logistic_grad(X, Y, Z, 0.0))
#array([ 6.52637312, -0.40378649, 2.76359183])



#ex08
print("\nex08")

# We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
df = pd.read_csv('./resources/data.csv', delimiter=',', header=None, index_col=False)
x, y = np.array(df.iloc[1:, 0:2]).astype(float), df.iloc[1:, 2].astype(float)

#Divide dataset
#indices = list(range(0, len(y)))
#random.shuffle(indices)
#random.shuffle(indices)
index_60pc = int(len(y) / 10 * 6)
index_80pc = int(len(y) / 10 * 8)
#indices_train = indices[:index_60pc]
#indices_cross = indices[index_60pc:index_80pc]
#indices_test = indices[index_80pc:]

#randomization doesn't work and jumbles up data for some absurd reason
x_train = x[:index_60pc]#[indices_train]
y_train = y[:index_60pc]#[indices_train]
x_cross = x[index_60pc:index_80pc]#[indices_cross]
y_cross = y[index_60pc:index_80pc]#[indices_cross]
x_test  = x[index_80pc:]#[indices_test ]
y_test  = y[index_80pc:]#[indices_test ]

"""
# We set our model with our hyperparameters : alpha, verbose and learning_rate
model = LinearRegression(alpha=0.001, n_cycle=100, n_epoch=20, verbose=False, learning_rate_type='constant')
model_ridge = LinearRegressionRidge(alpha=0.001, n_cycle=100, n_epoch=20, verbose=False, learning_rate_type='constant', lambda_=0.001)
model_poly = PolynomialRegression(alpha=0.0001, n_cycle=100, n_epoch=50, verbose=False, learning_rate_type='constant', degree=3)
#model_sklearn = sklearn.LinearRegression()


fig = mpl.figure()
model.plot_data_(x, y, fig, 111, "all")
mpl.show()

model.plot_data_all_(x_train, y_train, x_cross, y_cross, x_test, y_test)

# We fit our model to our dataset and display the score for the train and test datasets
model.train_(x_train, y_train, x_cross, y_cross, x_test, y_test, show_progress=False, show_hyperparameter_stats=False)
model_ridge.train_(x_train, y_train, x_cross, y_cross, x_test, y_test, show_progress=False, show_hyperparameter_stats=False)
model_poly.train_(x_train, y_train, x_cross, y_cross, x_test, y_test, show_progress=False, show_hyperparameter_stats=True)


model = LinearRegression(alpha=0.01, n_cycle=100, n_epoch=12, verbose=True, learning_rate_type='invscaling')
model.set_base_theta_(x_train, x_cross, x_test)
#print(model.theta)

print("RMSE before normalequation:\t" + str(model.rmse_(x_train, y_train)))
print("R2score before normalequation:\t" + str(model.r2score_(x_train, y_train)))
model.normalequation_(X, Y)
print("RMSE after normalequation: " + str(model.rmse_(x_train, y_train)))
print("R2score after normalequation: " + str(model.r2score_(x_train, y_train)))

model_ridge.plot_ridge_trace_(x, y, False)
"""


#ex09
print("\nex09")

# We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
df_train = pd.read_csv('./resources/train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
df_test = pd.read_csv('./resources/test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]


# We set our model with our hyperparameters : alpha, verbose and learning_rate
model = LogisticRegression([], alpha=0.01, n_cycle=100, n_epoch=12, verbose=False, learning_rate_type='constant')
model_ridge = LogisticRegressionRidge([], alpha=0.01, n_cycle=100, n_epoch=12, verbose=False, learning_rate_type='constant', lambda_=0.005)


"""
# We fit our model to our dataset and display the score for the train and test datasets

#model.plot_data_all_(x_train, y_train, x_train, y_train, x_test, y_test)
#print(y_train)
#print(y_test)

model.train_(x_train, y_train, x_train, y_train, x_test, y_test, False)
model_ridge.train_(x_train, y_train, x_train, y_train, x_test, y_test, False)
print(f'Score (normal) on training dataset : {model.score_(x_train, y_train)}')
print(f'Score (ridge ) on training dataset : {model_ridge.score_(x_train, y_train)}')
y_pred = model.predict_class_(x_test)
print(f'Score (normal) on test dataset : {(y_pred == y_test).mean()}')
y_pred = model_ridge.predict_class_(x_test)
print(f'Score (ridge ) on test dataset : {(y_pred == y_test).mean()}')

# epoch 0 : loss 2.711028065632692
# epoch 150 : loss 1.760555094793668
# epoch 300 : loss 1.165023422947427
# epoch 450 : loss 0.830808383847448
# epoch 600 : loss 0.652110347325305
# epoch 750 : loss 0.555867078788320
# epoch 900 : loss 0.501596689945403
# epoch 1050 : loss 0.469145216528238
# epoch 1200 : loss 0.448682476966280
# epoch 1350 : loss 0.435197719530431
# epoch 1500 : loss 0.425934034947101
# Score on train dataset : 0.7591904425539756
# Score on test dataset : 0.7637737239727289

model_ridge.plot_ridge_trace_(x_train, y_train, False)
"""


#ex10
print("\nex10")

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(zscore(X))
#array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559, 0.17240647, -1.89647119])

Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(zscore(Y))
#array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659, 0.28795027, -1.72770165])



#ex11
print("\nex11")

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(minmax(X))
#array([0.58333333, 1. , 0.33333333, 0.77777778, 0.91666667, 0.66666667, 0. ])

Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(minmax(Y))
#array([0.63636364, 1. , 0.18181818, 0.72727273, 0.93939394, 0.6969697 , 0. ])