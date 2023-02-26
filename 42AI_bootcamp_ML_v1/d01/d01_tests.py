import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ex00 import predict_
from ex01 import cost_elems_, cost_
from ex02 import fit_
from ex03 import LinearRegression as lr
from ex04 import plot_lrmodel, plot_lrcost
from ex05 import plot_age_vs_price, plot_thrust_vs_price, plot_tmeters_vs_price, plot_multilinear_regression
from ex06 import plot_normalequation

#ex00
print("ex00")
X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
print(predict_(theta1, X1))
#array([[2], [6], [10], [14.], [18.]])
X2 = np.array([[1], [2], [3], [5], [8]])
theta2 = np.array([[2.]])
print(predict_(theta2, X2))
#Incompatible dimension match between X and theta.
#None
X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta3 = np.array([[0.05], [1.], [1.], [1.]])
print(predict_(theta3, X3))
#array([[22.25], [44.45], [66.65], [88.85]])


#ex01
print("\n\nex01")
X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(cost_elems_(theta1, X1, Y1))
#array([[0.], [0.1], [0.4], [0.9], [1.6]])
print(cost_(theta1, X1, Y1))
#3.0
X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
Y2 = np.array([[19.], [42.], [67.], [93.]])
print(cost_elems_(theta2, X2, Y2))
#array([[1.3203125], [0.7503125], [0.0153125], [2.1528125]])
print(cost_(theta2, X2, Y2))
#4.238750000000004


#ex02
print("\n\nex02")
X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
theta1 = fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000)
print(theta1)
#array([[2.0023..],[3.9991..]])
print(predict_(theta1, X1))
#array([2.0023..], [6.002..], [10.0007..], [13.99988..], [17.9990..])

X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta2 = np.array([[42.], [1.], [1.], [1.]])
theta2 = fit_(theta2, X2, Y2, alpha = 0.0005, n_cycle=2000)#n_cycle=42000)
print(theta2)
#array([[41.99..],[0.97..], [0.77..], [-1.20..]])
print(predict_(theta2, X2))
#array([[19.5937..], [-2.8021..], [-25.1999..], [-47.5978..]])


#ex03
print("\n\nex03")
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
instance_lr = lr(np.array([[1.], [1.], [1.], [1.], [1]]))
print(instance_lr.predict_(X))
#array([[8.], [48.], [323.]])
print(instance_lr.cost_elems_(X,Y))
#array([[37.5], [0.], [1837.5]])
print(instance_lr.cost_(X,Y))
#1875.0
instance_lr.fit_(X, Y, alpha = 1.6e-4, n_cycle=2000)#n_cycle=200000)
print(instance_lr.theta)
#array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]]) >>>mylr.predict_(X)
#array([[23.499..], [47.385..], [218.079...]])
print(instance_lr.cost_elems_(X,Y))
#array([[0.041..], [0.062..], [0.001..]])
print(instance_lr.cost_(X,Y))
#0.1056..



#ex04
print("\n\nex04")
data = pd.read_csv("./resources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)

linear_model1 = lr(np.array([[89.0], [-8]]))
linear_model2 = lr(np.array([[89.0], [-6]]))
#linear_model1.fit_(Xpill, Yscore)
#linear_model2.fit_(Xpill, Yscore)
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

#plot_lrmodel(linear_model1, data, "Micrograms", "Score")
#plot_lrcost(linear_model1, data, "Micrograms", "Score")

print(linear_model1.mse_(Xpill, Yscore))
# 57.60304285714282
print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
print(linear_model2.mse_(Xpill, Yscore))
# 232.16344285714285
print(mean_squared_error(Yscore, Y_model2))
# 232.16344285714285


#ex05
print("\n\nex05")
#plot_age_vs_price()
#plot_thrust_vs_price()
#plot_tmeters_vs_price()
#plot_multilinear_regression()


#ex06
print("\n\nex06")
plot_normalequation()


#ex07
