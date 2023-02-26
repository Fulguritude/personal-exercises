import numpy as np
from ex00 import sum_
from ex01 import mean_mapped_, mean_
from ex02 import variance_
from ex03 import std_deviation_
from ex04 import dot_
from ex05 import mat_vec_prod_
from ex06 import mat_mat_prod_
from ex07 import mse_
from ex08 import vec_mse_
from ex09 import linear_mse_
from ex10 import vec_linear_mse_
from ex11 import gradient_
from ex12 import vec_gradient_

#ex00

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(sum_(X, lambda x: x))
#should be 7.0

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(sum_(X, lambda x: x**2))
#should be 949.0


#ex01
X = np.array([0, 15, -9, 7, 12, 3, -21])
print(mean_(X))
#should be 1.0
X = np.array([0, 15, -9, 7, 12, 3, -21])
print (mean_mapped_(X, lambda x: x ** 2))
#should be 135.57142857142858


#ex02
X = np.array([0, 15, -9, 7, 12, 3, -21])
print(variance_(X))
#134.57142857142858
print(variance_(X/2))
#33.642857142857146


#ex03
X = np.array([0, 15, -9, 7, 12, 3, -21])
print(std_deviation_(X))
#11.600492600378166
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(std_deviation_(Y))
#11.410700312980492


#ex04
X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(dot_(X, Y))
#917.0
print(dot_(X, X))
#949.0
print(dot_(Y, Y))
#915.0


#ex05
W = np.array([
[ -8, 8, -6, 14, 14, -9, -4],
[ 2, -11, -2, -11, 14, -2, 14], [-13, -2, -5, 3, -8, -4, 13],
[ 2, 13, -14, -15, -14, -15, 13], [ 2, -1, 12, 3, -7, -3, -6]])

X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7,1))

print(mat_vec_prod_(W, X))
#array([[ 497], [-356], [-345], [-270], [ -69]])
#printW.dot(X)

print(mat_vec_prod_(W, Y))
#array([[ 452], [-285], [-333], [-182], [-133]])
#W.dot(Y)


#ex06
W = np.array([
	[ -8, 8, -6, 14, 14, -9, -4],
	[ 2, -11, -2, -11, 14, -2, 14],
	[-13, -2, -5, 3, -8, -4, 13],
	[ 2, 13, -14, -15, -14, -15, 13],
	[ 2, -1, 12, 3, -7, -3, -6]
])
Z = np.array([
	[ -6, -1, -8, 7, -8],
	[ 7, 4, 0, -10, -10],
	[ 7, -13, 2, 2, -11],
	[ 3, 14, 7, 7, -4],
	[ -1, -3, -8, -4, -14],
	[ 9, -14, 9, 12, -7],
	[ -9, -4, -10, -3, 6]
])
print(mat_mat_prod_(W, Z))
"""
array([[ 45, 414, -3, -202, -163],
[-294, -244, -367, -79, 62],
[-107, 140, 13, -115, 385],
[-302, 222, -302, -412, 447],
[ 108, -33, 118, 79, -67]])
"""
#W.dot(Z)
print(mat_mat_prod_(Z,W))
"""
array([[ 148, 78, -116, -226, -76, 7, 45],
[ -88, -108, -30, 174, 364, 109, -42],
[-126, 232, -186, 184, -51, -42, -92],
[ -81, -49, -227, -208, 112, -176, 390],
[ 70, 3, -60, 13, 162, 149, -110],
[-207, 371, -323, 106, -261, -248, 83],
[ 200, -53, 226, -49, -102, 156, -225]])
"""
#Z.dot(W)


#ex07
X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(mse_(X, Y))
#4.285714285714286
print(mse_(X, X))
#0.0



#ex08
X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(vec_mse_(X, Y))
#4.285714285714286
print(vec_mse_(X, X))
#0.0


#ex09
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
Z = np.array([3,0.5,-6])
print(linear_mse_(X, Y, Z))
#2641.0

W = np.array([0,0,0])
print(linear_mse_(X, Y, W))
#130.71428571


#ex10

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
Z = np.array([3,0.5,-6])
print(vec_linear_mse_(X, Y, Z))
#2641.0

W = np.array([0,0,0])
print(vec_linear_mse_(X, Y, W))
#130.71428571


#ex11
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
Z = np.array([3,0.5,-6])
print(gradient_(X, Y, Z))
#array([ -37.35714286, 183.14285714, -393. ])

W = np.array([0,0,0])
print(gradient_(X, Y, W))
#array([ 0.85714286, 23.28571429, -26.42857143])
print(gradient_(X, X.dot(Z), Z))
#grad(X, X.dot(Z), Z)
#array([0., 0., 0.])



#ex12

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
Z = np.array([3,0.5,-6])
print(vec_gradient_(X, Y, Z))
#array([ -37.35714286, 183.14285714, -393. ])

W = np.array([0,0,0])
print(vec_gradient_(X, Y, W))
#array([ 0.85714286, 23.28571429, -26.42857143])
print(vec_gradient_(X, X.dot(Z), Z))
#grad(X, X.dot(Z), Z)
#array([0., 0., 0.])
