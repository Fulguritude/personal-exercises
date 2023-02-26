import numpy as np
import matplotlib.pyplot as mpl

from ex03 import LinearRegression as lr


def plot_lrmodel(lr, data, x_axis_key, y_axis_key):
	X = data[x_axis_key]
	Y = data[y_axis_key]
	fig = mpl.figure()
	ax = fig.add_subplot(111)
	ax.set_xlim(min(X) - 1, max(X) + 1)
	ax.set_ylim(min(Y) - 1, max(Y) + 1)
	ax.scatter(X, Y, color="lightblue")
	X_hat = np.arange(0, 7, 0.5)
	X_hat = X_hat.reshape((len(X_hat), 1))
	Y_hat = lr.predict_(X_hat)
	#print("X_hat: " + str(X_hat) + "\nY_hat:" + str(Y_hat) + "\n")
	ax.plot(X_hat, Y_hat, color="green", linewidth=3)
	mpl.show()

def plot_lrcost(lr, data, x_axis_key, y_axis_key):
	X = np.array(data[x_axis_key]).reshape(-1, 1)
	Y = np.array(data[y_axis_key]).reshape(-1, 1)
	fig = mpl.figure()
	ax = fig.add_subplot(111)
	ax.set_xlim(-15, 5)
	ax.set_ylim(0, 150)
	def set_curve(theta_0):
		theta_1s = np.arange(-15, 5, 0.1)
		costs = map(lambda t: 0.5 * sum(((t * X + theta_0 * np.ones(X.shape) - Y) ** 2) / len(Y)), theta_1s)
		color_ = ((theta_0 * 2) / 255, (theta_0 * 2) / 255, (theta_0 * 2) / 255)
		ax.plot(theta_1s, list(costs), color=color_, linewidth=3)
	ax.scatter(np.ones((1,1)) * lr.theta[1], np.ones((1,1)) * lr.cost_(X, Y), color="red")
	theta_0s = [lr.theta[0][0]]
	print(theta_0s[0])
	for i in range(40, 120, 10):
		theta_0s.append(i)
	for theta_0 in theta_0s:
		set_curve(theta_0)
	mpl.show()
