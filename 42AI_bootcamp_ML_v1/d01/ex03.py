import math
import numpy as np
import matplotlib.pyplot as mpl
from loading import get_current_loadbar
#sklearn.linear_model.LinearRegression

class LinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""

	def __init__(self, theta):
		"""
		Description:
			Generator of the class, initialize self.
		Args:
			theta: has to be a list or a numpy array, it is a vector of dimension (number of features + 1, 1).
		Raises:
			This method should not raise any Exception.
		"""
		self.theta = theta


	def predict_(self, X):
		"""
		Description:
			Prediction of output using the hypothesis function (linear model).
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
			X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
		Returns:
			pred: numpy.ndarray, a vector of dimension (number of the training examples,1).
			None if X does not match the dimension of theta.
		Raises:
			This function should not raise any Exception.
		"""
		if len(X) == 0 or len(X[0]) == 0 or len(X[0]) != len(self.theta) - 1:
			print ("predict error: " + str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(X[0]) != len(self.theta) - 1))
			return None
		return np.dot(X, self.theta[1:]) + self.theta[0]

	def cost_elems_(self, X, Y):
		"""
		Description:
			Calculates all the elements 0.5*M*(y_pred - y)^2 of the cost function.
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
			X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
			Y: has to be a numpy.ndarray, a matrix of dimensions (number of training examples, 1)
		Returns:
			J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
			None if there is a dimension matching problem between X, Y or theta.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
			len(X) != len(Y) or len(X[0]) != len(self.theta) - 1 or len(Y[0]) != 1):
			return None
		return 0.5 * ((self.predict_(X) - Y) ** 2) / len(Y)


	def cost_(self, X, Y):
		"""
		Description:
			Calculates the value of cost function.
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
			X: has to be a numpy.ndarray, a vector of dimension (number of training examples, number of features).
			Y: has to be a numpy.ndarray, a matrix of dimensions (number of training examples, 1)
		Returns:
			J_value : has to be a float.
			None if X does not match the dimension of theta.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
			len(X) != len(Y) or len(X[0]) != len(self.theta) - 1 or len(Y[0]) != 1):
			return None
		return np.sum(self.cost_elems_(X, Y))

	def vec_gradient_(self, X, Y):
		"""
		Computes a gradient vector. The two arrays must have the compatible dimensions.
		NB: this function get the gradient by minimizing the error as much as possible
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension n.
			X: has to be a numpy.ndarray, a matrix of dimension m * n.
			Y: has to be a numpy.ndarray, a vector of dimension m.
		Returns:
			The gradient as a numpy.ndarray, a vector of dimensions n * 1.
			None if x, y, or theta are empty numpy.ndarray.
			None if x, y and theta do not have compatible dimensions.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
			len(X) != len(Y) or len(X[0]) != len(self.theta) - 1 or len(Y[0]) != 1):
			#print(str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(Y) == 0) + " " + str(len(X) != len(Y)) + " " + str(len(X[0]) != len(self.theta) - 1) + " " + str(len(Y[0]) != 1))
			return None
		Y_hat = self.predict_(X).reshape(-1, 1)
		loss_vec = (Y_hat - Y)
		gradient = np.dot(np.transpose(X), loss_vec) / len(Y)
		result = np.ones((len(gradient) + 1, 1))
		result[0, 0] = sum(loss_vec)[0]
		result[1:, 0] = gradient[:, 0]
		return result

	def fit_(self, X, Y, alpha = 0.001, n_cycle = 3000, show_progress=False):
		"""
		Description:
			Performs a fit of Y(output) with respect to X.
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
			X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
			Y: has to be a numpy.ndarray, a vector of dimension (number of training examples, 1).
		Returns:
			new_theta: numpy.ndarray, a vector of dimension (number of the features +1, 1).
			None if there is a matching dimension problem.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
			len(X) != len(Y) or len(X[0]) != len(self.theta) - 1 or len(Y[0]) != 1):
			return None
		for cycle in range(n_cycle):
			if show_progress:
				print(get_current_loadbar(cycle, n_cycle), end=("\n" if cycle == n_cycle - 1 else "\r"))
			gradient = self.vec_gradient_(X, Y)
			new_theta = self.theta - alpha * gradient
			self.theta = new_theta
		return self.theta

	def fit_data_(self, data, x_axis_key, y_axis_key, alpha = 0.001, n_cycle = 3000, verbose=False):
		X = np.array(data[x_axis_key]).reshape(-1, 1)
		Y = np.array(data[y_axis_key]).reshape(-1, 1)
		if verbose:
			print("Fitting model... \ntheta: " + str(self.theta) + "\nX:" + str(X) + "\nY:" + str(Y))
		self.fit_(X, Y, alpha, n_cycle)

	def fit_data_multilinear_(self, data, x_axis_keys, y_axis_key, alpha = 0.0001, n_cycle = 30000):
		X = np.array(data[x_axis_keys])
		Y = np.array(data[[y_axis_key]])
		print("Fitting " + str(x_axis_keys) + " against " + y_axis_key)
		print("MSE before fit: " + str(self.mse_(X, Y)))
		self.fit_(X, Y, alpha, n_cycle, True)
		print("MSE after fit: " + str(self.mse_(X, Y)))

	def mse_(self, x, y):
		"""
		Computes the mean squared error of three non-empty numpy.ndarray,
			without any for-loop. The three arrays must have compatible dimensions.
		Args:
			y: has to be an numpy.ndarray, a vector of dimension m * 1.
			x: has to be an numpy.ndarray, a matrix of dimesion m * n.
			theta: has to be an numpy.ndarray, a vector of dimension n * 1.
		Returns:
			The mean squared error as a float.
			None if y, x, or theta are empty numpy.ndarray.
			None if y, x or theta does not share compatibles dimensions.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(x) == 0 or len(x[0]) == 0 or len(y) == 0 or len(self.theta) == 0
			or len(x[0]) != len(self.theta) - 1 or len(x) != len(y)):
			return None
		y_hat = self.predict_(x)
		dist_vec = y_hat - y
		return np.dot(np.transpose(dist_vec), dist_vec)[0][0] / len(y)

	def mae_(self, x, y):
		if (len(x) == 0 or len(x[0]) == 0 or len(y) == 0 or len(self.theta) == 0
			or len(x[0]) != len(self.theta) - 1 or len(x) != len(y)):
			return None
		y_hat = self.predict_(x)
		dist_vec = y_hat - y
		return sum(math.abs(dist_vec)) / len(y)

	def rmse_(self, x, y):
		"""
		Description:
			Calculate the RMSE between the predicted output and the real output.
		Args:
			X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
			Y: has to be a numpy.ndarray, a vector of dimension (number of training examples, 1).
		Returns:
			rmse: has to be a float.
			None if there is a matching dimension problem.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(x) == 0 or len(x[0]) == 0 or len(y) == 0 or len(self.theta) == 0
			or len(x[0]) != len(self.theta) - 1 or len(x) != len(y)):
			return None
		return math.sqrt(self.mse_(x, y))

	def r2score_(self, x, y):
		"""
		Description:
			Calculate the R2score between the predicted output and the output.
		Args:
			X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
			Y: has to be a numpy.ndarray, a vector of dimension (number of training examples, 1).
		Returns:
			r2score: has to be a float.
			None if there is a matching dimension problem.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(x) == 0 or len(x[0]) == 0 or len(y) == 0 or len(self.theta) == 0
			or len(x[0]) != len(self.theta) - 1 or len(x) != len(y)):
			return None
		y_mean = np.mean(y)
		total_sum_squares = np.sum((y - y_mean) ** 2)
		y_hat = self.predict_(x)
		regression_sum_squares = np.sum((y_hat - y_mean) ** 2)
		return regression_sum_squares / total_sum_squares


	def plot_model_(self, data, x_axis_key, y_axis_key, point_color="lightblue", line_color="green"):
		X = np.array(data[x_axis_key]).reshape(-1, 1)
		Y = np.array(data[y_axis_key]).reshape(-1, 1)
		fig = mpl.figure()
		ax = fig.add_subplot(111)
		ax.set_xlim(min(X) - 1, max(X) + 1)
		ax.set_ylim(min(Y) - 1, max(Y) + 1)
		ax.scatter(X, Y, color=point_color)
		X_hat = np.arange(min(X), max(X), (max(X) - min(X)) / 200)
		X_hat = X_hat.reshape((len(X_hat), 1))
		Y_hat = self.predict_(X_hat)
		#print("X_hat: " + str(X_hat) + "\nY_hat:" + str(Y_hat) + "\n")
		ax.plot(X_hat, Y_hat, color=line_color, linewidth=3)
		mpl.show()

	def plot_model_multilinear_(self, data, x_axis_keys, shown_x_axis_key, y_axis_key, base_color="lightblue", prediction_color="green"):
		X = np.array(data[x_axis_keys])
		X_shown = np.array(data[[shown_x_axis_key]])
		Y = np.array(data[[y_axis_key]])
		fig = mpl.figure()
		ax = fig.add_subplot(111)
		ax.set_xlim(min(X_shown) - 10, max(X_shown) + 10)
		ax.set_ylim(min(Y) - 100, max(Y) + 100)
		ax.scatter(X_shown, Y, color=base_color)
		Y_hat = self.predict_(X)
		ax.scatter(X_shown, Y_hat, color=prediction_color)
		mpl.show()

	def normalequation_(self, x, y):
		"""
		Description:
			Also called Ordinary Least Squares.
			(xT x)^-1 xT is called the Moore-Penrose inverse of a matrix
			Perform the normal equation to get the theta parameters of the hypothesis h and stock them in self.theta.
		Args:
			X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features)
			Y: has to be a numpy.ndarray, a vector of dimension (number of training examples,1)
		Returns:
			Returns self.theta
		Raises:
			This method should not raise any Exceptions.
		"""		
		if (len(x) == 0 or len(x[0]) == 0 or len(y) == 0 or len(self.theta) == 0
			or len(x[0]) != len(self.theta) - 1 or len(x) != len(y)):
			return None
		x_tmp = x
		x = np.ones((len(x), len(x[0]) + 1))
		x[:, 1:] = x_tmp 
		x_trans = np.transpose(x)
		print(np.dot(x_trans, x))
		inv_xT_x = np.linalg.inv(np.dot(x_trans, x))
		xT_y = np.dot(x_trans, y)
		self.theta = np.dot(inv_xT_x, xT_y)
		return self.theta

	def normalequation_data_(self, data, x_axis_keys, y_axis_key):
		X = np.array(data[x_axis_keys])
		Y = np.array(data[[y_axis_key]])
		print("Fitting " + str(x_axis_keys) + " against " + y_axis_key)
		print("MSE before normalequation: " + str(self.mse_(X, Y)))
		self.normalequation_(X, Y)
		print("New theta:" + str(self.theta))
		print("MSE after normalequation: " + str(self.mse_(X, Y)))



