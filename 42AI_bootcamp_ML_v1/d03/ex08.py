#https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets

import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
#sklearn.linear_model.LinearRegression


from loading import get_current_loadbar

class LinearRegression:
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""

	def __init__(self, theta=[], alpha=0.001, n_cycle=100, n_epoch=10, verbose=False, learning_rate_type='constant'):
		"""
		Description:
			Generator of the class, initialize self.
		Args:
			theta: has to be a list or a numpy array, it is a vector of dimension (number of features + 1, 1).
		Raises:
			This method should not raise any Exception.
		"""
		self.theta = theta
		self.alpha = alpha
		self.n_cycle = n_cycle
		self.n_epoch = n_epoch
		self.verbose = verbose
		self.learning_rate_type = learning_rate_type # can be 'constant' or 'invscaling'
		self.loss_list = []
		self.alpha_list = []
		self.crossval_loss_list = []


	def set_base_theta_(self, X_train, X_cross, X_test, has_left_ones_column=False):
		if has_left_ones_column:
			self.theta = np.ones((X_train.shape[1]))
		else:
			self.theta = np.ones((X_train.shape[1] + 1))

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
		if len(X) == 0 or len(X[0]) == 0 or len(self.theta) == 0:
			#print ("predict error: " + str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(X[0]) != len(self.theta) - 1))
			return None
		if len(X[0]) + 1 == len(self.theta):
			return np.dot(X, self.theta[1:]) + self.theta[0]
		if len(X[0]) == len(self.theta):
			return np.dot(X, self.theta)
		return None

	def loss_elems_(self, X, Y):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		return 0.5 * ((self.predict_(X) - Y) ** 2) / len(Y)

	def loss_(self, X, Y):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		return np.sum(self.loss_elems_(X, Y))

	def gradient_(self, X, Y):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			#print(str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(Y) == 0) + " " + str(len(self.theta) == 0 or len(X) != len(Y)) + " " + str(len(X[0]) != len(self.theta) - 1) + " " + str(len(Y[0]) != 1))
			return None
		Y_hat = self.predict_(X)
		loss_vec = (Y_hat - Y)
		gradient = np.dot(X.T, loss_vec)
		if len(X[0]) + 1 == len(self.theta):
			tmp = np.zeros((len(gradient) + 1))
			tmp[0] = sum(loss_vec)
			tmp[1:] = gradient
			gradient = tmp
		inv_m = 1 / len(loss_vec)
		return gradient * inv_m

	def fit_(self, X, Y, show_progress=False):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		for cycle in range(self.n_cycle):
			if show_progress:
				print(get_current_loadbar(cycle + 1, self.n_cycle), end=("\n" if cycle + 1 == self.n_cycle else "\r"))
			gradient = self.gradient_(X, Y)
			self.theta = self.theta - self.alpha * gradient 
		return self.theta

#	def fit_data_(self, data, x_axis_key, y_axis_key):
#		X = np.array(data[x_axis_key])
#		Y = np.array(data[y_axis_key])
#		if self.verbose:
#			print("Fitting model... \ntheta: " + str(self.theta) + "\nX:" + str(X) + "\nY:" + str(Y))
#		self.fit_(X, Y, self.alpha, self.n_cycle)
#
#	def fit_data_multilinear_(self, data, x_axis_keys, y_axis_key):
#		X = np.array(data[x_axis_keys])
#		Y = np.array(data[[y_axis_key]])
#		print("Fitting " + str(x_axis_keys) + " against " + y_axis_key)
#		print("MSE before fit: " + str(self.mse_(X, Y)))
#		self.fit_(X, Y, self.alpha, self.n_cycle, True)
#		print("MSE after fit: " + str(self.mse_(X, Y)))

	def mse_(self, X, Y):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		Y_hat = self.predict_(X)
		loss_vec = Y_hat - Y
		return np.dot(loss_vec.T, loss_vec) / len(loss_vec)

	def mae_(self, X, Y):
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		Y_hat = self.predict_(X)
		loss_vec = Y_hat - Y
		return sum(math.abs(loss_vec)) / len(loss_vec)

	def rmse_(self, X, Y):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		return math.sqrt(self.mse_(X, Y))

	def r2score_(self, X, Y):
		"""
		Description:
			Calculate the R2score between the predicted output and the output.
			Best possible score is 1.0; bad scores are near 0
		Args:
			X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
			Y: has to be a numpy.ndarray, a vector of dimension (number of training examples, 1).
		Returns:
			r2score: has to be a float.
			None if there is a matching dimension problem.
		Raises:
			This function should not raise any Exception.
		"""
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		Y_mean = np.mean(Y)
		total_sum_squares = np.sum((Y - Y_mean) ** 2)
		Y_hat = self.predict_(X)
		regression_sum_squares = np.sum((Y_hat - Y_mean) ** 2)
		return regression_sum_squares / total_sum_squares

#	def score_(self, x, y_true):
#		"""
#		Returns the mean accuracy on the given test data and labels.
#		Arg:
#			x: a 1d or 2d numpy ndarray for the samples
#			y: a scalar or a numpy ndarray for the correct labels
#		Returns:
#			Mean accuracy of self.predict(x_train) with respect to y_true
#			None on any error.
#		Raises:
#			This method should not raise any Exception.
#		"""
#		y_pred = self.predict_class_(x)		
#		if len(y_pred) != len(y_true):
#			return None
#		#print("y_pred.shape: " + str(y_pred.shape) + "\ny_true.shape: " + str(y_pred.shape))
#		return (y_pred == y_true).mean()

	def training_setup_(self, X_train, X_cross, X_test):
		print("\nSetup for training of linear model...")
		self.set_base_theta_(X_train, X_cross, X_test)
		#diagnostics:
		for i in range(len(X_train[0])):
			column = X_train[:, i]
			print("X_train column " + str(i) + ":\n")
			print("\t- mode    : " + str(stats.mode(column)))
			print("\t- mean    : " + str(np.mean(column)))
			print("\t- median  : " + str(np.median(column)))
			print("\t- variance: " + str(np.var(column)))
			#print("\t- stddev  : " + str(stats.stddev(column)))
			print("\t- min     : " + str(np.min(column)))
			print("\t- max     : " + str(np.max(column)))
			#print("\t- range   : " + str(stats.range(column)))

			column = X_cross[:, i]
			print("X_cross column " + str(i) + ":\n")
			print("\t- mode    : " + str(stats.mode(column)))
			print("\t- mean    : " + str(np.mean(column)))
			print("\t- median  : " + str(np.median(column)))
			print("\t- variance: " + str(np.var(column)))
			#print("\t- stddev  : " + str(stats.stddev(column)))
			print("\t- min     : " + str(np.min(column)))
			print("\t- max     : " + str(np.max(column)))
			#print("\t- range   : " + str(stats.range(column)))

			column = X_test [:, i]
			print("X_test  column " + str(i) + ":\n")
			print("\t- mode    : " + str(stats.mode(column)))
			print("\t- mean    : " + str(np.mean(column)))
			print("\t- median  : " + str(np.median(column)))
			print("\t- variance: " + str(np.var(column)))
			#print("\t- stddev  : " + str(stats.stddev(column)))
			print("\t- min     : " + str(np.min(column)))
			print("\t- max     : " + str(np.max(column)))
			#print("\t- range   : " + str(stats.range(column)))

	def training_handle_epoch_(self, X, Y, epoch, show_progress=False):
		new_loss = self.loss_(X, Y)
		self.loss_list.append(new_loss)
		self.fit_(X, Y, show_progress)
		#print("theta: " + str(self.theta))
		if self.learning_rate_type == 'invscaling':
			new_alpha = 1 / new_loss
			self.alpha = new_alpha
			self.alpha_list.append(new_alpha)
		if self.verbose:
			print("epoch " + str(epoch) + ": \tloss = " + str(self.loss_list[-1]))

	def train_(self, X_train, Y_train, X_cross, Y_cross, X_test, Y_test, show_progress=False, show_hyperparameter_stats=False):
		#TODO dimension checks
		self.training_setup_(X_train, X_cross, X_test)
		#print("testing..")
		#print(X)
		#print(Y)
		#print(self.theta)
		for epoch in range(self.n_epoch):
			self.crossval_loss_list = self.crossval_loss_list + [self.loss_(X_cross, Y_cross)]
			self.training_handle_epoch_(X_train, Y_train, epoch, show_progress)
		print("Score on training dataset:\trmse: " + str(self.rmse_(X_train, Y_train)) + "\t| r2_score: " + str(self.r2score_(X_train, Y_train)))
		print("Score on crossval dataset:\trmse: " + str(self.rmse_(X_cross, Y_cross)) + "\t| r2_score: " + str(self.r2score_(X_cross, Y_cross)))
		print("Score on test     dataset:\trmse: " + str(self.rmse_(X_test , Y_test )) + "\t| r2_score: " + str(self.r2score_(X_test , Y_test )))
		print("Final theta: " + str(self.theta) + "\t| norm: " + str(np.linalg.norm(self.theta)) + "\t| average spectral dist: " + str(np.linalg.norm(self.theta) / len(self.theta)))
		if show_hyperparameter_stats:
			self.plot_learning_()

	def plot_data_(self, X, Y, fig, int_code, label):
		ax = fig.add_subplot(int_code, projection="3d", label=label)
		X_1 = X.T[0]
		X_2 = X.T[1]
		ax.scatter(X_1, X_2, Y, c='r', marker='o')
		ax.set_xlabel("x1")
		ax.set_ylabel("x2")
		ax.set_zlabel("y")

	def plot_data_all_(self, X_train, Y_train, X_cross, Y_cross, X_test, Y_test):
		fig = mpl.figure()
		self.plot_data_(X_train, Y_train, fig, 131, "train")
		self.plot_data_(X_cross, Y_cross, fig, 132, "cross")
		self.plot_data_(X_test , Y_test , fig, 133, "test ")
		mpl.show()

	def plot_learning_(self):
		fig = mpl.figure()
		ax = fig.add_subplot(311)
		ax.plot(list(range(self.n_epoch)), self.loss_list, color="red", linewidth=3, label="Loss(epoch)")
		if self.learning_rate_type == "invscaling":
			ay = fig.add_subplot(312)
			ay.plot(list(range(self.n_epoch)), self.alpha_list, color="green", linewidth=3, label="Alpha(epoch)")
		az = fig.add_subplot(313)
		az.plot(list(range(self.n_epoch)), self.crossval_loss_list, color="purple", linewidth=3, label="CrossValLoss(epoch)")
		mpl.show()

	def plot_model_(self, data, x_axis_key, y_axis_key, point_color="lightblue", line_color="green"):
		X = np.array(data[x_axis_key])
		Y = np.array(data[y_axis_key])
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

	def normalequation_(self, X, Y):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		X_trans = X.T
		#print(np.dot(X_trans, X))
		inv_xT_x = np.linalg.inv(np.dot(X_trans, X))
		xT_y = np.dot(X_trans, Y)
		#print(self.theta)
		self.theta = np.dot(inv_xT_x, xT_y)
		#print(self.theta)
		return self.theta

	def normalequation_data_(self, data, x_axis_keys, y_axis_key):
		X = np.array(data[x_axis_keys])
		Y = np.array(data[[y_axis_key]])
		print("Fitting " + str(x_axis_keys) + " against " + y_axis_key)
		print("MSE before normalequation: " + str(self.mse_(X, Y)))
		self.normalequation_(X, Y)
		print("New theta:" + str(self.theta))
		print("MSE after normalequation: " + str(self.mse_(X, Y)))



class LinearRegressionRidge(LinearRegression):

	def __init__(self, theta=[], alpha=0.001, n_cycle=100, n_epoch=10, verbose=False, learning_rate_type='constant', lambda_=0.01):
		super().__init__(theta, alpha, n_cycle, n_epoch, verbose, learning_rate_type)
		self.lambda_ = lambda_

	def loss_elems_(self, X, Y):
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		if len(X[0]) == len(self.theta):
			return super().loss_elems_(X, Y) + self.lambda_ * 0.5 * np.dot(self.theta.T, self.theta) / len(Y)
		if len(X[0]) + 1 == len(self.theta):
			return super().loss_elems_(X, Y) + self.lambda_ * 0.5 * np.dot(self.theta[1:].T, self.theta[1:]) / len(Y)

	def regularized_gradient_(self, X, Y):
		"""
		Computes the regularized linear gradient of three non-empty numpy.ndarray, with two for-loop. The three arrays must have compatible dimensions.
		Args:
			y: has to be a numpy.ndarray, a vector of dimension m * 1.
			x: has to be a numpy.ndarray, a matrix of dimesion m * n.
			theta: has to be a numpy.ndarray, a vector of dimension n * 1. 
			lambda_: has to be a float.
		Returns:
			A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
			None if y, x, or theta are empty numpy.ndarray.
			None if y, x or theta does not share compatibles dimensions.
		Raises:
			This function should not raise any Exception.
		"""
		gradient = super().gradient_(X, Y)
		reg_param = (self.lambda_ * self.theta) / len(Y)
		return gradient + reg_param

	def fit_(self, X, Y, show_progress=False):
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
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		for cycle in range(self.n_cycle):
			if show_progress:
				print(get_current_loadbar(cycle + 1, self.n_cycle), end=("\n" if cycle + 1 == self.n_cycle else "\r"))
			gradient = self.regularized_gradient_(X, Y)
			self.theta = self.theta - self.alpha * gradient
		return self.theta

	def training_setup_(self, X_train, X_cross, X_test):
		print("\nSetup for training of ridge linear model...")
		self.set_base_theta_(X_train, X_cross, X_test)

#	def training_handle_epoch_(self, X, Y, epoch, show_progress=False):
#		super().training_handle_epoch_(X, Y, epoch, show_progress)

	def plot_ridge_trace_(self, X, Y, show_progress=False):
		theta_list = []
		lambda_list = []
		final_loss_list = []
		for i in range(-5, 7):
			self.lambda_ = 0.5 * 10 ** i
			lambda_list = lambda_list + [self.lambda_]
			print("\nTraining ridge model for lambda = " + str(self.lambda_))
			self.train_(X, Y, X, Y, X, Y, show_progress)
			theta_list.append(self.theta)
			final_loss_list.append(self.loss_list[-1])
		fig = mpl.figure()

		#ridge trace
		ax = fig.add_subplot(211)
		ax.set_xscale('log')
		ax.set_xlim(lambda_list[0] / 10, lambda_list[-1] * 10)
		##ax.set_ylim(-5, 5)
		#print(np.array(theta_list))
		#print(lambda_list)
		theta_mat = np.array(theta_list).T
		#print(theta_mat)
		for i in range(len(theta_mat)):
			theta_coef_evolution = theta_mat[i]
			line_color = i * 20 / 255
			line_color = 0.75 if i > 0.75 else (0. if i < 0. else i)
			line_color = (line_color, line_color, line_color)
			ax.plot(lambda_list, theta_coef_evolution, color=line_color, lw=3, label="Line theta_" + str(i))

		#performance
		ay = fig.add_subplot(212)
		ay.set_xscale('log')
		ay.set_xlim(lambda_list[0] / 10, lambda_list[-1] * 10)
		ay.plot(lambda_list, final_loss_list, color="red", lw=3, label="FinalLoss(lambda)")

		mpl.legend()
		mpl.show()

	#def train_(self, X, Y, X_test, Y_test, show_progress=False, show_hyperparameter_stats=False):



class PolynomialRegression(LinearRegression):

	def __init__(self, theta=[], alpha=0.001, n_cycle=100, n_epoch=10, verbose=False, learning_rate_type='constant', degree=3):
		super().__init__(theta, alpha, n_cycle, n_epoch, verbose, learning_rate_type)
		self.degree = degree
		self.X_train = np.zeros((1,1))
		self.X_test = np.zeros((1,1))

	def set_base_theta_(self, X_train, X_cross, X_test, has_left_ones_column=False):
		if has_left_ones_column:
			param_amount = X_train.shape[1] - 1
			self.theta = np.ones((param_amount * self.degree + 1))
			self.X_train = np.zeros((len(X_train), len(self.theta)))
			self.X_cross = np.zeros((len(X_cross), len(self.theta)))
			self.X_test  = np.zeros((len(X_test ), len(self.theta)))
			self.X_train[:, :param_amount + 1] = X_train
			self.X_cross[:, :param_amount + 1] = X_cross
			self.X_test [:, :param_amount + 1] = X_test 
			for i in range(1, self.degree):
				self.X_train[:, 1 + param_amount * i: 1 + param_amount * (i + 1)] = self.X_train[:, 1:1 + param_amount] ** (i + 1)
				self.X_cross[:, 1 + param_amount * i: 1 + param_amount * (i + 1)] = self.X_cross[:, 1:1 + param_amount] ** (i + 1)
				self.X_test [:, 1 + param_amount * i: 1 + param_amount * (i + 1)] = self.X_test [:, 1:1 + param_amount] ** (i + 1)
		else:
			param_amount = X_train.shape[1]
			self.theta = np.ones((param_amount * self.degree + 1))
			self.X_train = np.zeros((len(X_train), len(self.theta)))
			self.X_cross = np.zeros((len(X_cross), len(self.theta)))
			self.X_test  = np.zeros((len(X_test ), len(self.theta)))
			self.X_train[:, 0] = np.ones(len(X_train))
			self.X_cross[:, 0] = np.ones(len(X_cross))
			self.X_test [:, 0] = np.ones(len(X_test ))
			self.X_train[:, 1:param_amount + 1] = X_train
			self.X_cross[:, 1:param_amount + 1] = X_cross
			self.X_test [:, 1:param_amount + 1] = X_test
			for i in range(1, self.degree):
				self.X_train[:, 1 + param_amount * i : 1 + param_amount * (i + 1)] = self.X_train[:, 1:1 + param_amount] ** (i + 1)
				self.X_cross[:, 1 + param_amount * i : 1 + param_amount * (i + 1)] = self.X_cross[:, 1:1 + param_amount] ** (i + 1)
				self.X_test [:, 1 + param_amount * i : 1 + param_amount * (i + 1)] = self.X_test [:, 1:1 + param_amount] ** (i + 1)
			
			#self.X_train = self.X_train[:, [0, 1, 2, 4, 5]]
			#self.X_cross = self.X_cross[:, [0, 1, 2, 4, 5]]
			#self.X_test  = self.X_test [:, [0, 1, 2, 4, 5]]
			#self.theta = np.array([1, 1, 1, 1, 1])

	def training_setup_(self, X_train, X_cross, X_test):
		print("\nSetup for training of polynomial model...")
		self.set_base_theta_(X_train, X_cross, X_test)

	def train_(self, X_train, Y_train, X_cross, Y_cross, X_test, Y_test, show_progress=False, show_hyperparameter_stats=False):
		self.training_setup_(X_train, X_cross, X_test)
		X_train = self.X_train
		X_cross = self.X_cross
		X_test  = self.X_test 
		for epoch in range(self.n_epoch):
			self.crossval_loss_list = self.crossval_loss_list + [self.loss_(X_cross, Y_cross)]
			self.training_handle_epoch_(X_train, Y_train, epoch, show_progress)
		print("Score on training dataset:\trmse: " + str(self.rmse_(X_train, Y_train)) + "\t| r2_score: " + str(self.r2score_(X_train, Y_train)))
		print("Score on crossval dataset:\trmse: " + str(self.rmse_(X_cross, Y_cross)) + "\t| r2_score: " + str(self.r2score_(X_cross, Y_cross)))
		print("Score on test	 dataset:\trmse: " + str(self.rmse_(X_test , Y_test )) + "\t| r2_score: " + str(self.r2score_(X_test , Y_test )))
		print("Final theta: " + str(self.theta) + "\t| norm: " + str(np.linalg.norm(self.theta)) + "\t| average spectral dist: " + str(np.linalg.norm(self.theta) / len(self.theta)))
		if show_hyperparameter_stats:
			self.plot_learning_()
