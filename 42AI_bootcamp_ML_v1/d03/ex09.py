#https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_logistic.py

import numpy as np
import matplotlib.pyplot as mpl
from scipy.special import expit

from ex08 import LinearRegression, LinearRegressionRidge
from loading import get_current_loadbar


class LogisticRegression(LinearRegression):

	def __init__(self, theta=[], alpha=0.001, n_cycle=100, n_epoch=10, verbose=False, learning_rate_type='constant'):
		super().__init__(theta, alpha, n_cycle, n_epoch, verbose, learning_rate_type)

#	@staticmethod
#	def sigmoid_(x):
#		"""
#		Compute the sigmoid of a scalar or a list.
#		Args:
#			x: a scalar or list
#		Returns:
#			The sigmoid value as a scalar or list.
#			None on any error.
#		Raises:
#			This function should not raise any Exception.
#		"""
#		if isinstance(x, list):
#			return list(map(lambda t: 1 / (1 + np.exp(-t)), x))
#		return 1 / (1 + np.exp(-x))

#	def set_base_theta_(self, x):
#		self.theta = np.ones((x.shape[1] + 1, 1))

	def predict_prob_(self, X):
		"""
		Predict probability for class labels for samples in a dataset X.
		Arg:
			X: a 1d or 2d numpy ndarray for the samples
		Returns:
			y_pred, the predicted class label per sample.
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		#X = LogisticRegression.np_matrix_from_any(X)
		#print("len X[0]: " + str(len(X[0])) + " | len(theta): " + str(len(self.theta)))
		if len(X) == 0 or len(X[0]) == 0 or len(self.theta) == 0:
			print("LogisticRegression.predict_prob_: " + str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(X[0]) != len(self.theta) - 1))
			return None
		if len(X[0]) + 1 == len(self.theta):	
			return expit(np.dot(X, self.theta[1:]) + self.theta[0])
		if len(X[0]) == len(self.theta):
			return expit(np.dot(X, self.theta))

	def predict_class_(self, X):
		"""
			Predicts a class label for each element
		"""
		return ((self.predict_prob_(X) > 0.5) * 1)

	def predict_(self, X):
		return self.predict_prob_(X)

	def loss_elems_(self, X, Y_true, eps=1e-15):
		"""
		Description:
			Calculates logistic loss value for all element.
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
		Y_pred = self.predict_prob_(X)
		if len(Y_pred) != len(Y_true):
			print(str(len(Y_pred[0]) != m) + " " + str(len(Y_true[0]) != m))
			return None
		Y_pred = [(eps if Y_pred_i <= 0 else (1 - eps if Y_pred_i >= 1 else Y_pred_i)) for Y_pred_i in Y_pred]
		Y_pred = np.array(Y_pred)
		log_Ypred = np.log(Y_pred)
		log_1minus_Ypred = np.log(1 - Y_pred)
		inv_m = 1 / len(Y_true)
		return (-(Y_true * log_Ypred + (1 - Y_true) * log_1minus_Ypred)) * inv_m

	def training_setup_(self, X_train, X_cross, X_test):
		print("\nSetup for training of logistic model...")
		self.set_base_theta_(X_train, X_cross, X_test)

	def loss_(self, X, Y_true, eps=1e-15):
		"""
		Description:
			Computes the logistic loss value, or cross-entropy loss.
		Args:
			y_true: a scalar or a list for the correct labels
			y_pred: a scalar or a list for the predicted labels
			m: the length of y_true (should also be the length of y_pred)
			eps: machine precision epsilon (default=1e-15)
		Returns:
			The logistic loss value as a float.
			None on any error.
		Raises:
			This function should not raise any Exception.
		"""
		return sum(self.loss_elems_(X, Y_true, eps))

#	def gradient_(self, X, Y_true):
#		"""
#		Computes the gradient.
#		Args:
#			x: a 1d or 2d numpy ndarray for the samples
#			y_true: a scalar or a numpy ndarray for the correct labels
#			y_pred: a scalar or a numpy ndarray for the predicted labels
#		Returns:
#			The gradient as a scalar or a numpy ndarray of the width of x.
#			None on any error.
#		Raises:
#			This function should not raise any Exception.
#		"""
#		Y_pred = self.predict_prob_(X)
#		#print("y_true: " + str(len(Y_true)) + " " + str(Y_true))
#		#print("y_pred: " + str(len(Y_pred)) + " " + str(Y_pred))		
#		if len(Y_pred) != len(Y_true) or len(x) != len(Y_true):
#			print("LogisticRegression.gradient_: " + str(len(Y_pred) != len(Y_true)) + " " + str(len(X) != len(Y_true)))
#			return None
#		loss_vec = (Y_pred - Y_true)
#		gradient = np.dot(X.T, loss_vec)
#		if len(X[0]) + 1 == len(self.theta):
#			tmp = np.zeros((len(gradient) + 1))
#			tmp[0] = sum(loss_vec)
#			tmp[1:] = gradient
#			gradient = tmp
#		inv_m = 1 / len(loss_vec)
#		return gradient * inv_m

#	def fit_(self, X, Y, show_progress=False):
#		"""
#		Fit the model according to the given training data. Args:
#		X: a 1d or 2d numpy ndarray for the samples
#		Y: a scalar or a numpy ndarray for the correct labels Returns:
#		self : object
#		None on any error. Raises:
#		This method should not raise any Exception.
#		"""
#		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
#			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
#			print("LogisticRegression.fit_: " + str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(Y) == 0) + " " + str(len(X) != len(Y)) + " " + str(len(Y[0]) != 1))
#			return None
#		for cycle in range(self.n_cycle):
#			if show_progress:
#				print(get_current_loadbar(cycle + 1, self.n_cycle), end=("\n" if cycle + 1 == self.n_cycle else "\r"))
#			gradient = self.gradient_(X, Y)
#			self.theta = self.theta - self.alpha * gradient
#		return self.theta

	def score_(self, X, Y_true):
		"""
		Returns the mean accuracy on the given test data and labels.
		Arg:
			x: a 1d or 2d numpy ndarray for the samples
			y: a scalar or a numpy ndarray for the correct labels
		Returns:
			Mean accuracy of self.predict(x_train) with respect to y_true
			None on any error.
		Raises:
			This method should not raise any Exception.
		"""
		Y_pred = self.predict_class_(X)		
		if len(Y_pred) != len(Y_true):
			return None
		#print("Y_pred.shape: " + str(Y_pred.shape) + "\nY_true.shape: " + str(Y_pred.shape))
		return (Y_pred == Y_true).mean()

#	def train_(self, X, Y, X_test, Y_test, show_progress=False, show_hyperparameter_stats=False):
#		#TODO dimension checks
#		self.set_base_theta_(X)
#		step_list = np.arange(0, self.n_epoch, int(self.n_epoch / 8))
#		X = LogisticRegression.np_matrix_from_any(X)
#		Y = LogisticRegression.np_matrix_from_any(Y).T
#		#print("Y: " + str(Y))
#		X_test = LogisticRegression.np_matrix_from_any(X_test)
#		Y_test = LogisticRegression.np_matrix_from_any(Y_test).T
#		for epoch in range(self.n_epoch):
#			new_loss = self.loss_(X, Y, len(Y))
#			self.loss_list.append(new_loss)
#			self.fit_(X, Y, show_progress)
#			#print("theta: " + str(self.theta))
#			if self.learning_rate_type == 'invscaling':
#				new_alpha = 1 / new_loss
#				self.alpha = new_alpha
#				self.alpha_list.append(new_alpha)
#			if self.verbose and len(step_list) > 0 and epoch == int(step_list[0]):
#				print("epoch " + str(epoch) + ": \tloss = " + str(self.loss_list[-1]))
#				step_list =  step_list[1:]
#		print("Score on training dataset:\t" + str(self.score_(X, Y)))
#		print("Score on test	 dataset:\t" + str(self.score_(X_test, Y_test)))
#		if show_hyperparameter_stats:
#			self.plot_learning_()

	def train_(self, X_train, Y_train, X_cross, Y_cross, X_test, Y_test, show_progress=False, show_hyperparameter_stats=False):
		super().train_(X_train, Y_train, X_cross, Y_cross, X_test, Y_test, show_progress, show_hyperparameter_stats)
		print("Score for class labels on training dataset: " + str(self.score_(X_train, Y_train)))
		print("Score for class labels on crossval dataset: " + str(self.score_(X_cross, Y_cross)))
		print("Score for class labels on test     dataset: " + str(self.score_(X_test , Y_test )))

#	def plot_learning_(self):
#		fig = mpl.figure()
#		ax = fig.add_subplot(111)
#		ax.plot(list(range(self.n_epoch)), self.loss_list, color="red", linewidth=3)
#		if self.learning_rate_type == "invscaling":
#			ay = fig.add_subplot(211)
#			ay.plot(list(range(self.n_epoch)), self.alpha_list, color="green", linewidth=3)
#		mpl.show()



class LogisticRegressionRidge(LinearRegressionRidge, LogisticRegression):
	def __init__(self, theta=[], alpha=0.001, n_cycle=100, n_epoch=10, verbose=False, learning_rate_type='constant', lambda_=0.01):
		super().__init__(theta, alpha, n_cycle, n_epoch, verbose, learning_rate_type, lambda_)

	def loss_elems_(self, X, Y, eps=1e-15):
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or len(self.theta) == 0 or len(X) != len(Y)
			or (len(X[0]) != len(self.theta) and len(X[0]) + 1 != len(self.theta))):
			return None
		if len(X[0]) == len(self.theta):
			return LogisticRegression.loss_elems_(self, X, Y, eps) + self.lambda_ * 0.5 * np.dot(self.theta.T, self.theta) / len(Y)
		if len(X[0]) + 1 == len(self.theta):
			return LogisticRegression.loss_elems_(self, X, Y, eps) + self.lambda_ * 0.5 * np.dot(self.theta[1:].T, self.theta[1:]) / len(Y)

	def loss_(self, X, Y, eps=1e-15):
		return LogisticRegression.loss_(self, X, Y, eps)

	def training_setup_(self, X_train, X_cross, X_test):
		print("\nSetup for training of ridge logistic model...")
		self.set_base_theta_(X_train, X_cross, X_test)

	def train_(self, X_train, Y_train, X_cross, Y_cross, X_test, Y_test, show_progress=False, show_hyperparameter_stats=False):
		LinearRegressionRidge.train_(self, X_train, Y_train, X_cross, Y_cross, X_test, Y_test, show_progress, show_hyperparameter_stats)
		print("Score for class labels on training dataset: " + str(self.score_(X_train, Y_train)))
		print("Score for class labels on crossval dataset: " + str(self.score_(X_cross, Y_cross)))
		print("Score for class labels on test     dataset: " + str(self.score_(X_test , Y_test )))

	def fit_(self, X, Y, show_progress=False):
		LinearRegressionRidge.fit_(self, X, Y, show_progress)


