#https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_logistic.py

import numpy as np
import matplotlib.pyplot as mpl

from loading import get_current_loadbar


class LogisticRegression:
	def __init__(self, alpha=0.001, n_cycle=100, n_epoch=10, verbose=False, learning_rate_type='constant'):
		self.alpha = alpha
		self.n_cycle = n_cycle
		self.n_epoch = n_epoch
		self.verbose = verbose
		self.learning_rate_type = learning_rate_type # can be 'constant' or 'invscaling'
		self.theta = []
		self.loss_list = []
		self.alpha_list = []

	@staticmethod
	def np_matrix_from_any(X):
		if not isinstance(X, np.ndarray):
			X = np.array(X if isinstance(X, list) else [X])
		if not isinstance(X[0], np.ndarray):
			X = np.array(X if isinstance(X, list) else [X])
		return X

	@staticmethod
	def sigmoid_(x):
		"""
		Compute the sigmoid of a scalar or a list.
		Args:
			x: a scalar or list
		Returns:
			The sigmoid value as a scalar or list.
			None on any error.
		Raises:
			This function should not raise any Exception.
		"""
		if isinstance(x, list):
			return list(map(lambda t: 1 / (1 + np.exp(-t)), x))
		return 1 / (1 + np.exp(-x))

	def set_base_theta_(self, x):
		self.theta = np.ones((x.shape[1] + 1, 1))

	def predict_prob_(self, X):
		"""
		Predict class labels for samples in a dataset X.
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
		if len(X) == 0 or len(X[0]) == 0 or len(X[0]) != len(self.theta) - 1:
			print("LogisticRegression.predict_: " + str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(X[0]) != len(self.theta) - 1))
			return None
		y_pred = LogisticRegression.sigmoid_(np.dot(X, self.theta[1:]) + self.theta[0])
		y_pred = y_pred.reshape((-1, 1))
		return y_pred

	def predict_class_(self, X):
		return ((self.predict_prob_(X) > 0.5) * 1)

	def loss_(self, x, y_true, m, eps=1e-15):
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
		y_true_T = LogisticRegression.np_matrix_from_any(y_true).T
		y_pred = self.predict_prob_(x)
		if len(y_pred) != m or len(y_true) != m:
			print(str(len(y_pred[0]) != m) + " " + str(len(y_true[0]) != m))
			return None
		y_pred = [(eps if y_pred_i[0] <= 0 else (1 - eps if y_pred_i[0] >= 1 else y_pred_i[0])) for y_pred_i in y_pred]
		y_pred = np.array(y_pred)#.reshape((-1, 1))
		#print(y_pred)
		log_ypred = np.log(y_pred)
		log_1minus_ypred = np.log(1 - y_pred)
		return (-(np.dot(y_true_T, log_ypred) + np.dot((1 - y_true_T), log_1minus_ypred)) / m)[0]

	def gradient_(self, x, y_true):
		"""
		Computes the gradient.
		Args:
			x: a 1d or 2d numpy ndarray for the samples
			y_true: a scalar or a numpy ndarray for the correct labels
			y_pred: a scalar or a numpy ndarray for the predicted labels
		Returns:
			The gradient as a scalar or a numpy ndarray of the width of x.
			None on any error.
		Raises:
			This function should not raise any Exception.
		"""
		#x = LogisticRegression.np_matrix_from_any(x)
		#y_true = LogisticRegression.np_matrix_from_any(y_true)
		y_pred = self.predict_prob_(x)
		#print("y_true: " + str(len(y_true)) + " " + str(y_true))
		#print("y_pred: " + str(len(y_pred)) + " " + str(y_pred))		
		if len(y_pred) != len(y_true) or len(x) != len(y_true):
			print("LogisticRegression.gradient_: " + str(len(y_pred) != len(y_true)) + " " + str(len(x) != len(y_true)))
			return None
		loss_vec = (y_pred - y_true)
		gradient = np.dot(loss_vec.T, x).T
		result = np.ones((gradient.shape[0] + 1, 1))
		result[0, 0] = sum(loss_vec)[0]
		result[1:] = gradient
		return result / len(loss_vec)

	def fit_(self, X, Y, show_progress=False):
		"""
		Fit the model according to the given training data. Args:
		X: a 1d or 2d numpy ndarray for the samples
		Y: a scalar or a numpy ndarray for the correct labels Returns:
		self : object
		None on any error. Raises:
		This method should not raise any Exception.
		"""
		if (len(X) == 0 or len(X[0]) == 0 or len(Y) == 0 or
			len(X) != len(Y) or len(Y[0]) != 1): #or len(X[0]) != len(self.theta) - 1
			print("LogisticRegression.fit_: " + str(len(X) == 0) + " " + str(len(X[0]) == 0) + " " + str(len(Y) == 0) + " " + str(len(X) != len(Y)) + " " + str(len(Y[0]) != 1))
			return None
		for cycle in range(self.n_cycle):
			if show_progress:
				print(get_current_loadbar(cycle + 1, self.n_cycle), end=("\n" if cycle + 1 == self.n_cycle else "\r"))
			gradient = self.gradient_(X, Y)
			self.theta = self.theta - self.alpha * gradient
			#print(self.theta.T)
		return self.theta

	def score_(self, x, y_true):
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
		y_pred = self.predict_class_(x)		
		if len(y_pred) != len(y_true):
			return None
		#print("y_pred.shape: " + str(y_pred.shape) + "\ny_true.shape: " + str(y_pred.shape))
		return (y_pred == y_true).mean()

	def train_(self, X, Y, X_test, Y_test, show_progress=False, show_hyperparameter_stats=False):
		#TODO dimension checks
		self.set_base_theta_(X)
		step_list = np.arange(0, self.n_epoch, int(self.n_epoch / 8))
		X = LogisticRegression.np_matrix_from_any(X)
		Y = LogisticRegression.np_matrix_from_any(Y).T
		#print("Y: " + str(Y))
		X_test = LogisticRegression.np_matrix_from_any(X_test)
		Y_test = LogisticRegression.np_matrix_from_any(Y_test).T
		for epoch in range(self.n_epoch):
			new_loss = self.loss_(X, Y, len(Y))
			self.loss_list.append(new_loss)
			self.fit_(X, Y, show_progress)
			#print("theta: " + str(self.theta))
			if self.learning_rate_type == 'invscaling':
				new_alpha = 1 / new_loss
				self.alpha = new_alpha
				self.alpha_list.append(new_alpha)
			if self.verbose and len(step_list) > 0 and epoch == int(step_list[0]):
				print("epoch " + str(epoch) + ": \tloss = " + str(self.loss_list[-1]))
				step_list =  step_list[1:]
		print("Score on training dataset:\t" + str(self.score_(X, Y)))
		print("Score on test	 dataset:\t" + str(self.score_(X_test, Y_test)))
		if show_hyperparameter_stats:
			self.plot_learning_()

	def plot_learning_(self):
		fig = mpl.figure()
		ax = fig.add_subplot(111)
		ax.plot(list(range(self.n_epoch)), self.loss_list, color="red", linewidth=3)
		if self.learning_rate_type == "invscaling":
			ay = fig.add_subplot(211)
			ay.plot(list(range(self.n_epoch)), self.alpha_list, color="green", linewidth=3)
		mpl.show()



