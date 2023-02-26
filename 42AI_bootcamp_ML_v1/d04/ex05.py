import pandas as pd
import numpy as np
import graphviz as gv
from math import log2 as lg

from node import Node



#Information Entropy can be thought of as how how unpredictable a dataset is.
def entropy(array):
	"""
	Computes the Shannon Entropy of a non-empty numpy.ndarray
	Args:
		- array: numpy.ndarray
	Returns:
		float: shannon's entropy as a float
		None if input is not a non-empty numpy.ndarray
	"""
	if ((not isinstance(array, list) and not isinstance(array, np.ndarray)) or
		len(array) == 0):
		return None
	classes = list(set(array))
	class_amount = len(classes)
	acc = 0
	inv_len = 1 / len(array)
	for label in classes:
		prob = sum([elem == label for elem in array]) * inv_len
		acc = acc - prob * lg(prob)
	return acc


#Gini Impurity is the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the dataset.
#Gini impurity is a metric that evaluates the quality of a split in the dataset.
#G(X) = sum_{c € Classes} (p(c) * (1 - p(c)))
def gini(array):
	"""
	Computes the gini impurity of a non-empty numpy.ndarray
	Args:
		array: numpy.ndarray
	Return
		float: gini_impurity as a float or None if input is not a non-empty numpy.ndarray
	"""
	if ((not isinstance(array, list) and not isinstance(array, np.ndarray)) or
		len(array) == 0):
		return None
	inv_len = 1 / len(array)
	#array = inv_len * array
	classes = list(set(array))
	class_amount = len(classes)
	acc = 0
	inv_len = 1 / len(array)
	for label in classes:
		prob = sum([elem == label for elem in array]) * inv_len
		acc = acc + prob * (1 - prob)
	return acc


#https://victorzhou.com/blog/information-gain/
#Information Gain = how much Entropy we removed
#Information Gain is a metric that evaluates the quality of a split in the dataset.
#It is calculated for a split by subtracting the weighted entropies of each branch from the original entropy.
#In the case of Gini below, information gain is referred to as "Gini gain"
def information_gain(array_source, array_splits, criterion='gini'):
	"""
    Computes the information gain between the first and second array using the criterion ('gini' or 'entropy'). Also called Kullback Leibler Divergence
    Args:
		numpy.ndarray array_source: an array for data
		list array_children_list: list of numpy.ndarray, representing a split in the dataset
		str criterion: Should be in ['gini', 'entropy']
	Return:
		float: Shannon entropy as a float
		None if input is not a non-empty numpy.ndarray
		None if invalid input
	"""
	if ((not isinstance(array_source, list) and not isinstance(array_source, np.ndarray)) or
		len(array_source) == 0 or
		(not isinstance(array_splits, list) and not isinstance(array_splits, np.ndarray)) or
		len(array_splits) == 0 or
		len(np.ndarray.flatten(array_splits)) != len(array)
		):
		return None
	inv_len = 1. / len(array)
	metric = gini if criterion == "gini" else entropy
	start_information = metric(array_source)
	end_information = 0.
	for split in array_splits:
		end_information = end_information + inv_len * len(split) * metric(split)
	return start_information - end_information



class DecisionTreeClassifier:
	def __init__(self, criterion='gini', max_depth=10, min_elem_per_leaf=15, log_split_depth=10):
		"""
		:param str criterion: 'gini' or 'entropy'
		:param max_depth: max_depth of the tree (Decision tree creation stops splitting a node if node.depth >= max_depth)
		"""
		self.root = None # Root node of the tree
		self.criterion = criterion
		self.max_depth = max_depth
		self.min_elem_per_leaf = abs(min_elem_per_leaf)
		self.labels = []
		self.log_split_depth = abs(log_split_depth)

	def split_evaluate_data(self, X_feature, Y_label, split_val):
		inf_indices = []
		sup_indices = []
		for i in range(len(X_feature)):
			value = X_feature[i]
			if value <= split_val:
				inf_indices = inf_indices + [i]
			else:
				sup_indices = sup_indices + [i]
		info_gain = information_gain(Y_label, [Y_label[inf_indices], Y_label[sup_indices]], self.criterion)
		return info_gain

	def build_tree():
		def rec_build_tree():
			pass
		pass

	def prune_tree():
		pass

	def set_labels(self, Y);
		self.labels = list(set(Y))

	def add_node_to_tree(self, parent_node, Y_data, split_n_measure):
		if parent_node == None:
			node = Node(data=Y_data, labels=get_labels_from_data(Y_data),
					is_leaf=False, split_feature=split_n_measures[0], split_kind="<=", split_value=split_n_measure[1], split_gain=split_n_measure[2],
					left=None, right=None,
					depth=0)
			self.root = node
		else:
			parent_node.add_child_node()


	def fit(self, X, Y):
		"""
		Build the decision tree from the training set (X, y). The training set has m data_points (examples).
		Each of them has n features.
		Args:
			:param pandas.Dataframe X: Training input (m x n)
			:param pandas.Dataframe y: Labels (m x 1)
		Return: object self: Trained tree
		"""
		self.set_labels(Y)
		X_T = X.T
		splits_n_measures = []
		for i in range(len(X_T)):
			feature = X_T[i]
			mini = min(feature)
			maxi = max(feature)
			split_step = (maxi - mini) * (0.5 ** self.log_split_depth)
			splits = np.arange(mini, maxi, split_step)
			for j in splits:
				splits_n_measures = splits_n_measures + [(i, j, split_evaluate_data(feature, Y, splits[j]))]
			best_split = splits_n_measures[0]
		for i in range(len(splits_n_measures)):
			if best_split[2] < splits_n_measures[i][2]:
				best_split = splits_n_measures[i]
		self.add_node_to_tree(best_split)


	
	def squared_error(self, X, Y):
		return sum((self.predict(X) - Y) ** 2)



if __name__ == '__main__':
	from sklearn.model_selection import train_test_split
	from sklearn.datasets import load_iris
	# sklearn is not allowed in the classes.
	# Test on iris dataset
	iris = load_iris()
	X = pd.DataFrame(iris.data)
	y = pd.DataFrame(iris.target)
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
	dec_tree = DecisionTreeClassifier()
	dec_tree.fit(X_train, y_train)
	root = dec_tree.root
	print("TEST ON IRIS DATASET")
	print("Root split info = 'Feature_{}{}{}'\n".format(root.split_feature, root.split_kind, root.split_criteria))
	print("5 first lines of the labels of the left child of root=\n{}\n".format(root.left_child.y.head()))
	print("5 first lines of the labels of the right child of root=\n{}".format(root.right_child.y.head()))
	"""
	dot_data = tree.export_graphviz(classification_tree, out_file=None, 
	                     feature_names=iris.feature_names,  
	                     class_names=iris.target_names,  
	                     filled=True, rounded=True,  
	                     special_characters=True)  
	graph = graphviz.Source(dot_data)  
	graph.render("iris")
	"""


"""
TEST ON IRIS DATASET
Root split info = 'Feature_2 <= 1.9'

5 first lines of the labels of the left child of root =
   0
18 0
4  0
45 0
39 0
36 0

5 first lines of the labels of the right child of root =
    0
118 2
59  1
117 2
139 2
107 2
"""