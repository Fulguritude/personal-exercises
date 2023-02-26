class Node:
	def __init__(self,
					data=None, labels=None,
					is_leaf=False, split_feature=None, split_kind=None, split_value=None, split_gain=None,
					left=None, right=None,
					depth=-1
				):
		"""
		:param pandas.Dataframe data: features
		:param pandas.Dataframe labels: labels

		:param bool is_leaf: True if the node is a leaf of the tree
		:param int split_feature: column of the feature
		:param str split_kind: ["<=" or "="]; ie, quantitative vs qualitative
		:param split_value: value of the criteria used to split data
		:param split_gain: information gain when using this split

		:param Node left: node child where criteria is True
		:param Node right: node child where criteria is False

		:param int depth: depth level of the node in the tree
		"""
		# data
		self.X = data
		self.y = labels

		# split_info
		self.is_leaf = is_leaf
		self.split_feature = split_feature
		self.split_kind = split_kind
		self.split_value = split_value
		self.split_gain = split_gain
		if self.is_leaf:
			self.content = "Leaf"
		else:
			self.content = "Feature {} {} {}".format(self.split_feature, self.split_kind, self.split_criteria)

		# children
		self.left_child = left
		self.right_child = right

		# meta
		self.depth = depth

	def __str__(self):
		output_print = """{}\nNode depth = {}\n\n""".format(self.content, self.depth)
		if self.is_leaf:
			output_print += """X =\n{}\n\ny = \n{}\n""".format(self.X, self.y)
		return output_print

	def add_child_node(self,
						split_data, labels,
						split_feature=None, split_kind=None, split_value=None, split_gain=None,
						max_depth, is_left
						):
		if self.depth >= max_depth:
			print("add_child_node: Error: cannot extend tree beyond max_depth")
			return
		if self.is_leaf:
			print("add_child_node: Error: cannot extend leaf")
			return
		new_depth = self.depth + 1
		is_leaf = new_depth == max_depth
		if not is_leaf and (split_feature == None or split_kind == None or split_value == None):
		#	print("add_child_node: Error: missing features in non-leaf node")
		#	return
			print("add_child_node: Warning: missing features in non-leaf node")
		elif is_leaf and (split_feature != None or split_kind != None or split_value != None):
			print("add_child_node: Warning: ignoring feature for leaf node")
		node = Node(split_data, labels,
					is_leaf, split_feature, split_kind, split_value,
					left=None, right=None,
					depth=new_depth)
		if is_left:
			self.left_child = node
		else:
			self.right_child = node




