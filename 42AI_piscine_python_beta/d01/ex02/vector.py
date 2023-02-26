#https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
#https://stackoverflow.com/questions/13252333/python-check-if-all-elements-of-a-list-are-the-same-type
#https://stackoverflow.com/questions/44521074/operator-overloading-in-python-handling-different-types-and-order-of-parameters
#https://stackoverflow.com/questions/19684434/best-way-to-check-function-arguments-in-python #beartyping

import sys

class Vector:
	def __init__(self, *args):
		if len(args) == 1:
			if isinstance(args[0], list):
				try:
					self.length = len(args[0])
					self.values = [float(i) for i in args[0]]
				except ValueError:
					print("Element of Vector value list not a valid float")
					sys.exit(0)
			else:
				self.values = range(args[0])
				self.length = len(self.values)
		elif len(args) == 2:
			self.values = range(args[0], args[1])
			self.length = len(self.values)
		elif len(args) == 3:
			self.values = range(args[0], args[1], args[2])
			self.length = len(self.values)
		else:
			print("Invalid arguments to Vector constructor.")
			raise TypeError

	def __str__(self):
		return "(Vector " + str(self.values) + ")"

	def __repr__(self,other):
		return "(Vector:\n\tLength: " + str(self.length) + "\n\tValues: " + str(self.values) + "\n)"

	def __add__(self,other):
		if (isinstance(other, Vector)):
			if other.length != self.length:
				print("Invalid vector __add__")
				raise ValueError
			pair_lst = [(self.values[i], other.values[i]) for i in range(self.length)]
			return Vector([a + b for (a, b) in pair_lst])
		#assert ?
		res = Vector([a + other for a in self.values])
		return res

	def __sub__(self,other):
		if (isinstance(other, Vector)):
			if other.length != self.length:
				print("Invalid vector __sub__")
				raise ValueError
			pair_lst = [(self.values[i], other.values[i]) for i in range(self.length)]
			return Vector([a - b for (a, b) in pair_lst])
		#assert ?
		res = Vector([a - other for a in self.values])
		return res

	def __mul__(self,other):
		if (isinstance(other, Vector)):
			if other.length != self.length:
				print("Invalid vector __mul__")
				raise ValueError
			pair_lst = [(self.values[i], other.values[i]) for i in range(self.length)]
			tmp = [a * b for (a, b) in pair_lst]
			return sum(tmp) #dot product
		#assert ?
		return Vector([a * other for a in self.values]) #scalar mult	

	def __div__(self,other):
		if (isinstance(other, Vector)):
			if other.length != self.length:
				print("Invalid vector __div__")
				raise ValueError
			pair_lst = [(self.values[i], other.values[i]) for i in range(self.length)]
			return Vector([a / b for (a, b) in pair_lst])
		#assert ?
		res = Vector([a / other for a in self.values])
		return res

#radd rsub rmul rtruediv

