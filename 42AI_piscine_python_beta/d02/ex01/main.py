def what_are_the_vars(*args, **kwargs):
	obj = ObjectC()
	for kwattr in kwargs.items():
		if (kwattr[0][0:4] == "var_"):
			print("ERROR")
			return None
		setattr(obj, kwattr[0], kwattr[1])
	for i in range(len(args)):
		setattr(obj, "var_" + str(i), str(args[i]))
	return obj

class ObjectC(object):
	def __init__(self):
		pass

def doom_printer(obj):
	for attr in dir(obj):
		if attr[0] != '_':
			value = getattr(obj, attr)
			print("{}: {}".format(attr, value))
	print("end")

if __name__ == "__main__":
	obj = what_are_the_vars(7)
	doom_printer(obj)
	obj = what_are_the_vars("ft_lol", "Hi")
	doom_printer(obj)
	obj = what_are_the_vars()
	doom_printer(obj)
	obj = what_are_the_vars(12, "Yes", [0, 0, 0], a=10, hello="world")
	doom_printer(obj)
	obj = what_are_the_vars(42, a=10, var_0="world")
	doom_printer(obj)
	obj = what_are_the_vars(42, a=10, lol=None)
	doom_printer(obj)