t = (19, 42, 21)

def format_tuple_str(tuple):
	length = len(tuple)
	res = "The " + str(length) + " numbers are: "
	for i in range(length):
		res += str(t[i])
		if (i < length - 1):
			res += ", "
	return res

print(format_tuple_str(t))