tup = (0, 4, 132.42222, 10000, 12345.67)

def format_exptuple_str(t):
	length = len(t)
	res = "day_" + str(t[0]).zfill(2) + ", ex_" + str(t[1]).zfill(2) + " : "
	for i in range(2, length):
		res += '%.2e' % t[i]
		if (i < length - 1):
			res += ", "
	return res

print(format_exptuple_str(tup))