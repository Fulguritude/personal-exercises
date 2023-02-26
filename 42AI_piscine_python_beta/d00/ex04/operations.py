
import sys

if (len(sys.argv) == 3):
	valid_args = True
	try:
		x = int(sys.argv[1])
		y = int(sys.argv[2])
	except ValueError:
		valid_args = False
		print("Usage: python operations.py <number1> <number2>\nExample: python operations.py 10 3")
	if valid_args:
		sum_xy = x + y
		diff_xy = x - y
		prod_xy = x * y
		quot_xy = float('nan') if y == 0 else x / y
		rem_xy = float('nan') if y == 0 else x % y
		print(
				"Sum:         " + str(sum_xy) + "\n" +
				"Difference:  " + str(diff_xy) + "\n" +
				"Product:     " + str(prod_xy) + "\n" +
				"Quotient:    " + str(quot_xy) + "\n" +
				"Remainder:   " + str(rem_xy))
else:
	print("Usage: python operations.py <number1> <number2>\nExample: python operations.py 10 3")