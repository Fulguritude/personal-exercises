import sys

def odd_even_zero():
	if (len(sys.argv) != 2):
		print("ERROR")
		return
	try:
		int_arg = int(sys.argv[1])
	except ValueError:
		print("ERROR")
		return
	if (int_arg == 0):
		print("I'm Zero.")
	elif (int_arg % 2 == 0):
		print("I'm Even.")
	else:
		print("I'm Odd.")
	
odd_even_zero()
