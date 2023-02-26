#https://www.geeksforgeeks.org/decorators-in-python/ #third example
#https://www.guru99.com/reading-and-writing-files-in-python.html

import time
from random import randint

def log(func):
	def wrapper(*args, **kwargs):
		file = open("machine.log", "a+") #+ means if the file isn't there create it, a for append
		capital_func_name = " ".join([w.capitalize() for w in func.__name__.split("_")])
		logtxt = "Running: " + capital_func_name + "\t[ exec-time = "
		start = time.time()
		res = func(*args, **kwargs)
		end = time.time()
		exectime = end - start
		time_str = (('%.3f' % exectime) + " s  ]\n") if (exectime > 0.001) else (('%.3f' % (exectime * 1000)) + " ms ]\n")
		logtxt += time_str
		file.write(logtxt)
		file.close()
		return res
	return wrapper