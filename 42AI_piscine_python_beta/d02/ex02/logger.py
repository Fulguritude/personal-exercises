#https://www.geeksforgeeks.org/decorators-in-python/ #third example
#https://www.guru99.com/reading-and-writing-files-in-python.html

import time
from random import randint

class CoffeeMachine():
	water_level = 100

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


	@log
	def start_machine(self):
		if self.water_level > 20:
			return True
		else:
			print("Please add water!")
			return False

	@log
	def boil_water(self):
		return "boiling..."

	@log
	def make_coffee(self):
		if self.start_machine():
			for _ in range(20):
				time.sleep(0.1)
				self.water_level -= 1
			print(self.boil_water())
			print("Coffee is ready!")

	@log
	def add_water(self, water_level):
		time.sleep(randint(1, 5))
		self.water_level += water_level
		print("Glouglouglou...")


if __name__ == "__main__":
	machine = CoffeeMachine()
	for i in range(0, 5):
		machine.make_coffee()
	machine.make_coffee()
	machine.add_water(70)