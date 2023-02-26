import sys

class Recipe:
	def __init__(self, name, cooking_lvl, cooking_time, ingredients, description, recipe_type):
		if cooking_lvl < 1 or cooking_lvl > 5:
			print("cooking_lvl should be 1 <= cooking_lvl <= 5. Exiting.")
			sys.exit(0)
		if cooking_time < 0:
			print("cooking_time should be >= 0. Exiting.")
			sys.exit(0)
		if len(ingredients) == 0:
			print("ingredients list cannot be empty. Exiting.")
			sys.exit(0)
		if recipe_type != "starter" and recipe_type != "main_course" and recipe_type != "dessert":
			print("recipe_type shoulbd be 'starter', 'main_course' or 'dessert'. Exiting.")
			sys.exit(0)
		self.name = name
		self.cooking_lvl = cooking_lvl
		self.cooking_time = cooking_time
		self.ingredients = ingredients
		self.description = description
		self.recipe_type = recipe_type

	def __str__(self):
		res = "Recipe for " + self.name + ".\n"
		res += "Notes: " + self.description + "\n"
		res += "Ingredients list: " + str(self.ingredients) + "\n"
		res += "To be eaten for " + str(self.recipe_type) + ".\n"
		res += "Takes " + str(self.cooking_time) + " minutes of cooking.\n"
		res += "Difficulty: " + str(self.cooking_lvl) + "/5\n"
		return res






