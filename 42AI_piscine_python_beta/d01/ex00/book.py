from datetime import datetime as dt
import recipe
import sys

class Book:
	def __init__(self, name):
		self.name = name
		self.last_update = dt.today()
		self.creation_date = dt.today()
		self.recipes_list = {'starter': [], 'main_course': [], 'dessert': []}

	def __str__(self):
		res = "Book: " + self.name
		res += "Created: " + str(self.creation_date)
		res += "Last edited: " + str(self.last_update)
		res += "Recipes: "
		for r_lst in self.recipes_list.values():
			for recipe in r_lst:
				res += str(recipe) + "\n"
		return res


	def get_recipe_by_name(self, name):
		"""Return a recipe with the name `name` """
		for r_lst in self.recipes_list.values():
			for recipe in r_lst:
				if recipe.name == name:
					print(str(recipe))
					return recipe

	def get_recipes_by_types(self, recipe_type):
		"""Get all recipe names for a given type """
		try:
			res = self.recipes_list[recipe_type]
		except KeyError:
			print("Invalid recipe type")
			sys.exit(0)
		return res


	def add_recipe(self, recipe):
		"""Add a recipe to the book """
		try:
			self.recipes_list[recipe.recipe_type].append(recipe)
		except KeyError:
			print("Invalid recipe type.")
			sys.exit(0)
		self.last_update = dt.today()

