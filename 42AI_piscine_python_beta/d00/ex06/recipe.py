cookbook = {
	'sandwich' : {
		'ingredients': ['ham', 'bread', 'cheese'],
		'meal': 'lunch',
		'prep_time': 10		
	},
	'cake' : {
		'ingredients': ['flour', 'sugar', 'eggs'],
		'meal': 'dessert',
		'prep_time': 60
	},
	'salad' : {
		'ingredients': ['avocado', 'arugula', 'tomatoes', 'spinach'],
		'meal': 'lunch',
		'prep_time': 15
	}
}

print("Keys:")
for key in cookbook:
	print(key)
print("\nValues:")
for value in cookbook.values():
	print(value)
print("\nItem pairs:")
for items in cookbook.items():
	print(items)


def print_recipe(key):
	try:
		res = "Recipe for " + key + ":\n"
		res += "Ingredients list: " + str(cookbook[key]['ingredients']) + "\n"
		res += "To be eaten for " + str(cookbook[key]['meal']) + ".\n"
		res += "Takes " + str(cookbook[key]['prep_time']) + " minutes of cooking."
		print(res);
	except KeyError:
		print("Recipe not found in cookbook.")

def delete_recipe(key):
	try:
		del cookbook[key]
		print("Recipe deleted successfully.")
	except KeyError:
		print("Recipe not found in cookbook.")

def add_recipe(key, ingredients, mealtype, prep_time):
	cookbook[key] = {'ingredients' : ingredients, 'meal': mealtype, 'prep_time': prep_time}

def print_cookbook():
	for recipe in cookbook:
		print_recipe(recipe)
		print("")

def cookbook_program():
	s_arg = input("""
		Please select an option by typing the corresponding number:
		1: Add a recipe
		2: Delete a recipe
		3: Print a recipe
		4: Print the cookbook
		5: Quit\n""")
	try:
		i_arg = int(s_arg)
	except ValueError:
		print("Input is not a valid integer.")
		cookbook_program()
		return
	if (i_arg < 1 or 5 < i_arg):
		print(str(i_arg) + " is not a valid option.")
		cookbook_program()
		return
	if (i_arg == 1):
		print("Adding recipe.")
		recipe_name = input('Recipe name ?\n')
		ingredients = []
		wants_other_ing = True
		print("Choosing ingredients.")
		while wants_other_ing:
			in_choice = ''
			while in_choice != 'y' and in_choice != 'n':
				in_choice = input("Add another ingredient ? (y/n)\n")
			if in_choice == 'y':
				ing = input("Next ingredient: ")
				ingredients.append(ing)
				in_choice = ''
			else:
				wants_other_ing = False
		meal_type = input("What type of meal is it ? ")
		prep_time = -1
		while prep_time < 0:
			s_prep_time = input("How many minutes does it take to make ? ")
			try:
				prep_time = int(s_prep_time)
			except ValueError:
				print("Not a valid time in minutes.")
		add_recipe(recipe_name, ingredients, meal_type, prep_time)
		cookbook_program()
	elif (i_arg == 2):
		print("Deleting recipe.")
		recipe_to_delete = input("Recipe to delete: ")
		delete_recipe(recipe_to_delete)
		cookbook_program()
	elif (i_arg == 3):
		print("Printing recipe.")
		recipe_to_print = input("Recipe to print: ")
		print_recipe(recipe_to_print)
		cookbook_program()
	elif (i_arg == 4):
		print_cookbook()
		cookbook_program()
	else:
		print("Quitting program.")

cookbook_program()