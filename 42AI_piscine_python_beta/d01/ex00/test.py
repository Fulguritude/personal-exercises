from book import *
from recipe import *

book1 = Book("LomoYNaranja")

recipe1 = Recipe("Lomo", 1, 10, ["Jambon"], "", "starter")
recipe2 = Recipe("Naranja", 1, 0, ["Orange"], "A normal orange", "dessert")
recipe3 = Recipe("LomoDeNaranja", 3, 30, ["Jambon", "Orange"], "Don't try it at home.", "main_course")

book1.add_recipe(recipe1)
book1.add_recipe(recipe2)
book1.add_recipe(recipe3)

print(str(book1))


print("\nLooking for an orange:")
print(str(book1.get_recipe_by_name('Naranja')))


print(dir(book1))
