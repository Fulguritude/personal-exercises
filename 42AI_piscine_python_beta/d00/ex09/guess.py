from random import randint

solution = randint(1, 100)

print("""This is an interactive guessing game!
You have to enter a number between 1 and 99 to find out the secret number.
Type 'exit' to end the game.
Good luck!""")

guess = -1
play_loop = True
try_nb = 0

while guess != solution and play_loop:
	try_nb += 1
	guess = input("What's your guess between 1 and 99?\n")
	if guess == 'exit':
		print("Goodbye.")
		play_loop = False
	try:
		guess = int(guess)
	except ValueError:
		print("That's not a number.")
		continue
	if guess > solution:
		print("Too high!")
	elif guess < solution:
		print("Too low!")
	else:
		if try_nb == 1:
			print("Congratulations, you got it on the first try!")
		else:
			print("Congratulations, you've got it!\nYou won in " + str(try_nb) + " attempts!")
		play_loop = False
