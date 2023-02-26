
def text_analyzer(s_arg=None):
	"""This function counts the number of upper characters, lower characters,
    punctuation and spaces in a given text."""
	if (s_arg is None):
		s = input("What is the text to analyse?\n")
	else:
		s = str(s_arg)
	total = 0
	upper = 0
	lower = 0
	punct = 0
	space = 0
	for c in s:
		total += 1
		if c.isupper():
			upper += 1
		elif c.islower():
			lower += 1
		elif c.isspace():
			space += 1
		else:
			punct += 1
	print("The text contains " + str(total) + " characters:\n" +
			"- " + str(upper) + " upper letters\n" +
			"- " + str(lower) + " lower letters\n" +
			"- " + str(punct) + " punctuation marks\n" +
			"- " + str(space) + " spaces")
