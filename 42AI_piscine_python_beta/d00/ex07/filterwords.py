import sys

def length(lst):
	count_lst = [1 for i in lst]
	return sum(count_lst)

def filter_words(s, letter_min):
	s_lst = s.split()
	s_lst = [(s[:-1] if not s[-1].isalpha() else s) for s in s_lst]
	new_lst = [s for s in s_lst if length(s) >= letter_min]
	print(new_lst)


if length(sys.argv) != 3:
	print("ERROR")
else:
	try:
		s = sys.argv[1]
		letter_min = int(sys.argv[2])
	except ValueError:
		print("ERROR")
	if (isinstance(s, str)):
		filter_words(s, letter_min)
	else:
		print("ERROR")