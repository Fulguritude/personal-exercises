#https://stackoverflow.com/questions/14017996/is-there-a-way-to-pass-optional-parameters-to-a-function

from typing import Literal, List
from random import randint

Options = Literal["shuffle", "ordered", "unique"]

def randomize(
	lst : List[str],
) -> List[str]:
	res : List[str] = []
	length = len(lst)
	for i in range(length - 1):
		index = randint(0, length - 1 - i)
		elem = lst.pop(index)
		res.append(elem)
	return res

def generator(
    text   : str,
    sep    : str     = " ",
    option : Options = "ordered",
):
	s_lst = text.split(sep)
	if   option == "shuffle":
		s_lst = randomize(s_lst)
	elif option == "unique":
		s_lst = list(set(s_lst))
	elif option == "ordered":
		s_lst.sort()
	else:
		print("Incorrect option for generator().")
	for i in s_lst:
		yield i

text = "Le Lorem Ipsum est simplement du faux (*très* faux) texte."
for word in generator(text, sep = " ", option = "ordered"):
	print(word)
