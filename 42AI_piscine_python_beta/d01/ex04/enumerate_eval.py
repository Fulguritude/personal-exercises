#https://stackoverflow.com/questions/13252333/python-check-if-all-elements-of-a-list-are-the-same-type
#https://docs.python.org/3.3/library/functions.html#enumerate

def evaluate(coefs, words):
	len_coefs = len(coefs)
	len_words = len(words)
	if len_coefs != len_words:
		return -1
	if (not all(isinstance(x, (int, float)) for x in coefs) or
			not all(isinstance(x, (str)) for x in words)):
		print("Illegal argument to evaluate")
		raise ValueError
	pair_lst = [(coefs[i], words[i]) for i in range(len_words)]
	tmp_lst = [coef * len(word) for (coef, word) in pair_lst]
	return sum(tmp_lst)
