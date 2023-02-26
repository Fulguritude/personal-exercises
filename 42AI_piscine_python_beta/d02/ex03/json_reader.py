def is_parenthesis(c):
	return c == '{' or c == '}' or c == '(' or c == ')'

def parenthesis_validity(s):
	open_stack = ""
	for c in s:
		if not is_parenthesis(c):
			return False
		if open_stack == "" and (c == ")" or c == "}"):
			return False
		elif c == "(" or c == "{":
			open_stack += c
		elif c == ")":
			if open_stack[-1] == "(":
				open_stack = open_stack[:-1]
			else:
				return False
		elif c == "}":
			if open_stack[-1] == "{":
				open_stack = open_stack[:-1]
			else:
				return False
	return open_stack == ""

def syntax_validity(s):
	simple = "".join([c for c in filetext if (is_parenthesis(c) or c == '"' or c == ":" or c == ",")])
	if '"""' in simple:
		return False
	i = 0
	length = len(simple)
	while i < length:
		if simple[i] == '"':
			if i+2 >= length or simple[i+1] != '"' or (simple[i+2] == '{' or simple[i+2] == '['):
				return False
			else:
				i += 2
				continue
		elif simple[i] == ",":
			if i+1 >= length or simple[i+1] != '"':
				return False
			else:
				i += 1
				continue
		elif simple[i] == ":":
			if i+1 >= length or (simple[i+1] != '"' and simple[i+1] != '{' and simple[i+1] != '['):
				return False
			else:
				i += 1
				continue
		elif simple[i] == "{":
			if i+1 >= length or simple[i+1] != '"':
				return False
			else:
				i += 1
				continue
		elif simple[i] == "["
			if i+1 >= length or (simple[i+1] != '"' and simple[i+1] != ']'):
				return False
			else:
				i += 1
				continue
		#TODO finish including int floats ? 
		else:
			return False




def remove_whitespace(s):
	return "".join([c for c in s if (c != " " and c != "\n" and c != "\t")])


class Loadjson():
	def __init__(self, filename):
		self.data = {}
		try:
			file = open(filename, "r")
		except FileNotFoundError:
			print("JSON file not found.")
			return
		filetext = remove_whitespace(f.read())
		parentheses = "".join([c for c in filetext if is_parenthesis(c)])
		if not parenthesis_validity(parentheses):
			print("Invalid JSON syntax for parentheses.")
			return
		if not syntax_validity(filetext):
			print("Invalid JSON syntax.")
			return
		s_lst = filetext.split('"')
		




def print_formatted(json_dic):
