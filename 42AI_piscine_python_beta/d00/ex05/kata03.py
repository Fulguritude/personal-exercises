phrase = "The right format"

def format_padded_str(s):
	return s.rjust(42, '-')

print(format_padded_str(phrase))