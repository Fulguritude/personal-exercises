languages = {
       'Python': 'Guido van Rossum',
       'Ruby': 'Yukihiro Matsumoto',
       'PHP': 'Rasmus Lerdorf',
       }

def format_dict_str(dic):
	res = ""
	for key in dic:
		res += str(key) + " was created by " + str(dic[key]) + "\n"
	return res[:-1]

print(format_dict_str(languages))