#https://stackoverflow.com/questions/1369526/what-is-the-python-keyword-with-used-for



from json_reader import loadjson, print_formatted

if __name__ == "__main__":
	with loadjson('list.json') as js:
		data = js.getdata()
		print_formatted(data)