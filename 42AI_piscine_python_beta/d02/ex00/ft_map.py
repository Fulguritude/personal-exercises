#map is a function from A' to B', lst contains only elements of type A'

def ft_map(map_func, lst):
	return [map_func(a) for a in lst]