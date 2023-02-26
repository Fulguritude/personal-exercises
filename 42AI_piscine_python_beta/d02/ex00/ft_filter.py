#filter_func is a predicate, ie takes any argument of type A' but must return a boolean; lst contains only elements of type A'

def ft_filter(filter_func, lst):
	return [a for a in lst if filter_func(a)]