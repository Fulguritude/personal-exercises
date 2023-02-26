#reduce takes a binary operator in A' and applies it over the n elements of lst of type A', ie, reduce is the n-ary version of the operator
#if lst is Null, reduce should return the neutral element of the type A'

def ft_reduce(op_func, lst):
	res = lst[0]
	for i in range(1, len(lst)):
		res = op_func(res, lst[i])
	return res