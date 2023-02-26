import sys

rev_argv = sys.argv[:0:-1]

res = ""
for s in rev_argv:
	s = ''.join(c.lower() if c.isupper() else c.upper() for c in s)
	res = res + s[::-1] + ' '

res = res[:-1]

print(res)