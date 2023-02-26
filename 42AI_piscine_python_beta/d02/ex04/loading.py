from time import time
from time import sleep

start = time()

def ft_progress(lst):
	length = len(lst)
	for i in lst:
		percent = i / length * 100
		s_percent = '%.0f' % percent
		elapsed = time() - start
		s_elapsed = '%.2f' % elapsed
		eta = (100 - percent) * elapsed / (percent + 0.0000001)
		s_eta = '%.2f' % eta
		res = "ETA: " + s_eta + " [" + s_percent.rjust(3, ' ') + "%]["
		bar = ">".rjust(int((percent + 1) / (100 / 24)), '=') #percent + 1 looks nicer
		#24 columns for bar
		bar = bar.ljust(24, ' ')
		res = res + bar + "] " + str(i) + "/" + str(length) + " | elapsed time " + s_elapsed + "s"
		yield res



lst = range(1000)

for elem in ft_progress(lst):
	print(elem, end="\r")
	sleep(0.005)