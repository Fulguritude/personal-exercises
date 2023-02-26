from time import time
from time import sleep

start = time()

def ft_progress(lst, load_bar_size = 24):
	length = len(lst)
	for i in lst:
		percent = i / length * 100
		str_percent = "%.0f" % percent
		elapsed = time() - start
		str_elapsed = "%.2f" % elapsed
		eta = (100 - percent) * elapsed / (percent + 0.0000001)  # avoid div by 0
		str_eta = '%.2f' % eta
		load_bar_pieces_amount = int((percent + 1) / (100 / 24))  # percent + 1 looks nicer
		load_bar = ">".rjust(load_bar_pieces_amount, '=')
		load_bar = load_bar.ljust(load_bar_size, ' ')
		res = (
			f"ETA: {str_eta} "
			f"[{str_percent.rjust(3, ' ')}%]"
			f"[{load_bar}] "
			f"[{i}/{length} "
			f"| elapsed time {str_elapsed}s"
		)
		yield res



lst = range(1000)

for elem in ft_progress(lst):
	print(elem, end="\r")
	sleep(0.005)