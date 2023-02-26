d = (3,30,2019,9,25)

def format_date_str(date):
	hour = str(date[0])
	hour = hour.zfill(2)
	minute = str(date[1])
	minute = minute.zfill(2)
	year = str(date[2])
	year = year.zfill(4)
	month = str(date[3])
	month = month.zfill(2)
	day = str(date[4])
	day = day.zfill(2)
	return day + "/" + month + "/" + year + " " + hour + ":" + minute

print(format_date_str(d))