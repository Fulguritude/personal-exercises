let rec rec_countdown (a : int) : unit =
	if a <= 0 then
	(
		print_int(0);
		print_char('\n')
	)
	else
	(
		print_int(a);
		print_char('\n');
		rec_countdown(a - 1)
	)
;;

let countdown (a : int) : unit =
	for i = a to 0 do
		print_int(i);
		print_char('\n')
	done
;;

rec_countdown(3);
rec_countdown(0);
rec_countdown(-1);

print_char('\n');

rec_countdown(3);
rec_countdown(0);
rec_countdown(-1);



