let print_alphabet () : unit =
	for i = int_of_char('a') to int_of_char('z') do
		let c = char_of_int(i) in
		print_char(c)
	done;
	print_char('\n')
;;

print_alphabet();
