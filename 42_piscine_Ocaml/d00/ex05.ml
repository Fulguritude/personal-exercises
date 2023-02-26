let print_rev (s: string) : unit =
	let char_arr = String.get    (s) in
	let s_len    = String.length (s) in
	for i = 0 to s_len - 1 do
		let c_i = s_len - 1 - i in
		let c   = char_arr(c_i) in
		print_char(c);
	done;
	print_char('\n')
;;


print_rev("Hello World!")
