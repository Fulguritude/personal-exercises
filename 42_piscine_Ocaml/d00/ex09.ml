let print_comb2 () : unit =
	for i = 0     to 98 do
	for j = i + 1 to 99 do
		if i < 10 then print_int(0) else ();
		print_int(i);
		print_char(' ');
		if j < 10 then print_int(0) else ();
		print_int(j);
		if i != 98 || j != 99 then print_string(", ") else ()
	done
	done;
	print_char('\n')
;;

print_comb2();
