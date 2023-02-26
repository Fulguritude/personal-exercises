let print_comb () : unit =
	for i = 0     to 7 do
	for j = i + 1 to 8 do
	for k = j + 1 to 9 do
		print_int(i);
		print_int(j);
		print_int(k);
		if i != 7 || j != 8 || k != 9 then print_string(", ") else ()
	done
	done
	done;
	print_char('\n')
;;

print_comb();
