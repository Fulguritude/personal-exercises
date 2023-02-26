let test_sign (a : int) : unit =
	let sign = if a >= 0 then "positive" else "negative" in
	print_endline(sign)
;;

test_sign(42);
test_sign(0);
test_sign(-42);
