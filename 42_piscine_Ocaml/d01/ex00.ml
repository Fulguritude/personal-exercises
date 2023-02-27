let repeat_x (n : int) : string =
	let rec rec_repeat_x (acc : string) (n : int) : string =
		match n with
		| 0            -> acc
		| n when n > 0 -> (rec_repeat_x (acc ^ "x") (n - 1))
		| _            -> "Error"
	in
	let result = rec_repeat_x "" n in
	result
;;

let (==) a b : bool = (a = b) ;;

assert (repeat_x (-1) == "Error" ) ;;
assert (repeat_x ( 0) == ""      ) ;;
assert (repeat_x ( 1) == "x"     ) ;;
assert (repeat_x ( 2) == "xx"    ) ;;
assert (repeat_x ( 5) == "xxxxx" ) ;;
