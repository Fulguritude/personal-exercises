let repeat_string
	?str: (str: string = "x") (* lol the absence of spacing before the label's colon is mandatory *)
	(n : int)
: string =
	let rec rec_repeat_string (acc : string) (n : int) : string =
		match n with
		| 0            -> acc
		| n when n > 0 -> (rec_repeat_string (acc ^ str) (n - 1))
		| _            -> "Error"
	in
	let result = rec_repeat_string "" n in
	result
;;

let (==) a b : bool = (a = b) ;;

assert (repeat_string (-1)          == "Error"        );
assert (repeat_string 0             == ""             );
assert (repeat_string ~str:"Toto" 1 == "Toto"         );
assert (repeat_string 2             == "xx"           );
assert (repeat_string ~str:"a" 5    == "aaaaa"        );
assert (repeat_string ~str:"what" 3 == "whatwhatwhat" );
