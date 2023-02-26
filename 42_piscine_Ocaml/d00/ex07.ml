let (==) a b : bool = (a = b) ;;

let is_palindrome (s : string) : bool =
	let result    = ref true         in
	let char_arr  = String.get(s)    in
	let s_len     = String.length(s) in
	let parse_len = s_len / 2        in

	for i = 0 to parse_len - 1 do
		let opp_chars_equal = char_arr(i) == char_arr(s_len - 1 - i) in
		result := !result && opp_chars_equal
	done;

	!result
;;

let test_palindrome (s : string) : unit =
	let test_result = is_palindrome(s)            in
	let test_string = string_of_bool(test_result) in
	print_endline(test_string)
;;

test_palindrome("radar" );
test_palindrome("madam" );
test_palindrome("car"   );
test_palindrome(""      );
