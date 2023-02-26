let is_digit c : bool = '0' <= c && c <= '9';;
let is_lower c : bool = 'a' <= c && c <= 'z';;
let is_upper c : bool = 'A' <= c && c <= 'Z';;

let string_all
	(condition : (char -> bool) )
	(s         : string         )
: bool =
	let result   = ref true      in
	let char_arr = String.get(s) in

	for i = 0 to String.length(s) - 1 do
		let c = char_arr(i) in 
		result := !result && condition(c)
	done;

	!result
;;


let result1 = string_all is_digit "01234567890" ;;
let result2 = string_all is_digit "a0123456789" ;;
let result3 = string_all is_digit "0123456789a" ;;
let result4 = string_all is_lower "abcdefghijk" ;;
let result5 = string_all is_lower "0abcdefghij" ;;
let result6 = string_all is_lower "abcdefghijA" ;;
let result7 = string_all is_upper "ABCDEFGHIJK" ;;
let result8 = string_all is_upper "0ABCDEFGHIJ" ;;
let result9 = string_all is_upper "ABCDEFGHIJa" ;;

print_endline(string_of_bool(result1));
print_endline(string_of_bool(result2));
print_endline(string_of_bool(result3));
print_endline(string_of_bool(result4));
print_endline(string_of_bool(result5));
print_endline(string_of_bool(result6));
print_endline(string_of_bool(result7));
print_endline(string_of_bool(result8));
print_endline(string_of_bool(result9));
