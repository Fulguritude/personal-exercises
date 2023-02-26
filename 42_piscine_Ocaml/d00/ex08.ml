type case = Upper | Lower ;;

let is_upper (c : char) : bool = 'A' <= c && c <= 'Z';;
let is_lower (c : char) : bool = 'a' <= c && c <= 'z';;

let rot_n (rot : int) (s : string) : string =
	let rotate_val (base : case) (rot : int) (c : char) : char =
		let val_base = match base with
			| Upper -> int_of_char('A')
			| Lower -> int_of_char('a')
		in
		let val_old  = int_of_char(c)           - val_base in
		let val_new  = ((val_old + rot) mod 26) + val_base in
		char_of_int(val_new)
	in

	let rotate_char (rot : int) (c : char) : char =
		let result =
			match c with
			| c_upp when is_upper(c) -> rotate_val Upper rot c_upp
			| c_low when is_lower(c) -> rotate_val Lower rot c_low
			| _ -> c
		in
		result
	in

	let rotate = rotate_char rot     in
	let result = String.map rotate s in
	result
;;



let result1 = rot_n  1 "abcdefghijklmnopqrstuvwxyz";; let expect1 = "bcdefghijklmnopqrstuvwxyza";;
let result2 = rot_n 13 "abcdefghijklmnopqrstuvwxyz";; let expect2 = "nopqrstuvwxyzabcdefghijklm";;
let result3 = rot_n 42 "0123456789";;                 let expect3 = "0123456789";;
let result4 = rot_n  2 "OI2EAS67B9";;                 let expect4 = "QK2GCU67D9";;
let result5 = rot_n  0 "Damned !";;                   let expect5 = "Damned !";;
let result6 = rot_n 42 "";;                           let expect6 = "";;
let result7 = rot_n  1 "NBzlk qnbjr !";;              let expect7 = "OCaml rocks !";;

let (==) a b : bool = (a = b) ;;

assert (result1 == expect1) ;;
assert (result2 == expect2) ;;
assert (result3 == expect3) ;;
assert (result4 == expect4) ;;
assert (result5 == expect5) ;;
assert (result6 == expect6) ;;
assert (result7 == expect7) ;;
