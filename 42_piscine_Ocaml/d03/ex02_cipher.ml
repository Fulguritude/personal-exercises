(*
	electing to not do in 2 files, since it really is irrelevant,
	I've already proved I know how to structure into modules and import
	in the previous exercise
*)

type case = Upper | Lower ;;

let is_upper (c : char) : bool = 'A' <= c && c <= 'Z';;
let is_lower (c : char) : bool = 'a' <= c && c <= 'z';;

let rot_n (rot : int) (s : string) : string =
	let rotate_val (base : case) (rot : int) (c : char) : char =
		let val_base = match base with
			| Upper -> int_of_char('A')
			| Lower -> int_of_char('a')
		in
		let val_old = int_of_char(c) - val_base in
		let val_rot = (val_old + rot) mod 26 in
		let val_rot = if val_rot < 0 then val_rot + 26 else val_rot in
		let val_new = val_rot + val_base in
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

let   rot42 (s: string): string = rot_n ( 42) (s);;
let unrot42 (s: string): string = rot_n (-42) (s);;

let   caesar (rot: int) (s: string): string = rot_n ( rot) (s);;
let uncaesar (rot: int) (s: string): string = rot_n (-rot) (s);;

let xor (key: int) (s: string): string =
	let xor_char = fun c -> char_of_int(key lxor int_of_char (c)) in
	String.map (xor_char) s
;;
let unxor (key: int) (s: string): string = xor (key) (s);;


let rec encrypt   (s: string) (f_list: (string -> string) list): string =
	match f_list with
	| [] -> s
	| f_h :: f_t -> encrypt (f_h (s)) (f_t)
;;

let rec decrypt (s: string) (f_list: (string -> string) list): string =
	match f_list with
	| [] -> s
	| f_h :: f_t -> decrypt (f_h (s)) (f_t)
;;

let   xor42 =   xor 42;;
let unxor42 = unxor 42;;

let   caesar13 =   caesar 13;;
let uncaesar13 = uncaesar 13;;

let   cipher =          [   rot42 ;   xor42;   caesar13 ] ;;
let decipher = List.rev [ unrot42 ; unxor42; uncaesar13 ] ;;

let test_string = "Hello SpongeBob ! Hi Patrick !"

let result_encrypted = encrypt (test_string      ) (  cipher) ;;
let result_decrypted = decrypt (result_encrypted ) (decipher) ;;

print_endline(test_string);
print_endline(result_encrypted);
print_endline(result_decrypted);
