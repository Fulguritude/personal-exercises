open Ex00_people;;
open Ex01_doctor;;

Random.self_init();;

let get_random_char (): char =
	let possible_val_amount = 26 * 2 + 10 in
	let random_val          = Random.int (possible_val_amount) in
	match random_val with
	| n when n < 26 -> char_of_int(int_of_char('a') + n     )
	| n when n < 52 -> char_of_int(int_of_char('A') + n - 26)
	| n             -> char_of_int(int_of_char('0') + n - 52)
;;


let get_random_chars (): string =
	let char1 = String.make 1 (get_random_char()) in
	let char2 = String.make 1 (get_random_char()) in
	let char3 = String.make 1 (get_random_char()) in
	char1 ^ char2 ^ char3
;;

class dalek =
	object
		val         name   : string = "Dalek" ^ get_random_chars();
		val mutable hp     : int    = 100 ;
		val mutable shield : bool   = true ;

	method to_string : string = Printf.sprintf "{name:\"%s\",hp:%d,shield:%b}" (name) (hp) (shield);

	method talk =
		match Random.int (4) with
		| 0 -> print_endline ("Explain! Explain!");
		| 1 -> print_endline ("Exterminate! Exterminate!");
		| 2 -> print_endline ("I obey!");
		| _ -> print_endline ("You are the Doctor! You are the enemy of the Daleks!");
	;

	method exterminate (p: people): unit = p#set_hp (0); shield <- false;

	method die: unit = print_endline ("Emergency Temporal Shift!");

	end
;;

