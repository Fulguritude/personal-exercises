open Ex00_people;;

let tardis_ascii = "
        ___
_______(_@_)_______
| POLICE      BOX |
|_________________|
 | _____ | _____ |
 | |###| | |###| |
 | |###| | |###| |
 | _____ | _____ |
 | || || | || || |
 | ||_|| | ||_|| |
 | _____ |$_____ |
 | || || | || || |
 | ||_|| | ||_|| |
 | _____ | _____ |
 | || || | || || |
 | ||_|| | ||_|| |
 |_______|_______|
"
;;

let nest_json (s: string): string =
	let double_backslashes = Str.global_replace (Str.regexp {|\\|}) ("\\\\") (s) in
	let backslashed_quotes = Str.global_replace (Str.regexp {|\"|}) ("\\\"") (double_backslashes) in
	backslashed_quotes
;;

class doctor (x_name: string) (x_age: int) (x_person: people) =
	let () = print_endline ("Initializing doctor object") in
	object
		val         name     : string = x_name ;
		val mutable age      : int    = x_age ;
		val         sidekick : people = x_person ;
		val mutable hp       : int    = 100 ;

		method      get_name : string = name ;
		method      get_age  : int    = age  ;

		method to_string =
			Printf.sprintf "{name:\"%s\",age:%d,hp:%d,sidekick:%s}"
			(name)
			(age)
			(hp)
			(sidekick#to_string)
		;

		method talk = print_endline ("Hi! I'm the Doctor!");

		method travel_in_time (start: int) (arrival: int): unit =
			(* is the Doctor even changing age ? meh, it's just to show how to assign to a mutable field so idgaf *)
			print_endline(tardis_ascii);
			let yeardelta = arrival - start in
			age <- age + yeardelta
		;

		method use_sonic_screwdriver () = print_endline ("Whiiiiwhiiiwhiii Whiiiiwhiiiwhiii Whiiiiwhiiiwhiii");

		method private regenerate (): unit = hp <- 100;
	end
;;

let sidekick = new people ("Watson") ;;
let hero     = new doctor ("Holmes") (27) (sidekick) ;;

print_endline(hero#to_string);;
hero#talk;;
hero#use_sonic_screwdriver();;
hero#travel_in_time (2023) (1970);;
print_endline(string_of_int(hero#get_age));;
