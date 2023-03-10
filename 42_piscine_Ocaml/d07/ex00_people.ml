class people x_name =
	let () = print_endline ("Initializing people object") in
	object
		val         name : string = x_name ;
		val mutable hp   : int    = 100 ;

		method set_hp (new_hp: int): unit = hp <- new_hp;

		method to_string = Printf.sprintf "{name:\"%s\",hp:%d}" (name) (hp);

		method talk = print_endline ("I'm " ^ name ^ "! Do you know the Doctor?");

		method die = print_endline ("Aaaarghh!");
	end
;;

let person = new people "Laios" ;;
print_endline(person#to_string);;
person#talk;;
person#die;;
