open Ex01_molecules;;

class alkane (x_name: string) (n: int) =
	object
		inherit
			let c_count = if n = 1 then "" else string_of_int(n) in
			let h_count = string_of_int(2 * n + 2) in
			molecule (x_name) ("C" ^ c_count ^ "H" ^ h_count)
	end
;;

let methane  = new alkane ("methane"  ) (  1) ;;
let ethane   = new alkane ("ethane"   ) (  2) ;;
let propane  = new alkane ("propane"  ) (  3) ;;
let butane   = new alkane ("butane"   ) (  4) ;;
let pentane  = new alkane ("pentane"  ) (  5) ;;
let hexane   = new alkane ("hexane"   ) (  6) ;;
let heptane  = new alkane ("heptane"  ) (  7) ;;
let octane   = new alkane ("octane"   ) (  8) ;;
let nonane   = new alkane ("nonane"   ) (  9) ;;
let decane   = new alkane ("decane"   ) ( 10) ;;
let undecane = new alkane ("undecane" ) ( 11) ;;
let dodecane = new alkane ("dodecane" ) ( 12) ;;

let get_alkane (n: int): alkane =
	match n with
	|  1 -> methane
	|  2 -> ethane
	|  3 -> propane
	|  4 -> butane
	|  5 -> pentane
	|  6 -> hexane
	|  7 -> heptane
	|  8 -> octane
	|  9 -> nonane
	| 10 -> decane
	| 11 -> undecane
	| 12 -> dodecane
	|  _ -> failwith "get_alkane(): unimplemented"
;;

print_endline("\n" ^ "Alkanes:");
print_endline(methane  #to_string);
print_endline(ethane   #to_string);
print_endline(propane  #to_string);
print_endline(butane   #to_string);
print_endline(pentane  #to_string);
print_endline(hexane   #to_string);
print_endline(heptane  #to_string);
print_endline(octane   #to_string);
print_endline(nonane   #to_string);
print_endline(decane   #to_string);
print_endline(undecane #to_string);
print_endline(dodecane #to_string);