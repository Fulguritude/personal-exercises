class atom
	(x_name   : string )
	(x_symbol : string )
	(x_number : int    )
	(x_weight : float  )
=
	object (s)
		method name   = x_name;
		method symbol = x_symbol;
		method number = x_number;
		method weight = x_weight;

		method to_string =
			Printf.sprintf
			"{name:\"%s\",symbol:\"%s\",number:%d,weight:%f}"
			(s # name   )
			(s # symbol )
			(s # number )
			(s # weight )
		;

		method equals (other: atom) =
			s#name   = other#name   &&
			s#symbol = other#symbol &&
			s#number = other#number &&
			s#weight = other#weight
		;

		method hill_cmp (other: atom): int =
			match (s#symbol, other#symbol) with
			| "C", "C" ->  0
			| "C",   _ ->  1
			|   _, "C" -> -1
			| "H", "H" ->  0
			| "H",   _ ->  1
			|   _, "H" -> -1
			| (s, o) ->
			(
				if      s = o then ( 0)
				else if s < o then ( 1)
				else               (-1)
			)
		;
	end
;;


let hydrogen = new atom ("hydrogen" ) ("H"  ) (  1) (  1.008  );;
let carbon   = new atom ("carbon"   ) ("C"  ) (  6) ( 12.011  );;
let oxygen   = new atom ("oxygen"   ) ("O"  ) (  8) ( 15.999  );;
let nitrogen = new atom ("nitrogen" ) ("N"  ) (  7) ( 14.007  );;
let helium   = new atom ("helium"   ) ("He" ) (  2) (  4.0026 );;
let silicon  = new atom ("silicon"  ) ("Si" ) ( 14) ( 28.085  );;

let is_digit c : bool = '0' <= c && c <= '9';;
let is_lower c : bool = 'a' <= c && c <= 'z';;
let is_upper c : bool = 'A' <= c && c <= 'Z';;

let atom_of_string(symbol: string): atom =
	match symbol with
	| "H"  -> hydrogen
	| "C"  -> carbon
	| "O"  -> oxygen
	| "N"  -> nitrogen
	| "He" -> helium
	| "Si" -> silicon
	| _    -> failwith ("atom_of_string(): invalid atom symbol '" ^ symbol ^ "'")
;;

let atoms_of_string (s: string): atom list =
	let symbol = Str.global_replace (Str.regexp {|[0-9]|}    ) ("") (s) in
	let count  = Str.global_replace (Str.regexp {|[a-zA-Z]|} ) ("") (s) in
	let count  = if count = "" then 1 else int_of_string(count) in
	if symbol = "" then
		[]
	else
		List.init (count) (fun _ -> atom_of_string (symbol) )
;;
