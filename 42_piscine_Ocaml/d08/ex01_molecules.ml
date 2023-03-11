open Ex00_atom;;

type atom_count = (atom * int) ;;

class virtual molecule
	(x_name    : string )
	(x_formula : string )
=
	object (s: 's)

		method atoms : atom list =

			let splittable_str = Str.global_replace (Str.regexp {|\([A-Z]\)|}) (",\\1") (x_formula) in
			let split_str      = String.split_on_char (',') (splittable_str) in
			let raw_list_list  = List.map (atoms_of_string) (split_str) in
			let raw_list       = List.fold_left (@) ([]) (raw_list_list) in
			let sorted_list    = List.sort (fun x y -> x#hill_cmp(y)) (raw_list) in
			sorted_list
		;

		method name    = x_name;
		method formula = s#to_string;

		method to_atomcount_list: atom_count list =
			let fold_op (l: atom_count list) (a: atom): atom_count list =
				match l with
				| []                                                 -> [(a, 1)]
				| (prev_atom, count) :: t when prev_atom # equals(a) -> (a, count + 1) :: t
				| l                                                  -> (a, 1) :: l
			in
			let atom_counts = List.fold_left (fold_op) ([]) (s#atoms) in
			atom_counts
		;

		method to_string: string =
			let to_atoms_string ((a, count): atom * int): string =
				let count_str = if count = 1 then "" else string_of_int(count) in
				a # symbol ^ count_str
			in
			let strls  = List.map (to_atoms_string) (s#to_atomcount_list) in
			let result = List.fold_left (^) ("") (strls) in
			result
		;

		method equals (other: 's): bool =
			s#to_string = other#to_string
		;

		method cmp (o: 's): int =
			let s_str = s#to_string in
			let o_str = o#to_string in
			if      s_str = o_str then ( 0)
			else if s_str < o_str then (-1)
			else                       ( 1)
		;

		method get_carbon_rank : int =
			let l = s#to_atomcount_list in
			let v = List.find (fun (x, c) -> x#equals(carbon)) (l) in
			let (_, n) = v in
			n
		;
	end
;;

class water          = object inherit molecule ("water"          ) ("OH2"      ) end ;;
class carbon_dioxyde = object inherit molecule ("carbon_dioxyde" ) ("O2C"      ) end ;;
class dioxygen       = object inherit molecule ("dioxygen"       ) ("O2"       ) end ;;
class methane        = object inherit molecule ("methane"        ) ("H4C"      ) end ;;
class ethanol        = object inherit molecule ("ethanol"        ) ("OH6C2"    ) end ;;
class nitroglycerin  = object inherit molecule ("nitroglycerin"  ) ("N3C3H5O9" ) end ;;

let ex_water          = new water ;;
let ex_carbon_dioxyde = new carbon_dioxyde ;;
let ex_dioxygen       = new dioxygen ;;
let ex_methane        = new methane ;;
let ex_ethanol        = new ethanol ;;
let ex_nitroglycerin  = new nitroglycerin ;;

print_endline (ex_water          # to_string) ;;
print_endline (ex_carbon_dioxyde # to_string) ;;
print_endline (ex_methane        # to_string) ;;
print_endline (ex_ethanol        # to_string) ;;
print_endline (ex_nitroglycerin  # to_string) ;;
