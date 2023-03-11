open Ex00_atom;;
open Ex01_molecules;;

let op_fold_atomcount_list
	(l: atom_count list)
	(e: atom_count     )
: atom_count list =
	match (l, e) with
	| ((val1, count1) :: lis_t, (val2, count2)) when val1#equals(val2) ->
	(
		(val1, count1 + count2) :: lis_t
	)
	| (l, e) ->
	(
		e :: l
	)
;;

let fold_atomcount_list (atoms_count: atom_count list) =
	List.fold_left (op_fold_atomcount_list) ([]) (atoms_count)
;;

(*
	I know the subject asks for these tuple lists, but a hash table
	would probably have been better.
*)
type reactants = (molecule * int) list ;;

let atoms_count_of_reactants (r: reactants): atom_count list =
	let atoms_count = List.map (fun (x, mol_count) -> (x#to_atomcount_list, mol_count)) (r) in
	let atoms_count =
		List.map
		(
			fun (l, mol_count) ->
			(
				List.map
				(
					fun (atom, atom_count) ->
					(
						(atom, mol_count * atom_count)
					)
				)
				(l)
			)
		)
		(atoms_count)
	in
	let atoms_count = List.flatten (atoms_count) in
	let atoms_count = List.sort (fun (a1, _) (a2, u) -> a1#hill_cmp(a2)) (atoms_count) in
	let atoms_count = fold_atomcount_list (atoms_count) in
	atoms_count
;;

class virtual reaction (input: reactants) (output: reactants) =

	object (s: 's)
		method get_start  : reactants = if not s#is_balanced then failwith "Unbalanced reaction" ; input;
		method get_result : reactants = if not s#is_balanced then failwith "Unbalanced reaction" ; output;

		method virtual balance: reaction

		method is_balanced: bool =
			let input_counts  = atoms_count_of_reactants(input  ) in
			let output_counts = atoms_count_of_reactants(output ) in
			input_counts = output_counts
		;

		method to_string : string =
			let string_of_reactant = fun (m, c) -> string_of_int(c) ^ " " ^ m#to_string in
			let string_of_reactants =
				fun l ->
				(
					let strls = List.map (string_of_reactant) (l) in
					List.fold_left (fun x y -> x ^ " + " ^ y) ("") (strls)
				)
			in
			string_of_reactants (input) ^ " -> " ^ string_of_reactants (output)
		;

	end
;;

