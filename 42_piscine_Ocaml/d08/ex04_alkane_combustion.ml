open Ex00_atom;;
open Ex01_molecules;;
open Ex02_alkanes;;
open Ex03_reactions;;

(*
	(*
		There's surely a way to find the general answer using GCDs/LCMs of degree 1
		multivariate polynomials over the naturals, but let's not go down that rabbit hole
	*)

	let gcd (a: int) (b: int): int =
		let rec rec_gcd (a: int) (b: int): int =
			let rem = a mod b in
			match rem with
			| 0 -> Stdlib.abs(a)
			| _ -> rec_gcd (b) (rem)
		in
		rec_gcd (a) (b)
	;;
	
	let lcm (a: int) (b: int): int =
		a * b / gcd (a) (b)
	;;


	let atomcount_add
		(l1: atom_count list)
		(l2: atom_count list)
	: atom_count list =
		match (l1, l2) with
		| [], [] -> []
		| l1, [] -> l1
		| [], l2 -> l2
		| l1, l2 ->
		(
			let hill_cmp  = (fun (x, _) (y, _) -> x#hill_cmp(y)) in
			let l1_sorted = List.sort  (hill_cmp) (l1) in
			let l2_sorted = List.sort  (hill_cmp) (l2) in
			let l_merged  = List.merge (hill_cmp) (l1_sorted) (l2_sorted) in
			fold_atomcount_list (l_merged)
		)
	;;
	
	let atomcount_scale
		(l: atom_count list )
		(n: int             )
	: atom_count list =
		match l with
		| [] -> []
		| l ->
		(
			let result = List.map (fun (a, c) -> (a, n * c)) (l) in
			result
		)
	;;

 	(* incomplete attempt at a general reaction balancing function *)
	class alkane_combustion (input: reactants) (output: reactants) =
		object
			inherit reaction (input) (output)

			method balance : reaction =
				let extract_molecule = fun (x, _) -> x in
				let sort_molecules = 
					fun x y ->
					let x_str = x#to_string in
					let y_str = y#to_string in
					if      x_str = y_str then ( 0)
					else if x_str < y_str then (-1)
					else                       ( 1)
				in 
				let input_molecules  = List.sort_uniq (sort_molecules) (List.map (extract_molecule) (input  )) in
				let output_molecules = List.sort_uniq (sort_molecules) (List.map (extract_molecule) (output )) in
				let input_atoms  = List.sort_uniq (fun x y -> x#hill_cmp(y)) (List.flatten (List.map (fun x -> x#atoms) (input_molecules  ))) in
				let output_atoms = List.sort_uniq (fun x y -> x#hill_cmp(y)) (List.flatten (List.map (fun x -> x#atoms) (output_molecules ))) in
				if input_atoms != output_atoms then failwith "balance(): irreconcilable reaction";
			;
		end
	;;
*)

class alkane_combustion (reactants: alkane list) =
	(*
		a * C(n)H(2n+2) + b * O(2) = c * CO(2) + d H(2)O
		=>
		C :   n    a      -  c      = 0
		O :            2b - 2c -  d = 0
		H : (2n+2) a           - 2d = 0
		=> (setting a to 1 since it does the job)
		c =   n
		d =   n + 1
		b = (3n + 1) / 2
	*)
	object (s)
		inherit
			let products = [(ex_carbon_dioxyde, 1); (ex_water, 1)] in
			let l = List.sort_uniq (fun x y -> x#cmp(y)) (reactants) in
			reaction (List.map (fun x -> (x, 1)) (l)) (products)

		method balance : alkane_combustion =
			let ranks   = List.map (fun (x, _) -> x # get_carbon_rank) (s # get_start) in 
			let ranks   = List.filter (fun x -> x != 0) (ranks) in
			let alkanes = List.map (fun n -> (get_alkane(n),          n          ) ) (ranks) in
			let o2s     = List.map (fun n -> (ex_dioxygen,       (3 * n + 1) / 2 ) ) (ranks) in
			let co2s    = List.map (fun n -> (ex_carbon_dioxyde,      n          ) ) (ranks) in
			let h2os    = List.map (fun n -> (ex_water,               n + 1      ) ) (ranks) in
			let new_input  : (molecule * int) list = List.flatten [o2s; alkanes] in
			let new_output : (molecule * int) list = List.flatten [co2s; h2os] in
			new alkane_combustion (new_input) (new_output)
		;
	end
;;