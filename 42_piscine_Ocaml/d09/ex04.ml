(* "Forbidden functions: None", lol, gonna be simple and do based on lists *)

let rec list_contains (a: 'a) (l: 'a list): bool =
	List.exists (fun x -> x = a) (l)	
;;

let rec list_uniq (l: 'a list): 'a list =
	match l with
	| [] -> []
	| h :: t ->
	(
		let acc      = list_uniq (t) in
		let contains = list_contains (h) (acc) in
		let result   = if contains then acc else h :: acc in
		result
	)
;;

module Set =
	struct
		type 'a t = 'a list;;

		let return (a: 'a): 'a t = [a] ;;

		let flatten (ssa: 'a t t): 'a t =
			let la = List.flatten ssa in
			let sa = list_uniq (la) in
			sa
		;;

		let map (sa: 'a t) (f: 'a -> 'b): 'b t =
			List.map (f) (sa)
		;;

		let bind (sa: 'a t) (sf: 'a -> 'b t): 'b t =
			flatten (map (sa) (sf))
		;;

		let union (sa1: 'a t) (sa2: 'a t): 'a t =
			let lu = sa1 @ sa2 in
			let su = list_uniq (lu) in
			su
		;;

		let inter (sa1: 'a t) (sa2: 'a t): 'a t =
			List.filter (fun x -> list_contains (x) (sa2)) (sa1)
		;;

		let diff (sa1: 'a t) (sa2: 'a t): 'a t =
			List.filter (fun x -> not (list_contains (x) (sa2))) (sa1)
		;;

		let filter (sa: 'a t) (pred :'a -> bool): 'a t =
			List.filter (pred) (sa)
		;;

		let foreach (sa: 'a t) (do_op: 'a -> unit): unit =
			let _ = List.map (do_op) (sa) in
			()
		;;

		let for_all (sa: 'a t) (pred: 'a -> bool): bool =
			List.for_all (pred) (sa)
		;;

		let exists (sa: 'a t) (pred: 'a -> bool): bool =
			List.exists (pred) (sa)
		;;

	end
;;