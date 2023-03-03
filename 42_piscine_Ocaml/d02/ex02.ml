let gray (n: int): unit =
	let concat_with_space = (fun x y -> (x ^ (if x = "" then "" else " ") ^ y)) in

	(* reflect and prefix method *)
	let rec rec_gray (n: int): string list =
		match n with
		| 0 ->
		(
			[""]
		)
		| n when n > 0 ->
		(
			let acc     = rec_gray (n - 1) in
			let rev_acc = List.rev (acc) in
			let acc     = List.map (fun x -> "0" ^ x) (     acc) in
			let rev_acc = List.map (fun x -> "1" ^ x) ( rev_acc) in
			let new_acc = List.append (acc) (rev_acc) in
			new_acc
		)
		| _ ->
		(
			["Error"]
		)
	in
	let result = rec_gray n in
	print_endline(List.fold_left concat_with_space "" result)
;;


gray 1;; (* 0 1 *)
gray 2;; (* 00 01 11 10 *)
gray 3;; (* 000 001 011 010 110 111 101 100 *)
