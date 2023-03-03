(* Look and Say sequence *)

let rec sequence (n: int): string =
	let op_fold_countingtuple
		(l: (int * char) list)
		(e: int * char)
	: (int * char) list =
		match (l, e) with
		| ((count1, val1) :: lis_t, (count2, val2)) when val1 = val2 ->
		(
			(count1 + count2, val1) :: lis_t
		)
		| (l, e) ->
		(
			e :: l
		)
	in

	let countingtuple_to_string (t: int * char): string =
		match t with
		| (count, value) ->
			string_of_int(count) ^ (String.make 1 value)
	in

	match n with
	| 0 -> "1"
	| n when n > 0 ->
	(
		let s       = sequence (n - 1) in
		let s_len   = String.length s in
		let l_char  = List.init (s_len) (String.get s) in
		let l_count = List.map (fun x -> (1, x)) l_char in
		let l_fold  = List.fold_left op_fold_countingtuple [] l_count in
		let l_rev   = List.rev l_fold in
		let l_str   = List.map (countingtuple_to_string) l_rev in
		let s_str   = List.fold_left (^) "" l_str in
		s_str
	)
	| _ -> "Error"
;;

assert ( sequence 0 = "1"        ) ;;
assert ( sequence 1 = "11"       ) ;;
assert ( sequence 2 = "21"       ) ;;
assert ( sequence 3 = "1211"     ) ;;
assert ( sequence 4 = "111221"   ) ;;
assert ( sequence 5 = "312211"   ) ;;
assert ( sequence 6 = "13112221" ) ;;
