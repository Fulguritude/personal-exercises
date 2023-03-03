let concat_with_comma = (fun x y -> (x ^ (if x = "" then "" else ", ") ^ y));;

let crossover (* should be named inter *)
	(l1 : 'a list)
	(l2 : 'a list)
: 'a list =
	let contains (l : 'a list) (a : 'a): bool = List.exists (fun b -> b = a) l in
	let rec rec_inter
		(acc : 'a list)
		(l1  : 'a list)
		(l2  : 'a list)
	: 'a list =
		match (l1, l2) with
		| [], _ -> acc
		| _, [] -> acc
		| (l1_h :: l1_t, l2) when contains l2 l1_h ->
			(* Which is faster: creating a new smaller list each time, or parsing over full initial list ? *)
			(* let new_l1 = l1_t in *)
			(* let new_l2 = l2   in *)
			let new_l1 = List.filter (fun elem -> elem != l1_h) l1_t in
			let new_l2 = List.filter (fun elem -> elem != l1_h) l2   in
			if contains acc l1_h then
				rec_inter          acc  new_l1 new_l2
			else
				rec_inter (l1_h :: acc) new_l1 new_l2
		| (l1_h :: l1_t, l2) ->
			rec_inter acc l1_t l2


	in
	rec_inter [] l1 l2
;;



let l1 = [4; 5; 1; 2; 3; 3];;
let l2 = [3; 5; 6];;

let l_inter = crossover l1 l2 ;;

let formatted_result = List.map string_of_int l_inter ;;
print_endline(List.fold_left concat_with_comma "" formatted_result);;
