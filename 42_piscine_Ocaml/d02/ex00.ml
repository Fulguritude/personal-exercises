let concat_with_comma = (fun x y -> (x ^ (if x = "" then "" else ", ") ^ y));;

let encode (l: 'a list): (int * 'a) list =
	let rec rec_encode
		(acc : (int * 'a) list)
		(lis :        'a  list)
	: (int * 'a) list =
		(* print_endline(List.fold_left (concat_with_comma) "" (List.map string_of_int lis)); *)
		match lis with
		| [] ->
			acc
		| lis_h :: lis_t ->
			let new_acc = 
				match acc with
					| [] ->
						[(1, lis_h)]
					| acc_h :: acc_t ->
						match (acc_h, lis_h) with
						| ((acc_count, acc_val), lis_val) ->
							if acc_val = lis_val then
								(acc_count + 1, acc_val) :: acc_t
							else
								(1, lis_val) :: acc
			in
			rec_encode new_acc lis_t
	in
	let result = rec_encode [] l in
	List.rev result
;;

let int_tuple_to_string (t: int * int): string =
	match t with | (t1, t2) ->
	"(" ^ string_of_int(t1) ^ "," ^ string_of_int(t2) ^ ")"
;;

let test1 = [0; 1; 1; 2; 3; 3; 0; 0];;

let result1 = encode(test1);;
let formatted_result = List.map int_tuple_to_string result1 ;;
print_endline(List.fold_left concat_with_comma "" formatted_result);;




(*
	Version without using a tuple because
	https://stackoverflow.com/questions/26005545/extract-nth-element-of-a-tuple
*)

type 'a rle_atom =
{
	count: int;
	value: 'a;
};;

let encode (l: 'a list): 'a rle_atom list =
	let rec rec_encode
		(acc : 'a rle_atom list)
		(lis : 'a          list)
	: 'a rle_atom list =
		match lis with
		| [] ->
		(
			acc
		)
		| lis_h :: lis_t ->
		(
			let new_acc =
			(
				match acc with
				| [] ->
				(
					[{count = 1; value = lis_h}]
				)
				| acc_h :: acc_t ->
				(
					match (acc_h, lis_h) with
					| ({ count = acc_count; value = acc_val }, lis_val) ->
					(
						if acc_val = lis_val then
							{count = acc_count + 1; value = acc_val} :: acc_t
						else
							{count = 1; value = lis_val} :: acc
					)
				)
			)
			in
			rec_encode new_acc lis_t
		)
	in
	let result = rec_encode [] l in
	List.rev result
;;



let rle_atom_to_string (a: 'a rle_atom): string =
	match a with | {count = count; value = value} ->
	"{count:" ^ string_of_int(count) ^ ", value:" ^ string_of_int(value) ^ "}"
;;

let test1 = [0; 1; 1; 2; 3; 3; 0; 0];;

let result1 = encode(test1);;
let formatted_result = List.map rle_atom_to_string result1 ;;
print_endline(List.fold_left concat_with_comma "" formatted_result);;
