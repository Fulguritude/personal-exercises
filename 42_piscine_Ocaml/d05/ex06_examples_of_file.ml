(*
Random.self_init() ;;

let shuffle_list (l: 'a list): 'a list =
	List.sort (fun n m -> (Random.int 11) - 5) (l)
;;
*)



(* List version because why not *)
(*
let read_file (filename: string): string list =
	let lines = ref [] in
	let in_channel = open_in (filename) in
	try
	(
		while true;
		do
			lines := input_line (in_channel) :: !lines
		done;
		!lines
	)
	with End_of_file ->
	(
		close_in in_channel;
		List.rev !lines
	)
;;

let convert_line (strls: string list): float list * string =
	let l_rev  = List.rev (strls) in
	match l_rev with
	| [] -> raise (Failure ("Invalid data format"))
	| s :: strls_rev ->
	(
		let strls = List.rev strls_rev in
		let f64ls = List.map (float_of_string) (strls) in
		(f64ls, s)
	)
;;
*)

let read_file (filename: string): string array =
	let lines = ref [||] in
	let in_channel = open_in (filename) in
	try
	(
		while true;
		do
			let new_value = [| input_line (in_channel)|] in
			lines := Array.append (!lines) (new_value)
		done;
		!lines
	)
	with End_of_file ->
	(
		close_in in_channel;
		!lines
	)
;;

let convert_line (strarr: string array): float array * string =
	let arr_len = Array.length strarr in
	let f64arr  = ref [||] in
	for i = 0 to arr_len - 2 do
		let new_value = [| float_of_string (strarr.(i)) |] in
		f64arr := Array.append (!f64arr) (new_value)
	done;
	(!f64arr, strarr.(arr_len - 1))
;;

let examples_of_file (filename: string): (float array * string) array =
	let strlist_rows  = read_file (filename) in
	let split_csv_row = fun x -> (Array.of_list (String.split_on_char (',') (x))) in
	let dataframe     = Array.map (split_csv_row ) (strlist_rows ) in
	let df_formatted  = Array.map (convert_line  ) (dataframe    ) in
	df_formatted
;;

let args = Sys.argv ;;
if Array.length (args) != 2 then raise (Failure ("expects single filename arg")) else () ;;
let filename = args.(1) ;;
let df_formatted = examples_of_file (filename) ;;
