
type radar = float array * string ;;

let eu_dist (a1: float array) (a2: float array): float =
	let len1 = Array.length (a1) in
	let len2 = Array.length (a2) in
	if len1 != len2 then raise (Failure ("eu_dist: mismatched lengths"));

	let acc = ref 0. in
	for i = 0 to len1 - 1
	do
		let dist = a1.(i) -. a2.(i) in
		let quad = dist   *. dist   in
		acc := !acc +. quad
	done;
	Stdlib.sqrt (!acc)
;;

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

let format_csv_line (str: string): radar =
	let strarr  = Array.of_list ( String.split_on_char (',') (str) ) in
	let arr_len = Array.length strarr in
	let f64arr  = ref [||] in
	for i = 0 to arr_len - 2 do
		let new_value = [| float_of_string (strarr.(i)) |] in
		f64arr := Array.append (!f64arr) (new_value)
	done;
	(!f64arr, strarr.(arr_len - 1))
;;


let read_and_format_csv (filename: string): radar list =
	let strlist_rows  = read_file (filename) in
	let df_formatted  = List.map (format_csv_line) (strlist_rows) in
	df_formatted
;;


let k_nn (df: radar list) (k: int) (test_val: radar): string =
	let (v_test, c_test) = test_val in
	let distances        = List.map  (fun (v, c)          -> let res = eu_dist (v) (v_test) in (res, c) ) (df        ) in
	let sorted_distances = List.sort (fun (d1, _) (d2, _) -> if d1 -. d2 < 0. then -1 else 1            ) (distances ) in
	match sorted_distances with
	| [] -> failwith "k_nn(): no dataset"
	| l  ->
	(
		if k > List.length (l) then failwith "k_nn(): dataset too small for k";
		let (_, l_classes) = List.split (l) in
		let add_pair = (fun pairpair -> let (t1x, t1y), (t2x, t2y) = pairpair in (t1x + t2x, t1y + t2y)) in
		let good_bad = ref (0, 0) in
		for i = 0 to k - 1 do
			let klass = List.nth (l_classes) (i) in
			if klass = "g" then (good_bad := add_pair (!good_bad, (1, 0)))
			else                (good_bad := add_pair (!good_bad, (0, 1)))
		done;
		let good, bad = !good_bad in
		if      good > bad then
			"g"
		else if good < bad then
			"b"
		else
			List.nth l_classes (0)
	)
;;


(* Took the first line and rounded it up *)
let test_val_str = "1,0,0.995,-0.058,0.852,0.023,0.833,-0.377,1,0.037,0.852,-0.177,0.597,-0.449,0.605,-0.382,0.843,-0.385,0.582,-0.321,0.569,-0.296,0.369,-0.473,0.568,-0.512,0.411,-0.462,0.213,-0.341,0.423,-0.545,0.186,-0.453,g";;
let test_val     = format_csv_line (test_val_str) ;;

let args = Sys.argv ;;
if Array.length (args) != 2 then raise (Failure ("expects single filename arg")) else () ;;
let filename = args.(1) ;;
let df_formatted = read_and_format_csv (filename) ;;
let result = k_nn (df_formatted) (10) (test_val) ;;

print_endline (result) ;;
