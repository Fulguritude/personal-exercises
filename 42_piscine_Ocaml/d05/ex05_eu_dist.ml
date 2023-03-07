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

let v0 = [|  0.;  0.;  0. |] ;;
let v1 = [|  1.;  2.;  3. |] ;;
let v2 = [| -1.; -2.; -3. |] ;;

print_endline(string_of_float( eu_dist (v0) (v1) ));;
print_endline(string_of_float( eu_dist (v1) (v1) ));;
print_endline(string_of_float( eu_dist (v1) (v2) ));;
