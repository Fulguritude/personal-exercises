(*
	Symbol reuse can often be ambiguous, but for a language like OCaml where
	+ and +. are distinct, having assignment and boolean equality use the
	same symbol is really maddening.
*)
let (==) a b : bool = (a = b) ;;

let rec power (a: int) (b: int): int =
	match b with
	| 0 -> 1
	| 1 -> a
	| n -> 
		let b             = power a (n / 2)                 in
		let odd_power_val = (if n mod 2 == 0 then 1 else a) in
		b * b * odd_power_val
;;

let result1 = string_of_int(power 2 4);;
let result2 = string_of_int(power 3 0);;
let result3 = string_of_int(power 0 5);;

print_endline(result1);;
print_endline(result2);;
print_endline(result3);;

