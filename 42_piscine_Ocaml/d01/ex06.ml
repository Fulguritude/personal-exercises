let rec iter (f: (int -> int)) (x: int) (n: int): int =
	match n with
	| n when n > 0 -> iter (f) (f x) (n - 1)
	| 0            -> x
	| _            -> -1
;;


assert ( (iter (fun x -> x * x) 2 4) = 65536 );;
assert ( (iter (fun x -> x * 2) 2 4) =    32 );;
