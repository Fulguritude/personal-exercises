let rec tak (x: int) (y: int) (z: int): int =
	match (x, y, z) with
	| (x, y, z) when y < x ->
		tak
			(tak (x - 1) (y) (z))
			(tak (y - 1) (z) (x))
			(tak (z - 1) (x) (y))
	| _ -> z
;;

assert ( tak 1         2     3 =     3 );;
assert ( tak 5        23     7 =     7 );;
assert ( tak 9         1     0 =     1 );;
assert ( tak 1         1     1 =     1 );;
assert ( tak 0        42     0 =     0 );;
assert ( tak 23498 98734 98776 = 98776 );;
