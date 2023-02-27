let rec ackermann (m: int) (n: int): int =
	match (m, n) with
	| (m, n) when m = 0          -> n + 1
	| (m, n) when m > 0 && n = 0 -> ackermann (m - 1) (1)
	| (m, n) when m > 0 && n > 0 -> ackermann (m - 1) (ackermann (m) (n-1))
	| _                          -> -1
;;

assert ( ackermann (-1) 7 =    -1 );;
assert ( ackermann ( 0) 0 =     1 );;
assert ( ackermann ( 2) 3 =     9 );;
assert ( ackermann ( 4) 1 = 65533 );;
