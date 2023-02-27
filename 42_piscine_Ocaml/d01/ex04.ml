let rec fibonacci (n : int): int =
	match n with
	| 0            -> 0
	| 1            -> 1
	| n when n > 0 -> fibonacci (n - 1) + fibonacci (n - 2)
	| _            -> -1
;;

assert ( fibonacci (-42)  = -1 ) ;;
assert ( fibonacci (  1)  =  1 ) ;;
assert ( fibonacci (  3)  =  2 ) ;;
assert ( fibonacci (  6)  =  8 ) ;;
