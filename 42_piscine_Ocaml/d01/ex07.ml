let rec converges
	( f          : ('a -> 'a) )
	( x          : 'a         )
	( iterations : int        )
: bool =
	let new_x = f x in
	let result =
		if new_x = x then
		(
			true
		)
		else
		(
			match iterations with
			| i when i > 0 -> converges (f) (new_x) (i - 1)
			(* | 0            -> false *)
			| _            -> false
		)
	in
	result
;;

assert ( converges        (( * ) 2) 2 5 = false ) ;;
assert ( converges (fun x -> x / 2) 2 3 = true  ) ;;
assert ( converges (fun x -> x / 2) 2 2 = true  ) ;;
