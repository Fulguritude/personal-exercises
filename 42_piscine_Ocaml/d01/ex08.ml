let rec sum
	( f           : (int -> float) )
	( bound_start : int            )
	( bound_final : int            )
: float =
	if bound_start > bound_final then
		Float.nan
	else
		let rec terminal_rec_sum
			(acc  : float          )
			(f    : (int -> float) )
			(iter : int            )
		: float =
			let acc = acc +. f(iter) in
			match iter with
			| i when i > bound_start -> terminal_rec_sum (acc) (f) (i - 1)
			| i                      -> acc
		in
		terminal_rec_sum (0.) (f) (bound_final)
;;

let sqr = fun i -> float_of_int (i * i);;
assert ( sum (sqr) (1) (10) = 385. );;
