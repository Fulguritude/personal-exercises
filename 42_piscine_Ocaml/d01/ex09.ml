let rec leibniz_pi (delta: float): int =
	let expr =
		fun (i: int): float ->
		(
			let fl = float_of_int(i) in
			4. *. ((-1.) ** fl) /. (2. *. fl +. 1.) (* ignoring x4 *)
		)
	in

	let default_pi = 4. *. atan (1.) in (* ignoring x4 *)

	let rec terminal_rec_leibniz_pi (acc: int) (value: float): int =
		let found_delta = default_pi -. value in
		(* print_endline(string_of_float(found_delta)); *)
		if Float.abs(found_delta) < delta then
			acc
		else
			terminal_rec_leibniz_pi (acc + 1) (value +. expr(acc))
	in

	terminal_rec_leibniz_pi 0 0.
;;

let delta = 0.000005;;
let nb_steps  = leibniz_pi (delta);;
print_endline (string_of_int(nb_steps));
