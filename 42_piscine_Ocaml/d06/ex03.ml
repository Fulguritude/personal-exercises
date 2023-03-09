module type FIXED = sig
	type t

	val of_float : float -> t
	val of_int   : int   -> t

	val to_float  : t -> float
	val to_int    : t -> int
	val to_string : t -> string

	val zero : t
	val one  : t

	val succ : t -> t
	val pred : t -> t

	val min : t -> t -> t
	val max : t -> t -> t

	val gth : t -> t -> bool
	val lth : t -> t -> bool
	val gte : t -> t -> bool
	val lte : t -> t -> bool
	val eqp : t -> t -> bool (** physical equality *)
	val eqs : t -> t -> bool (** structural equality *)

	val add : t -> t -> t
	val sub : t -> t -> t
	val mul : t -> t -> t
	val div : t -> t -> t

	val foreach : t -> t -> (t -> unit) -> unit
end

module type FRACTIONAL_BIT_AMOUNT = sig
	(* describes the amount of binary values between each int *)
	val bits : int
end

let powers_of_ten =
	[
		1.;
		10.;
		100.;
		1000.;
		10000.;
		100000.;
		1000000.;
		10000000.;
		100000000.;
		1000000000.;
		10000000000.;
		100000000000.;
		1000000000000.;
		10000000000000.;
		100000000000000.;
		10000000000000000.;
		100000000000000000.;
		1000000000000000000.;
		10000000000000000000.;
		100000000000000000000.;
	]
;;

module Make (T : FRACTIONAL_BIT_AMOUNT): FIXED = struct
	type t = int ;;


	let zero : t = 0;;
	let one  : t = Int.shift_left (1) (T.bits) ;;

	let succ (v: t): t = v + 1 ;;
	let pred (v: t): t = v - 1 ;;

	let get_floor_bits (v: t) = Int.shift_right (v) (T.bits);;
	let get_fract_bits (v: t) = Int.logand (v) (pred(one));;

	let of_float  (v: float): t =
		(* let mantissa, exponent = frexp (v) in *)
		let fract, floor = modf (v) in
		let floor_bits   = Int.shift_left (int_of_float (floor)) (T.bits) in
		let fract_bits   = int_of_float ( Float.round( ldexp (fract) (T.bits) ) ) in
		let result       = floor_bits + fract_bits in
		result
	;;

	let of_int    (v: int): t =
		Int.shift_left (v) (T.bits)
	;;

	let to_float  (v: t): float =
		let floor_bits = get_floor_bits(v) in
		let fract_bits = get_fract_bits(v) in
		let floor = float_of_int (floor_bits) in
		let fract = ldexp (float_of_int (fract_bits)) (-T.bits) in
		floor +. fract
	;;

	let to_int    (v: t): int =
		let floor_bits = get_floor_bits(v) in
		floor_bits
	;;

	let to_string (v: t): string =
		let sign_v, abs_v = if v >= 0 then "", v else "-", -v in
		let floor_bits = get_floor_bits(abs_v) in
		let fract_bits = get_fract_bits(abs_v) in
		let fract_f64  = to_float(fract_bits) in
		let two_by_five_multiplier = List.nth (powers_of_ten) (T.bits) in
		let fract_decimal_repr = int_of_float( fract_f64 *. two_by_five_multiplier ) in
		let fract_str = string_of_int(fract_decimal_repr) in
		let padding = String.init (T.bits - String.length (fract_str)) (fun _ -> '0') in
		sign_v ^ string_of_int(floor_bits) ^ "." ^ padding ^ fract_str
	;;

	let min  (a: t) (b: t): t = if a < b then a else b ;;
	let max  (a: t) (b: t): t = if a < b then b else a ;;

	let lth  (a: t) (b: t): bool = a <  b;;
	let gth  (a: t) (b: t): bool = a >  b;;
	let lte  (a: t) (b: t): bool = a <= b;;
	let gte  (a: t) (b: t): bool = a >= b;;
	let eqs  (a: t) (b: t): bool = a =  b;; (** structural equality *)
	let eqp  (a: t) (b: t): bool = a == b;; (** physical equality *)

	let add  (a: t) (b: t): t = a + b;;
	let sub  (a: t) (b: t): t = a - b;;
	let mul  (a: t) (b: t): t = a * b;;
	let div  (a: t) (b: t): t = a / b;;

	let foreach (first: t) (final: t) (do_op: (t -> unit)): unit =
		let l = List.init (final - first + 1) (fun i -> first + i) in
		List.iter (do_op) (l)
	;;
end

module Fixed4 : FIXED = Make (struct let bits = 4 end) ;;
module Fixed8 : FIXED = Make (struct let bits = 8 end) ;;

let () =
	let x8 = Fixed8.of_float (-21.10) in
	let y8 = Fixed8.of_float 21.32 in
	let r8 = Fixed8.add x8 y8 in
	print_endline (Fixed8.to_string x8);
	print_endline (Fixed8.to_string y8);
	print_endline (Fixed8.to_string r8);
	Fixed4.foreach (Fixed4.zero) (Fixed4.one) (fun f -> print_endline (Fixed4.to_string f))
;;
