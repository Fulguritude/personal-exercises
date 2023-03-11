module type MONOID =
	sig
		type element
		val zero1 : element 
		val zero2 : element 
		val add : element -> element -> element
		val sub : element -> element -> element
		val mul : element -> element -> element
		val div : element -> element -> element
	end
;;

(* don't add the module type signature `: MONOID` or you get errors *)
module INT =
	struct
		type element = int ;;
		let zero1 = 0 ;;
		let zero2 = 1 ;;

		let add = ( + ) ;;
		let sub = ( - ) ;;
		let mul = ( * ) ;;
		let div = ( / ) ;;
	end
;;

module FLOAT =
	struct
		type element = float ;;
		let zero1 = 0. ;;
		let zero2 = 1. ;;

		let add = ( +. );;
		let sub = ( -. );;
		let mul = ( *. );;
		let div = ( /. );;
	end
;;


module type CALC =
	functor (M : MONOID) ->
	sig
		val add   : M.element -> M.element -> M.element
		val sub   : M.element -> M.element -> M.element
		val mul   : M.element -> M.element -> M.element
		val div   : M.element -> M.element -> M.element
		val power : M.element -> int       -> M.element
		(* factorial on floats ? what, you want me to compute Gamma(x) ? Lanczos ? too lazy *)
		(* val fact  : M.element              -> M.element *)
	end
;;

module Calc: CALC =
	functor (M : MONOID) ->
	struct
		type element = M.element
		let add = M.add ;;
		let sub = M.sub ;;
		let mul = M.mul ;;
		let div = M.div ;;
		let power a n = List.fold_left (M.mul) (M.zero2) (List.init (n) (fun _ -> a));;
		(* let fact  a   = ;; *)
	end
;;	

module Calc_int   = Calc(INT   ) ;;
module Calc_float = Calc(FLOAT ) ;;

let () =
	print_endline (string_of_int   (Calc_int   .power 3   3));
	print_endline (string_of_float (Calc_float .power 3.0 3));
	print_endline (string_of_int   (Calc_int   .mul (Calc_int   .add 20   1  ) 2  ));
	print_endline (string_of_float (Calc_float .mul (Calc_float .add 20.0 1.0) 2.0))
;;
