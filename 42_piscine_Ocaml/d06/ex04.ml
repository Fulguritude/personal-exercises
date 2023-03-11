module type VAL =
sig
	type t ;;

	val add : t -> t -> t ;;
	val mul : t -> t -> t ;;
end


module Val_Int : (VAL with type t = int) =
struct
	type t = int ;;
	let add = ( + ) ;;
	let mul = ( * ) ;;
end

module Val_Float : (VAL with type t = float) =
struct
	type t = float ;;
	let add = ( +. ) ;;
	let mul = ( *. ) ;;
end

module Val_String : (VAL with type t = string) =
struct
	type t = string ;;
	let add s1 s2 = if (String.length s1) > (String.length s2) then s1 else s2 ;;
	let mul = ( ^ ) ;;
end

module type EVALEXPR =
sig
	type t ;;

	type expr = Add of (expr * expr) | Mul of (expr * expr) | Value of t ;;

	val eval : expr -> t ;;
end

module type MAKEEVALEXPR =
	functor (T: VAL) ->
	(
		EVALEXPR with type t = T.t
	)
;;

module MakeEvalExpr : MAKEEVALEXPR =
	functor (T : VAL) ->
	struct
		type t = T.t ;;
		type expr =
			| Add   of (expr * expr)
			| Mul   of (expr * expr)
			| Value of (T.t)
		;;
		let rec eval (v: expr): t =
			match v with
			| Add   (a, b) -> T.add (eval(a)) (eval(b))
			| Mul   (a, b) -> T.mul (eval(a)) (eval(b))
			| Value  a     -> a
		;;
	end
;;

module EvalExpr_Int    : (EVALEXPR with type t := Val_Int    .t) = MakeEvalExpr (Val_Int    ) ;;
module EvalExpr_Float  : (EVALEXPR with type t := Val_Float  .t) = MakeEvalExpr (Val_Float  ) ;;
module EvalExpr_String : (EVALEXPR with type t := Val_String .t) = MakeEvalExpr (Val_String ) ;;

let ie = EvalExpr_Int    .Add (EvalExpr_Int    .Value (40   ), EvalExpr_Int   .Value (2    ) );;
let fe = EvalExpr_Float  .Add (EvalExpr_Float  .Value (41.5 ), EvalExpr_Float .Value (0.92 ) );;
let se = EvalExpr_String .Mul (
	EvalExpr_String .Value ("very "),
	(
		EvalExpr_String.Add (
			EvalExpr_String.Value ("very long"),
			EvalExpr_String.Value ("short")
		)
	)
);;


let () = Printf.printf "Res = %d\n" (EvalExpr_Int    .eval (ie)) ;;
let () = Printf.printf "Res = %f\n" (EvalExpr_Float  .eval (fe)) ;;
let () = Printf.printf "Res = %s\n" (EvalExpr_String .eval (se)) ;;
