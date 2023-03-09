module type PAIR =
	sig val pair : (int * int)
end

module type VAL =
	sig val x : int
end

(* FIX ME !!! *)
module Pair = struct
	let pair = ( 21, 42 );;
end

module MakeFst (X : PAIR) = struct
	let (x, _) = X.pair;;
end

module MakeSnd (X : PAIR) = struct
	let (_, x) = X.pair;;
end

module Fst : VAL = MakeFst (Pair) ;;
module Snd : VAL = MakeSnd (Pair) ;;

let () = Printf.printf "Fst.x = %d, Snd.x = %d\n" Fst.x Snd.x ;;
