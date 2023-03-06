module Value = struct

	type t = 
		| T2 
		| T3 
		| T4 
		| T5 
		| T6 
		| T7 
		| T8 
		| T9 
		| T10 
		| Jack 
		| Queen 
		| King 
		| Ace
	;;


	let all (): t list =
		[
			T2;
			T3;
			T4;
			T5;
			T6;
			T7;
			T8;
			T9;
			T10;
			Jack;
			Queen;
			King;
			Ace
		]
	;;

	let toInt (card: t): int =
		match card with
		| T2    -> 1
		| T3    -> 2
		| T4    -> 3
		| T5    -> 4
		| T6    -> 5
		| T7    -> 6
		| T8    -> 7
		| T9    -> 8
		| T10   -> 9
		| Jack  -> 10
		| Queen -> 11
		| King  -> 12
		| Ace   -> 13
	;;

	(** returns "2", ..., "10", "J", "Q", "K" or "A" *)
	let toString(card: t): string =
		match card with
		| T2    -> "2"
		| T3    -> "3"
		| T4    -> "4"
		| T5    -> "5"
		| T6    -> "6"
		| T7    -> "7"
		| T8    -> "8"
		| T9    -> "9"
		| T10   -> "10"
		| Jack  -> "J"
		| Queen -> "Q"
		| King  -> "K"
		| Ace   -> "A"
	;;

	(** returns "2", ..., "10", "Jack", "Queen", "King" or "As" *)
	let toStringVerbose (card: t): string =
		match card with
		| T2    -> "2"
		| T3    -> "3"
		| T4    -> "4"
		| T5    -> "5"
		| T6    -> "6"
		| T7    -> "7"
		| T8    -> "8"
		| T9    -> "9"
		| T10   -> "10"
		| Jack  -> "Jack"
		| Queen -> "Queen"
		| King  -> "King"
		| Ace   -> "Ace"
	;;

	let next (card: t): t =
		match card with
		| T2    -> T3
		| T3    -> T4
		| T4    -> T5
		| T5    -> T6
		| T6    -> T7
		| T7    -> T8
		| T8    -> T9
		| T9    -> T10
		| T10   -> Jack
		| Jack  -> Queen
		| Queen -> King
		| King  -> Ace
		| Ace   -> invalid_arg "Value.next(): Ace has no next card"
	;;

	let previous (card: t): t =
		match card with
		| T2    -> invalid_arg "Value.previous(): T2 has no previous card"
		| T3    -> T2
		| T4    -> T3
		| T5    -> T4
		| T6    -> T5
		| T7    -> T6
		| T8    -> T7
		| T9    -> T8
		| T10   -> T9
		| Jack  -> T10
		| Queen -> Jack
		| King  -> Queen
		| Ace   -> King
	;;
end
