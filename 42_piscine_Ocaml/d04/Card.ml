(* Electing to use open instead of copy-pasting *)

open Color;;
open Value;;


let cartesian (l1: 'a list) (l2: 'b list): (('a * 'b) list) =
	let ll =
		List.map
			(
				fun e1 ->
					List.map
						(
							fun e2 -> (e1, e2)
						)
						(l2)
			)
			(l1)
	in
	List.concat (ll)
;;

module Card = struct
	
	type t =
	{
		color: Color.t;
		value: Value.t;
	};;


	let newCard (v: Value.t) (c: Color.t): t = {value = v; color = c} ;;
	let getValue (card: t): Value.t = card.value ;;
	let getColor (card: t): Color.t = card.color ;;


	let isSpade   (card: t): bool = (card.color = Spade   ) ;;
	let isHeart   (card: t): bool = (card.color = Heart   ) ;;
	let isDiamond (card: t): bool = (card.color = Diamond ) ;;
	let isClub    (card: t): bool = (card.color = Club    ) ;;
	let isOf      (card: t) (color: Color.t): bool =
		match color with
		| Spade   -> isSpade   (card)
		| Heart   -> isHeart   (card)
		| Diamond -> isDiamond (card)
		| Club    -> isClub    (card)
	;;


	let isAllSpades   (cards: t list) = List.fold_left (fun x y -> x && isSpade   (y) ) (true) (cards) ;;
	let isAllHearts   (cards: t list) = List.fold_left (fun x y -> x && isHeart   (y) ) (true) (cards) ;;
	let isAllDiamonds (cards: t list) = List.fold_left (fun x y -> x && isDiamond (y) ) (true) (cards) ;;
	let isAllClubs    (cards: t list) = List.fold_left (fun x y -> x && isClub    (y) ) (true) (cards) ;;

	let allSpades   (): t list = List.map (fun x -> newCard (x) (Spade   ) ) (Value.all()) ;;
	let allHearts   (): t list = List.map (fun x -> newCard (x) (Heart   ) ) (Value.all()) ;;
	let allDiamonds (): t list = List.map (fun x -> newCard (x) (Diamond ) ) (Value.all()) ;;
	let allClubs    (): t list = List.map (fun x -> newCard (x) (Club    ) ) (Value.all()) ;;

	let all (): t list =
		let all_colors = Color.all() in
		let all_values = Value.all() in
		let all_tuples = cartesian (all_colors) (all_values) in
		let extract    = (fun x -> let (c, v) = x in newCard v c ) in
		let result     = List.map extract all_tuples in
		result
	;;


	let toString (card: t): string =
		Value.toString(card.value) ^ Color.toString(card.color)
	;;

	let toStringVerbose (card: t): string =
		"Card(" ^ Value.toStringVerbose(card.value) ^ ", " ^ Color.toStringVerbose(card.color) ^ ")"
	;;


	(* Going for the bridge order ♠ > ♥ > ♦ > ♣ ; and Value before Color *)
	let compare (card1: t) (card2: t): int =
		if      Value.toInt(card1.value) < Value.toInt(card2.value) then ( -1 )
		else if Value.toInt(card1.value) > Value.toInt(card2.value) then (  1 )
		else
		(
			if      Color.toInt(card1.color) < Color.toInt(card2.color) then ( -1 )
			else if Color.toInt(card1.color) > Color.toInt(card2.color) then (  1 )
			else                                                             (  0 )
		)
	;;


	let max (card1: t) (card2: t): t =
		let cmp = compare (card1) (card2) in
		if cmp >= 0 then card1 else card2
	;;

	let min (card1: t) (card2: t): t =
		let cmp = compare (card1) (card2) in
		if cmp <= 0 then card1 else card2
	;;


	let best (cards: t list): t =
		match cards with
		| []     -> invalid_arg "Card.best(): empty list"
		| h :: t -> List.fold_left (fun x y -> max (x) (y)) (h) (t)
	;;

end
