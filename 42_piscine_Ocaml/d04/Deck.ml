open Card;;


Random.self_init();;


let shuffle_list (l: 'a list): 'a list =
	List.sort (fun n m -> (Random.int 11) - 5) (l)
;;

module Deck = struct

	type t = Card.t list ;;

	let newDeck (): t =
		let all_cards = Card.all() in
		shuffle_list (all_cards)
	;;

	let toStringList        (deck: t): string list = List.map (Card.toString        ) (deck) ;;
	let toStringListVerbose (deck: t): string list = List.map (Card.toStringVerbose ) (deck) ;;

	let drawCard (deck: t): Card.t * t =
		match deck with
		| []     -> raise (Failure "drawCard(): deck is empty")
		| h :: t -> (h, t)
	;;

end
