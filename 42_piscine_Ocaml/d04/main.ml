open Deck;;

let deck = Deck.newDeck();;
let l    = Deck.toStringList(deck) ;;

List.map (fun s -> print_endline(s);) (l);;
