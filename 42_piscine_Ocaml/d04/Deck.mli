val shuffle_list : 'a list -> 'a list
module Deck :
  sig
    type t = Card.Card.t list
    val newDeck : unit -> t
    val toStringList : t -> string list
    val toStringListVerbose : t -> string list
    val drawCard : t -> Card.Card.t * t
  end
