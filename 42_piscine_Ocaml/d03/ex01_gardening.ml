type 'a tree = Nil | Node of 'a * 'a tree * 'a tree ;;

let rec height (t: 'a tree): int =
	match t with
	| Nil -> 0
	| Node (_, child1, child2) -> 1 + max (height(child1)) (height(child2))
;;

let rec size (t: 'a tree): int =
	match t with
	| Nil -> 0
	| Node (_, child1, child2) -> 1 + size(child1) + size(child2)
;;
