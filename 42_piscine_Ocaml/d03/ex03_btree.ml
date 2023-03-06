open Ex01_gardening.TreeCanvas;;



let rec power (a: int) (b: int): int =
	match b with
	| 0 -> 1
	| 1 -> a
	| n ->
	(
		let b             = power a (n / 2)                 in
		let odd_power_val = (if n mod 2 == 0 then 1 else a) in
		b * b * odd_power_val
	)
;;



module BTree = struct

	include Ex01_gardening.Tree ;;


	let get_value (node: 'a tree): 'a option =
		match node with
		| Nil                -> None
		| Node (value, _, _) -> Some value
	;;


	let get_example_bst (): int tree =
		Node
		(
			8,
			Node
			(
				3,
				Node(1, Nil, Nil),
				Node
				(
					6,
					Node(4, Nil, Nil),
					Node(7, Nil, Nil)
				)
			),
			Node
			(
				10,
				Nil,
				Node
				(
					14,
					Node(13, Nil, Nil),
					Nil
				)
			)
		)
	;;


	let is_bst (tree: 'a tree): bool =
		let rec rec_is_bst
			(tree        : 'a tree   )
			(bound_lower : 'a option )
			(bound_upper : 'a option )
		: bool =
			let result =
				match tree with
				| Nil -> true
				| Node (value, child_l, child_r) ->
				(
					let old_b_l = bound_lower in
					let old_b_u = bound_upper in
					let new_b_l = match old_b_l with | Some b_l when value < b_l -> Some b_l | _ -> Some value in
					let new_b_u = match old_b_u with | Some b_u when value > b_u -> Some b_u | _ -> Some value in
					let result_l = rec_is_bst (child_l) (old_b_l) (new_b_u) in
					let result_r = rec_is_bst (child_r) (new_b_l) (old_b_u) in
					let result_current = 
						match (new_b_l, new_b_u) with
						| (None,     None     ) -> true
						| (Some b_l, None     ) -> b_l <= value
						| (None,     Some b_u ) ->                 value <= b_u
						| (Some b_l, Some b_u ) -> b_l <= value && value <= b_u
					in
					result_l && result_r && result_current
				)
			in
			result
		in

		rec_is_bst (tree) (None) (None)
	;;


	let is_perfect (tree: 'a tree): bool =
		let node_amount_plus_1   = (size (tree)) + 1 in
		let tree_height          = height (tree) in
		let expected_node_amount = power (2) (tree_height) in
		node_amount_plus_1 = expected_node_amount && is_bst(tree)
	;;


	let is_balanced_tree (tree: 'a tree): bool =
		let rec get_height_minmax (tree: 'a tree): int * int =
			match tree with
			| Nil -> (0, 0)
			| Node (_, child_l, child_r) ->
			(
				let min_l, max_l = get_height_minmax (child_l) in
				let min_r, max_r = get_height_minmax (child_r) in
				let true_min     = Stdlib.min (min_l) (min_r)  in
				let true_max     = Stdlib.max (max_l) (max_r)  in
				(1 + true_min, 1 + true_max)
			)
		in
		let min_height, max_height = get_height_minmax (tree) in
		max_height - min_height < 2
	;;


	let is_balanced (tree: 'a tree): bool =
		is_balanced_tree(tree) && is_bst(tree)
	;;


	let search_bst (bst: 'a tree) (search_val: 'a): bool =
		if not ( is_bst(bst) ) then failwith "cannot search: not a bst" else ();

		let rec rec_search_bst (bst: 'a tree) (search_val: 'a): bool =
			match bst with
			| Nil -> false
			| Node (value, child_l, child_r) ->
			(
				match value with
				| v when search_val < v -> rec_search_bst (child_l) (search_val)
				| v when search_val > v -> rec_search_bst (child_r) (search_val)
				| _                     -> true
			)
		in
		rec_search_bst (bst) (search_val)
	;;


	let insert_bst (bst: 'a tree) (new_val: 'a): 'a tree =
		if not ( is_bst(bst) ) then failwith "cannot insert: not a bst" else ();
		let rec rec_insert_bst (bst: 'a tree) (new_val: 'a): 'a tree =
			match bst with
			| Nil -> Node (new_val, Nil, Nil)
			| Node (value, child_l, child_r) ->
			(
				match value with
				| v when new_val <= v -> let new_child_l = rec_insert_bst (child_l) (new_val) in Node(value, new_child_l,     child_r)
				| _                   -> let new_child_r = rec_insert_bst (child_r) (new_val) in Node(value,     child_l, new_child_r)
			)
		in
		rec_insert_bst (bst) (new_val)
	;;


	let delete_bst (bst: 'a tree) (search_val: 'a): 'a tree =
		if not ( is_bst(bst) ) then failwith "cannot delete: not a bst" else ();

		let rec find_successor (bst: 'a tree) (curr_succ: 'a tree): 'a tree =
			let curr_succ_val =
				match curr_succ with
				| Node (v, _, _) -> v
				| _ -> failwith "find_successor bad impl"
			in
			match bst with
			| Nil -> curr_succ
			| Node (value, child_l, child_r) ->
			(
				if value <= curr_succ_val then ( find_successor (child_l) (bst)       )
				else                           ( find_successor (child_r) (curr_succ) )
			)
		in

		let rec rec_delete_bst (bst: 'a tree) (search_val: 'a): 'a tree =
			match bst with
			| Nil -> Nil
			| Node (value, child_l, child_r) ->
			(
				match value with
				| v when search_val < v -> let new_child_l = rec_delete_bst (child_l) (search_val) in Node(value, new_child_l,     child_r)
				| v when search_val > v -> let new_child_r = rec_delete_bst (child_r) (search_val) in Node(value,     child_l, new_child_r)
				| _ ->
				(
					match (child_l, child_r) with
					| ( Nil,                                   Nil ) -> Nil
					| (      Node (val_l, child_ll, child_lr), Nil ) -> Node (val_l, child_ll, child_lr)
					| ( Nil, Node (val_r, child_rl, child_rr)      ) -> Node (val_r, child_rl, child_rr)
					| _ ->
					(
						let succ                           = find_successor (child_r) (child_r) in
						let succ_val                       = match succ with | Node (v, _, _) -> v | _ -> failwith "delete_bst bad impl" in
						let succ_subtree_with_succ_removed = rec_delete_bst (child_r) (succ_val) in
						Node (succ_val, child_l, succ_subtree_with_succ_removed)
					)
				)
			)
		in
		rec_delete_bst (bst) (search_val)
	;;


	let bst_of_list (l: 'a list): 'a tree =
		let rec rec_bst_of_list (l: 'a list) (acc: 'a tree): 'a tree =
			match l with
			| []     -> acc
			| h :: t ->
			(
				let new_acc = insert_bst (acc) (h) in
				rec_bst_of_list (t) (new_acc)
			)
		in
		rec_bst_of_list (l) (Nil)
	;;
end


let not_bst        = BTree.Node(4, BTree.Node(6, Nil, Nil), Nil);;
let example_bst    = BTree.get_example_bst();;

let test_bst_false = BTree.is_bst(not_bst);;
let test_bst_true  = BTree.is_bst(example_bst);;

print_endline("Expected false, got " ^ string_of_bool(test_bst_false ));;
print_endline("Expected true,  got " ^ string_of_bool(test_bst_true  ));;



let bst_perfect_1 = BTree.Node(4, Nil, Nil);;
let bst_perfect_2 = BTree.Node(4, Node(2, Nil, Nil), Node(6, Nil, Nil));;
let bst_perfect_3 = BTree.Node(4, Node(2, Node(1, Nil, Nil), Node(3, Nil, Nil)), Node(6, Node(5, Nil, Nil), Node(7, Nil, Nil)));;
let bst_not_perfect = example_bst;;
let test_bst_false1 = BTree.is_perfect(not_bst);;
let test_bst_false2 = BTree.is_perfect(bst_not_perfect);;
let test_bst_true_1 = BTree.is_perfect(bst_perfect_1);;
let test_bst_true_2 = BTree.is_perfect(bst_perfect_2);;
let test_bst_true_3 = BTree.is_perfect(bst_perfect_3);;

print_endline("Expected false, got " ^ string_of_bool(test_bst_false1));;
print_endline("Expected false, got " ^ string_of_bool(test_bst_false2));;
print_endline("Expected true,  got " ^ string_of_bool(test_bst_true_1));;
print_endline("Expected true,  got " ^ string_of_bool(test_bst_true_2));;
print_endline("Expected true,  got " ^ string_of_bool(test_bst_true_3));;


let test_bst_false1 = BTree.is_balanced(not_bst);;
let test_bst_false2 = BTree.is_balanced(bst_not_perfect);; (* isn't balanced either, see node at value 10 *)
let test_bst_true_1 = BTree.is_balanced(bst_perfect_1);;
let test_bst_true_2 = BTree.is_balanced(bst_perfect_2);;
let test_bst_true_3 = BTree.is_balanced(bst_perfect_3);;

print_endline("Expected false, got " ^ string_of_bool(test_bst_false1));;
print_endline("Expected false, got " ^ string_of_bool(test_bst_false2));;
print_endline("Expected true,  got " ^ string_of_bool(test_bst_true_1));;
print_endline("Expected true,  got " ^ string_of_bool(test_bst_true_2));;
print_endline("Expected true,  got " ^ string_of_bool(test_bst_true_3));;


(* To test insert and delete, uncomment the below *)
(* Random.self_init();; *)
Random.init(421);;
let node_value_list  = List.init 20 (fun n -> n) ;;
let shuffled_list    = List.sort (fun n m -> (Random.int 10) - 5) (node_value_list) ;;
let bst              = BTree.bst_of_list(shuffled_list) ;;
let bst_after_del    = BTree.delete_bst (bst) (9) ;; (* needs to be before string conversion, which would fuck up the ordering *)
let example_tree     = Ex01_gardening.Tree.stringtree_of_tree (bst          ) (string_of_int) ;;
let example_tree_del = Ex01_gardening.Tree.stringtree_of_tree (bst_after_del) (string_of_int) ;;


(* let render = fun () -> draw_tree (example_tree);; *)
let render = fun () -> draw_tree (example_tree_del);;
run (render) ;;
