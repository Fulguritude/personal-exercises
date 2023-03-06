(* I'll finish this later *)

(*
module AVLTree = struct

	include Ex03_btree.BTree ;;

	type 'a avl_tree = (int * 'a) tree ;;


	let balance_avl (avl: 'a avl_tree): 'a avl_tree =
		let rec rec_balance_avl (avl: 'a avl_tree): int * 'a avl_tree =
			(* height of tree * 'a avl_tree *)
			match avl with
			| Nil -> (0, Nil)
			| Node (value, child_l, child_r) ->
			(
				let height_l, avl_l = rec_balance_avl (child_l) in
				let height_r, avl_r = rec_balance_avl (child_r) in
				let node_balance    = height_r - height_l in
				if  node_balance <= -2 || 2 <= node_balance then
				(

				)
				else
				(

				)
				let new_avl         = Node((node_balance, value), avl_l, avl_r) in
				let new_height      = 1 + max (height_l) (height_r) in
				(new_height, new_avl)
			)
		in
	;;


	let avl_of_bst (bst: 'a tree): 'a avl_tree =
		if not ( is_bst(bst) ) then failwith "cannot convert to avl: not a bst" else ();

		let rec rec_avl_of_bst (bst: 'a tree): int * 'a avl_tree =
			(* height of tree * 'a avl_tree *)
			match bst with
			| Nil -> (0, Nil)
			| Node (value, child_l, child_r) ->
			(
				let height_l, avl_l = rec_avl_of_bst (child_l) in
				let height_r, avl_r = rec_avl_of_bst (child_r) in
				let node_balance    = height_r - height_l in
				let avl             = Node((node_balance, value), avl_l, avl_r) in
				let new_height      = 1 + max (height_l) (height_r) in
				(new_height, avl)
			)
		in
		let _, result = rec_avl_of_bst (bst) in
		balance_avl (result)
	;;

	let insert_avl (avl: 'a avl_tree) (new_val: 'a): 'a avl_tree =
		if not ( is_bst(avl) ) then failwith "cannot insert in avl: not a bst" else ();

		let new_bst = insert_bst (avl) ((0, new_val)) in

	;;


end
*)