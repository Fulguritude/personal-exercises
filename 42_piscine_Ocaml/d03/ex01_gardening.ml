
(* https://ocaml.github.io/graphics/graphics/Graphics/index.html *)

open Graphics;;
(* open Ex00_graphics.Tree;; *)
open Ex00_graphics.Canvas;;


module Tree = struct
	(* include Ex00_graphics.Tree ;; *)
	type 'a tree = Nil | Node of 'a * 'a tree * 'a tree ;;


	let get_example_tree (): string tree =
		Node
		(
			"0",
			Node
			(
				"00",
				Node
				(
					"000",
					Nil,
					Node ("0001", Nil, Nil)
				),
				Node
				(
					"001",
					Node ("0010", Nil, Nil),
					Nil
				)
			),
			Node
			(
				"01",
				Node
				(
					"010",
					Nil,
					Nil
				),
				Node
				(
					"011",
					Node ("0110", Nil, Nil),
					Node ("0111", Nil, Nil)
				)
			)
		)
	;;


	let rec map_tree
		(tree   :  'a tree  )
		(b_of_a : ('a -> 'b))
	: 'b tree =
		match tree with
		| Nil -> Nil
		| Node (value, child1, child2) ->
		(
			Node
			(
				b_of_a (value),
				map_tree (child1) (b_of_a),
				map_tree (child2) (b_of_a)
			)
		)
	;;


	let stringtree_of_tree
		(tree            :  'a tree      )
		(string_of_alpha : ('a -> string))
	: string tree =
		map_tree (tree) (string_of_alpha)
	;;


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

end



module TreeCanvas = struct
	include Ex00_graphics.Canvas ;;

	type orientation = Vertical | Horizontal ;;

	let display_mode = Vertical;;
	let node_h       = 20 ;;
	let node_mid_h   = node_h / 2 ;;



	let draw_tree_node
		(value      : string )
		(rect_x     : int    )
		(rect_y     : int    )
		(rect_w     : int    )
		(rect_h     : int    )
		(str_size_x : int    )
		(str_size_y : int    )
	: unit =
		let rect_mid_w = rect_x + rect_w / 2 in
		let rect_mid_h = rect_y + rect_h / 2 in
		draw_rect (rect_x) (rect_y) (rect_w) (rect_h);
		let str_pos_x = rect_mid_w - str_size_x / 2 in
		let str_pos_y = rect_mid_h - str_size_y / 2 in
		moveto (str_pos_x) (str_pos_y);
		draw_string value;
	;;



	let draw_tree_edge_vertical
		(node         : string Tree.tree )
		(parent_right : int              )
		(parent_mid_h : int              )
		(child__left  : int              )
		(child__mid_h : int              )
	: unit =
		match node with
		| Nil -> ()
		| Node (_, _, _) ->
		(
			moveto (parent_right) (parent_mid_h);
			lineto (child__left ) (child__mid_h);
		)
	;;


	let draw_tree_edge_horizontal
		(node         : string Tree.tree )
		(parent_bot   : int              )
		(parent_mid_w : int              )
		(child__top   : int              )
		(child__mid_w : int              )
	: unit =
		match node with
		| Nil -> ()
		| Node (_, _, _) ->
		(
			moveto (parent_bot) (parent_mid_w);
			lineto (child__top) (child__mid_w);
		)
	;;


	let draw_tree_edge
		(node       : string Tree.tree )
		(parent_out : int              )
		(parent_mid : int              )
		(child__in  : int              )
		(child__mid : int              )
	: unit =
		match display_mode with
		| Horizontal ->
		(
			draw_tree_edge_horizontal
				(node       )
				(parent_out )
				(parent_mid )
				(child__in  )
				(child__mid )
		)
		| Vertical ->
		(
			draw_tree_edge_vertical
				(node       )
				(parent_mid )
				(parent_out )
				(child__mid )
				(child__in  )
		)
	;;


	let draw_tree (tree: string Tree.tree): unit =

		let rec rec_draw_tree
			(tree       : string Tree.tree )
			(root_pos   : int * int        )
			(split_dist : int              )
		: unit =
			match tree with
			| Nil -> ()
			| Node (value, child1, child2) ->
			(
				let (     pos_x,      pos_y) = root_pos in
				let (str_size_x, str_size_y) = text_size (value) in
				let (    rect_w,     rect_h) = (str_size_x + 10, node_h) in (* str_siz_y + 10) in *)
				let new_split_dist = split_dist / 2 in

				let parent_out = if display_mode = Horizontal then pos_x + rect_w     else pos_y + rect_h     in
				let parent_mid = if display_mode = Horizontal then pos_y + rect_h / 2 else pos_x + rect_w / 2 in

				let child1_in = parent_out   + 20             in
				let child2_in = child1_in                     in

				let child1_pos_other = if display_mode = Horizontal then pos_y + new_split_dist else pos_x + new_split_dist in
				let child2_pos_other = if display_mode = Horizontal then pos_y - new_split_dist else pos_x - new_split_dist in

				let child1_mid = parent_mid + new_split_dist in
				let child2_mid = parent_mid - new_split_dist in

				let child1_pos = if display_mode = Horizontal then (child1_in, child1_pos_other) else (child1_pos_other, child1_in) in
				let child2_pos = if display_mode = Horizontal then (child2_in, child2_pos_other) else (child2_pos_other, child2_in) in

				draw_tree_node
					(value      )
					(pos_x      )
					(pos_y      )
					(rect_w     )
					(rect_h     )
					(str_size_x )
					(str_size_y )
				;
				draw_tree_edge
					(child1     )
					(parent_out )
					(parent_mid )
					(child1_in  )
					(child1_mid )
				;
				draw_tree_edge
					(child2     )
					(parent_out )
					(parent_mid )
					(child2_in  )
					(child2_mid )
				;
				rec_draw_tree (child1) (child1_pos) (new_split_dist);
				rec_draw_tree (child2) (child2_pos) (new_split_dist);
			)
		in

		let init_split_dist = if display_mode = Horizontal then window_mid_h else window_mid_w in
		let root_pos =
			match display_mode with
			| Horizontal -> (20,               window_mid_h - node_h / 2 )
			| Vertical   -> (window_mid_w - 5, 20                        )
		in 
		rec_draw_tree (tree) (root_pos) (init_split_dist)
	;;

end



(* To test, uncomment the below *)
(* let example_tree = Canvas.get_example_tree();; *)
(* let render = fun () -> Canvas.draw_tree (example_tree);; *)
(* Canvas.run (render) ;; *)
