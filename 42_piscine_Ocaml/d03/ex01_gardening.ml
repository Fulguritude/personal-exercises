
(* https://ocaml.github.io/graphics/graphics/Graphics/index.html *)

open Graphics;;
open Ex00_graphics.Tree;;
open Ex00_graphics.Canvas;;

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


module Canvas = struct
	include Ex00_graphics.Canvas ;;

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

	let rec stringtree_of_tree
		(tree            :  'a tree      )
		(string_of_alpha : ('a -> string))
	: string tree =
		match tree with
		| Nil -> Nil
		| Node (value, child1, child2) ->
		(
			Node
			(
				string_of_alpha (value),
				stringtree_of_tree (child1) (string_of_alpha),
				stringtree_of_tree (child2) (string_of_alpha)
			)
		)
	;;

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

	let draw_tree_edge
		(node         : string tree )
		(parent_right : int         )
		(parent_mid_h : int         )
		(child__left  : int         )
		(child__mid_h : int         )
	: unit =
		match node with
		| Nil -> ()
		| Node (_, _, _) ->
		(
			moveto (parent_right) (parent_mid_h);
			lineto (child__left ) (child__mid_h);
		)
	;;

	let draw_tree (tree: string tree): unit =

		let rec rec_draw_tree
			(tree       : string tree )
			(root_pos   : int * int   )
			(split_dist : int         )
		: unit =
			match tree with
			| Nil -> ()
			| Node (value, child1, child2) ->
			(
				let (     pos_x,      pos_y) = root_pos in
				let (str_size_x, str_size_y) = text_size (value) in
				let (    rect_w,     rect_h) = (str_size_x + 10, node_h) in (* str_siz_y + 10) in *)
				let new_split_dist = split_dist / 2 in

				let parent_right =        pos_x + rect_w         in
				let child1_pos_x = parent_right + 20             in
				let child2_pos_x = child1_pos_x                  in
				let parent_mid_h =        pos_y + node_h / 2     in
				let child1_pos_y =        pos_y + new_split_dist in
				let child2_pos_y =        pos_y - new_split_dist in
				let child1_mid_h = parent_mid_h + new_split_dist in
				let child2_mid_h = parent_mid_h - new_split_dist in
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
					(child1       )
					(parent_right )
					(parent_mid_h )
					(child1_pos_x )
					(child1_mid_h )
				;
				draw_tree_edge
					(child2       )
					(parent_right )
					(parent_mid_h )
					(child2_pos_x )
					(child2_mid_h )
				;
				rec_draw_tree (child1) ((child1_pos_x, child1_pos_y)) (new_split_dist);
				rec_draw_tree (child2) ((child2_pos_x, child2_pos_y)) (new_split_dist);
			)
		in

		let init_split_dist = window_mid_h in
		let root_pos        = (20, window_mid_h - node_h / 2) in
		rec_draw_tree (tree) (root_pos) (init_split_dist)
	;;

end


let example_tree = Canvas.get_example_tree();;
let render = fun () -> Canvas.draw_tree (example_tree);;

Canvas.run (render) ;;
