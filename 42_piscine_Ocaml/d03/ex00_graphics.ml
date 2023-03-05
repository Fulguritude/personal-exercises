(* https://dev.realworldocaml.org/files-modules-and-programs.html *)

(* https://ocaml.org/docs/first-hour#a-module-from-opam *)
(* https://stackoverflow.com/questions/48058993/error-unbound-module-in-ocaml *)
(* https://stackoverflow.com/questions/61741713/telling-ocaml-where-opam-packages-are *)
(* https://ocamlverse.net/content/quickstart_ocaml_project_dune.html *)

(* https://ocaml.github.io/graphics/graphics/Graphics/index.html *)
(* https://v2.ocaml.org/releases/4.01/htmlman/libgraph.html *)


open Graphics;;


module Tree = struct
	type 'a tree = Nil | Node of 'a * 'a tree * 'a tree ;;

	(* Graphics.open_graph *)
	(* Graphics.lineto *)
	(* Graphics.moveto *)
	(* Graphics.draw_string *)

	let draw_square
		(pos_x : int)
		(pos_y : int)
		(size  : int)
	: unit =
		let x_left  = pos_x        in
		let y_bot   = pos_y        in
		let x_right = pos_x + size in
		let y_top   = pos_y + size in
		moveto (x_left  ) (y_bot);
		lineto (x_right ) (y_bot);
		lineto (x_right ) (y_top);
		lineto (x_left  ) (y_top);
		lineto (x_left  ) (y_bot);
	;;

	let rec draw_tree_node (node : 'a tree): unit =
		let node_h = 50 in
		let x_pos = 20 in
		let y_pos = (window_h - node_h) / 2 in
		match node with
		| Nil ->
		(
			draw_square (x_pos) (y_pos) (node_h);
			moveto (x_pos + 5) (y_pos + node_h / 2 - 5);
			draw_string "Nil"
		)
		| Node (value, child1, child2) ->
		(
			draw_square (x_pos) (y_pos) (node_h);
			moveto (x_pos + 5) (y_pos + node_h / 2 - 5);
			draw_string "Value";
			moveto (x_pos + node_h      ) (y_pos + node_h / 2      );
			lineto (x_pos + node_h + 20 ) (y_pos + node_h / 2 + 30 );
			moveto (x_pos + node_h      ) (y_pos + node_h / 2      );
			lineto (x_pos + node_h + 20 ) (y_pos + node_h / 2 - 30 );
			draw_square (x_pos + node_h + 20) (y_pos + node_h / 2 + 10 ) (node_h);
			draw_square (x_pos + node_h + 20) (y_pos + node_h / 2 - 50 ) (node_h);
			moveto (x_pos + node_h + 20 + 5) (y_pos + node_h / 2 + 30 - 5 / 2); draw_string "Nil";
			moveto (x_pos + node_h + 20 + 5) (y_pos + node_h / 2 - 30 - 5 / 2); draw_string "Nil";

		)
	;;
end




let frame_duration = 1. /. 60. ;;
let window_w = 800 ;;
let window_h = 600 ;;

let string_of_window_size (w: int) (h: int): string =
	" " ^ string_of_int(w) ^ "x" ^ string_of_int(h)
;;


open_graph (string_of_window_size (window_w) (window_h));
set_color (rgb 0 0 0);

let node = Tree.Node ("Bob", Nil, Nil) in

(*
	Electing to go against the subject and use a 60 FPS game loop,
	because I'm not a fucking animal, and I respect myself, even
	if the authors don't. @:)
*)
while true do
	let loop_start = Sys.time() in

	(* Holy shit lmfao the (0,0) coordinate is at the BOTTOM left XDD *)
	(* draw_square (window_w / 2) (window_h / 2) (50) *)
	Tree.draw_tree_node(node);

	let loop_end = Sys.time() in

	let render_time   = loop_end -. loop_start in
	let leftover_time = frame_duration -. render_time in
	Unix.sleepf(leftover_time);
done;;
