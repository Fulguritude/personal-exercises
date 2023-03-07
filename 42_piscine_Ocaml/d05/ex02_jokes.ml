(* let jokes = Array.init 5 (fun x -> "Joke " ^ string_of_int(x)) ;; *)

let jokes =
	[|
		"Alice";
		"Bob";
		"Charlie";
		"Derek";
		"Eve";
	|]
;;

Random.self_init() ;;

let joke = jokes.(Random.int 5) ;;
print_endline(joke);;
